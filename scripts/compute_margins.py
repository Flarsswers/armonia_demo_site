#!/usr/bin/env python3
"""
compute_margins.py — 用 CLAP 计算每个候选编辑音频的
    margin = S_target − S_source

其中:
  S_target = cos(CLAP_audio(x_edit), CLAP_text(c_tgt))   ← 复用已有 clap_score
  S_source = cos(CLAP_audio(x_edit), CLAP_text(c_src))   ← 本脚本计算

使用与 metrics/eval_clap.py 完全相同的 CLAP 管线
（HTSAT-base + music_audioset_epoch_15_esc_90.14.pt, 48kHz 单声道, 纯 cosine）。

候选池:
  MusicDelta: 4 个 ratio 的全部 metrics_*.json（约 37K）
  Melodia:    rho sweep 全部 metrics_*.json，跳过 NoPGD（约 16K）

输出（每行一个候选，JSON Lines）:
  margins/musicdelta.jsonl
  margins/melodia.jsonl

用法:
    CUDA_VISIBLE_DEVICES=0 python compute_margins.py
"""

import csv
import json
import sys
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchaudio

# PyTorch >=2.6 默认 weights_only=True，无法加载旧格式 checkpoint。
# 该权重为本地可信文件（Meta CLAP 官方权重），patch 回 weights_only=False。
_torch_load_orig = torch.load
def _torch_load_patched(*a, **kw):
    kw.setdefault("weights_only", False)
    return _torch_load_orig(*a, **kw)
torch.load = _torch_load_patched

REPO = Path("/DATA1_4T/shared/AudioLDM2/A_ENM_Inversion")
for p in (str(REPO), str(REPO / "metrics")):
    if p not in sys.path:
        sys.path.insert(0, p)
from metrics.eval_clap import BatchCLAPEvaluator

# laion_clap 新版 factory.load_state_dict 无条件删除
# 'text_branch.embeddings.position_ids'，而当前 checkpoint 没有该键。
# 换成直接 torch.load + pop 默认值（与旧版行为一致）。
# 注意：eval_clap 以顶层名 "meta_clap_consistency" 导入，需同时 patch 两个模块名。
def _load_clap_state_dict_patched(clap_model, path):
    ckpt = torch.load(path, map_location="cpu")
    # 训练 checkpoint：权重包装在 state_dict 键下，且带 DataParallel 的 module. 前缀
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    ckpt = {k[7:] if k.startswith("module.") else k: v
            for k, v in ckpt.items()}
    ckpt.pop("text_branch.embeddings.position_ids", None)
    clap_model.model.load_state_dict(ckpt)
for _name in ("meta_clap_consistency", "metrics.meta_clap_consistency"):
    if _name in sys.modules:
        sys.modules[_name].load_clap_state_dict = _load_clap_state_dict_patched

OUT_DIR = Path("/DATA4_4T/yzqr/quiz/demo/margins")
CLAP_CKPT = "/DATA6_6T/yy/AudioEditingCode-codeclean_new/evals/music_audioset_epoch_15_esc_90.14.pt"

MD_BASE = REPO / "results/musicdelta"
MD_RATIOS = ["ratio0.4", "ratio0.5", "ratio0.6", "ratio0.8"]
MD_CFG = "K3_lr0.001_lamEdit1_v8"

ML_BASE = REPO / "results/melodia/melodia_results_attn_enm_v8_rho_sweep"

SRC_CSV = "/DATA6_6T/yy/AudioEditingCode-codeclean/MedleyMDPrompts/captions_sources.csv"
MD_SRC_DIR = "/DATA6_6T/yy/musicdelta_mix-wav_and_prompt"

TARGET_SR = 48000
BATCH = 32


# ---------------------------------------------------------------------------
# Source prompt 映射
# ---------------------------------------------------------------------------
def load_md_source_prompts():
    """MusicDelta: captions_sources.csv 变体 1；缺失时回退到 prompt.json 第一条 Original Prompt。"""
    prompts = {}
    with open(SRC_CSV) as f:
        for filename, caption in csv.reader(f):
            if filename == "filename":
                continue
            # 结果树类别名 = 完整目录名，如 "MusicDelta_80sRock"
            cat = filename.replace("_MIX.wav", "")
            prompts.setdefault(cat, caption.strip())
    # 回退：结果树里可能有的类别不在 csv 中
    for cat_dir in sorted(Path(MD_SRC_DIR).iterdir()):
        cat = cat_dir.name
        if cat in prompts:
            continue
        pj = cat_dir / "prompt.json"
        if pj.exists():
            try:
                data = json.loads(pj.read_text())
                prompts[cat] = data[0]["Original Prompt"]
            except Exception:
                pass
    return prompts


def ml_source_prompt(ds: str, cat: str) -> str:
    if "Instrument" in ds:
        return f"a solo {cat} music"
    if "Style" in ds:
        return f"a typical {cat} music"
    if "Mood" in ds:
        return f"a {cat} music"
    return f"[{ds}] {cat}"


# ---------------------------------------------------------------------------
# 候选收集
# ---------------------------------------------------------------------------
def collect_candidates():
    md_prompts = load_md_source_prompts()
    candidates = []   # (group, prompt, wav, s_target, meta)
    n_skip = 0

    # ---- MusicDelta ----
    for ratio in MD_RATIOS:
        root = MD_BASE / f"musicdelta_results_attn_enm_v8_no_src_{ratio}" / MD_CFG
        if not root.exists():
            continue
        for mfp in sorted(root.rglob("metrics_*.json")):
            rel = mfp.relative_to(root)
            parts = rel.parts
            if len(parts) < 3:
                continue
            cat, seed = parts[0], parts[1]
            target = mfp.stem[len("metrics_"):]
            wav = mfp.parent.parent / f"{target}.wav"
            if not wav.exists():
                n_skip += 1
                continue
            try:
                s_t = json.loads(mfp.read_text())["metrics"]["clap_score"]
            except Exception:
                n_skip += 1
                continue
            src_prompt = md_prompts.get(cat)
            if not src_prompt:
                n_skip += 1
                continue
            meta = {"group": "musicdelta", "ratio": ratio, "cat": cat,
                    "seed": seed, "target": target, "wav": str(wav)}
            candidates.append((src_prompt, str(wav), float(s_t), meta))

    # ---- Melodia ----
    for mfp in sorted(ML_BASE.rglob("metrics_*.json")):
        rel = mfp.relative_to(ML_BASE)
        parts = rel.parts
        if len(parts) < 4 or "NoPGD" in parts[1]:
            continue
        ds, rho, cat_vel = parts[0], parts[1], parts[2]
        target = mfp.stem[len("metrics_"):]
        cat = cat_vel.rsplit("_", 1)[0] if "_" in cat_vel else cat_vel
        wav = mfp.parent.parent / "wavs" / f"{target}.wav"
        if not wav.exists():
            n_skip += 1
            continue
        try:
            s_t = json.loads(mfp.read_text())["metrics"]["clap_score"]
        except Exception:
            n_skip += 1
            continue
        src_prompt = ml_source_prompt(ds, cat)
        meta = {"group": "melodia", "rho": rho, "ds": ds, "cat": cat,
                "vel": cat_vel.rsplit("_", 1)[1] if "_" in cat_vel else "",
                "target": target, "wav": str(wav)}
        candidates.append((src_prompt, str(wav), float(s_t), meta))

    print(f"候选总数: {len(candidates)}  (跳过缺失: {n_skip})")
    return candidates


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_md = OUT_DIR / "musicdelta.jsonl"
    out_ml = OUT_DIR / "melodia.jsonl"
    out_md.touch()
    out_ml.touch()

    done = set()
    for p in (out_md, out_ml):
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["key"])
            except Exception:
                pass
    print(f"已完成断点: {len(done)}")

    candidates = collect_candidates()
    candidates = [c for c in candidates
                  if _cand_key(c[3]) not in done]
    print(f"待计算: {len(candidates)}")

    # ---- CLAP 模型 ----
    print("加载 CLAP 模型 …")
    evaluator = BatchCLAPEvaluator(device="cuda", model_path=CLAP_CKPT)
    model = evaluator.model
    tokenizer = evaluator._tokenizer

    # ---- 文本 embedding（按唯一 source prompt 缓存） ----
    uniq_prompts = list(OrderedDict.fromkeys(c[0] for c in candidates))
    print(f"唯一 source prompt: {len(uniq_prompts)}")
    text_emb = {}
    with torch.no_grad():
        for i in range(0, len(uniq_prompts), 256):
            chunk = uniq_prompts[i:i + 256]
            emb = model.get_text_embedding(chunk, tokenizer=tokenizer,
                                           use_tensor=True).cpu()
            for p, e in zip(chunk, emb):
                text_emb[p] = e
    print("文本 embedding 完成")

    # ---- 音频 embedding + margin（分批） ----
    f_md = open(out_md, "a")
    f_ml = open(out_ml, "a")
    try:
        for i in range(0, len(candidates), BATCH):
            chunk = candidates[i:i + BATCH]
            audios = []
            for src_prompt, wav, s_t, meta in chunk:
                try:
                    w, sr = torchaudio.load(wav)
                except Exception:
                    continue
                if w.abs().sum() < 1e-6:
                    continue
                w = w.mean(dim=0, keepdim=True)
                if sr != TARGET_SR:
                    w = torchaudio.functional.resample(w, sr, TARGET_SR)
                audios.append((w, src_prompt, s_t, meta))
            if not audios:
                continue
            max_len = max(a[0].shape[1] for a in audios)
            wav_batch = torch.zeros(len(audios), max_len)
            for j, (w, _, _, _) in enumerate(audios):
                wav_batch[j, :w.shape[1]] = w
            with torch.no_grad():
                audio_emb = model.get_audio_embedding_from_data(
                    wav_batch.to("cuda"), use_tensor=True)
                for j, (_, src_prompt, s_t, meta) in enumerate(audios):
                    te = text_emb[src_prompt].to("cuda")
                    s_src = float(F.cosine_similarity(
                        audio_emb[j:j + 1], te.unsqueeze(0),
                        dim=1, eps=1e-8)[0])
                    rec = dict(meta)
                    rec["key"] = _cand_key(meta)
                    rec["clap_score"] = s_t
                    rec["src_clap"] = round(s_src, 4)
                    rec["margin"] = round(s_t - s_src, 4)
                    f = f_md if meta["group"] == "musicdelta" else f_ml
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_md.flush()
            f_ml.flush()
            if (i // BATCH) % 50 == 0:
                print(f"  进度: {min(i + BATCH, len(candidates))}/{len(candidates)}")
    finally:
        f_md.close()
        f_ml.close()
    print("完成")


def _cand_key(meta: dict) -> str:
    g = meta["group"]
    if g == "musicdelta":
        return f"md|{meta['ratio']}|{meta['cat']}|{meta['seed']}|{meta['target']}"
    return f"ml|{meta['rho']}|{meta['ds']}|{meta['cat']}|{meta['vel']}|{meta['target']}"


if __name__ == "__main__":
    main()
