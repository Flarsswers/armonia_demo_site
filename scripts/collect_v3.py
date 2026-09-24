#!/usr/bin/env python3
"""
collect_v3.py — 用 target CLAP − source CLAP margin 重新筛选 demo 与主观听评案例。

筛选标准（师兄指示）:
    margin = S_target − S_source = CLAP(编辑音频, target prompt) − CLAP(编辑音频, source prompt)
    margin 越大 → 编辑越充分地把 source 语义转移到 target。

MusicDelta: 每个 (cat, seed, target) 在 4 个 ratio 中选 margin 最优；
            再取 margin 最高的 10 个不重复歌曲类别。
Melodia:    每个 (ds, cat, vel, target) 在 rho 中选 margin 最优；
            每个数据集先取 2 个不重复类别，再按 margin 补齐到 10。

输出:
    demo/demo_cases.json          （旧文件备份为 demo_cases_zscore_backup.json）
    ../questionnaire_cases.json   （听评：2 个 MusicDelta + 3 个 Melodia）

用法:
    python collect_v3.py
"""

import json
import shutil
from pathlib import Path
from collections import Counter, defaultdict

OUT_DIR = Path("/DATA4_4T/yzqr/quiz/demo")
MARGIN_DIR = OUT_DIR / "margins"

MUSICDELTA_BL = "/DATA6_6T/yy/musicdelta_results"
MUSICDELTA_SRC = "/DATA6_6T/yy/musicdelta_mix-wav_and_prompt"
MD_SRC_CSV = "/DATA6_6T/yy/AudioEditingCode-codeclean/MedleyMDPrompts/captions_sources.csv"
MELODIA_BL = "/DATA6_6T/yy/exp_results_models"
MELODIA_SRC = "/DATA6_6T/yy/MelodiaEdit"

MD_METHODS = ["sdedit", "ddpm", "ddim", "ddpm-100", "ddim-1000", "musicgen", "magus",
              "ours_melodia_(ours)"]

# 对照组查找：method key → (根目录, 子目录模板列表)
# {ds}=instrument/style/mood, {t}=T_STARTS, {cv}={cat}_{vel}
# 模板同时覆盖 generated 与 real 数据集的目录命名差异
ML_METHODS = {
    "sdedit": ("sdedit", ["generated_{ds}_t-start/{t}/{cv}/wavs/{tg}.wav",
                          "real_instrument_t-start/{t}/{cv}/wavs/{tg}.wav",
                          "real_style_t-start/{t}/{cv}/wavs/{tg}.wav"]),
    "sdedit_worse": ("sdedit_worse", ["generated_{ds}_t-start/{t}/{cv}/wavs/{tg}.wav"]),
    "ddpm": ("ddpm", ["generated_{ds}_t-start_new/{t}/{cv}/wavs/{tg}.wav",
                      "generated_{ds}_t-start/{t}/{cv}/wavs/{tg}.wav",
                      "real_instrument_new/{t}/{cv}/wavs/{tg}.wav",
                      "real_style_t-start_new/{t}/{cv}/wavs/{tg}.wav"]),
    "ddpm_worse": ("ddpm_worse", ["generated_{ds}_t-start_new/{t}/{cv}/wavs/{tg}.wav",
                                  "gen_{ds}_t-start_new/{t}/{cv}/wavs/{tg}.wav",
                                  "real_style_t-start_new/{t}/{cv}/wavs/{tg}.wav"]),
    "ddim_worse": ("ddim_worse", ["gen_{ds}_t-start_new/{t}/{cv}/wavs/{tg}.wav",
                                  "real_instrument_t-start_new/{t}/{cv}/wavs/{tg}.wav",
                                  "real_style_t-start_new/{t}/{cv}/wavs/{tg}.wav"]),
    "musicgen": ("musicgen", ["Musicgen_instrument_results/{t}/{cv}/wavs/{tg}.wav",
                              "instrument_real/{t}/{cv}/wavs/{tg}.wav",
                              "style_generate_2/{t}/{cv}/wavs/{tg}.wav",
                              "style_real/{t}/{cv}/wavs/{tg}.wav",
                              "mood_generate/{t}/{cv}/wavs/{tg}.wav"]),
    "magus": ("magus", ["instruments_generate/{t}/{cv}/wavs/{tg}.wav",
                        "instrument_real/{t}/{cv}/wavs/{tg}.wav",
                        "style_generate/{t}/{cv}/wavs/{tg}.wav",
                        "style_real/{t}/{cv}/wavs/{tg}.wav",
                        "mood_generate/{t}/{cv}/wavs/{tg}.wav"]),
    # Melodia 基线（demo 里显示为 "Melodia"）
    "ours_ours": ("ours", ["generated_t_start_instrument_exp_no_invert_prompt copy/{sub}/{t}/{cv}/wavs/{tg}.wav",
                           "real_instrument_exp_no_prompt_cfg5.5/{sub}/{t}/{cv}/wavs/{tg}.wav",
                           "generated_style_exp_new/{sub}/{t}/{cv}/wavs/{tg}.wav",
                           "generated_mood_exp/{sub}/{t}/{cv}/wavs/{tg}.wav",
                           "real_style_t_start_exp/{sub}/{t}/{cv}/wavs/{tg}.wav"]),
    "ours_ours_final": ("ours_final", ["generated_t_start_instrument_exp_no_invert_prompt copy/{sub}/{t}/{cv}/wavs/{tg}.wav",
                                       "real_instrument_exp/{sub}/{t}/{cv}/wavs/{tg}.wav",
                                       "generated_style_exp/{sub}/{t}/{cv}/wavs/{tg}.wav",
                                       "generated_mood_exp/{sub}/{t}/{cv}/wavs/{tg}.wav",
                                       "real_style_t_start_exp/{sub}/{t}/{cv}/wavs/{tg}.wav",
                                       "real_style_exp_no_prompt/{sub}/{t}/{cv}/wavs/{tg}.wav"]),
}
T_STARTS = ["100", "80", "120", "140", "60", "40", "160", "180", "200",
            "300", "400", "500", "600", "700", "800", "900", "1000"]
SUB_DIRS = ["su[4-9,17-24]", "empty"]

DS_SRC_MAP = {
    "Generated_Instrument": "Generated_Instrument_Dataset",
    "Generated_Style": "Generated_Style_Dataset",
    "Generated_Mood": "Generated_Mood_Dataset",
    "Real_Instrument": "Real_Instrument_Dataset",
    "Real_Style": "Real_Style_Dataset",
}
DS_BL_MAP = {
    "Generated_Instrument": "instrument",
    "Real_Instrument": "instrument",
    "Generated_Style": "style",
    "Real_Style": "style",
    "Generated_Mood": "mood",
}


def load_md_captions():
    """MusicDelta 每首歌的真实 source 描述（captions_sources.csv 变体 1）。"""
    import csv
    prompts = {}
    with open(MD_SRC_CSV) as f:
        for filename, caption in csv.reader(f):
            if filename == "filename":
                continue
            cat = filename.replace("_MIX.wav", "")
            prompts.setdefault(cat, caption.strip())
    return prompts


def load_margins():
    rows = []
    for p in sorted(MARGIN_DIR.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_md_baselines(cat, seed, target):
    baselines = {}
    for method in MD_METHODS:
        # Melodia 基线在 ours/8-14/1000/ 下的独立结构
        if method == "ours_melodia_(ours)":
            for tgt_var in [target, target.rstrip("."), target + "."]:
                wav_path = Path(
                    f"{MUSICDELTA_BL}/ours/8-14/1000/{cat}/{tgt_var}/wavs/{seed}.wav")
                if wav_path.exists():
                    baselines[method] = str(wav_path)
                    break
            continue
        for tgt_var in [target, target.rstrip("."), target + "."]:
            wav_path = Path(f"{MUSICDELTA_BL}/{method}/{cat}/{tgt_var}/wavs/{seed}.wav")
            if wav_path.exists():
                baselines[method] = str(wav_path)
                break
        else:
            mdir = Path(MUSICDELTA_BL) / method / cat
            if mdir.exists():
                tgt_clean = target.rstrip(".")
                for d in mdir.iterdir():
                    if not d.is_dir():
                        continue
                    dname = d.name.rstrip(".")
                    if tgt_clean.startswith(dname[:80]) or dname.startswith(tgt_clean[:80]):
                        wav_path = d / "wavs" / f"{seed}.wav"
                        if wav_path.exists():
                            baselines[method] = str(wav_path)
                            break
    return baselines


def find_ml_baselines(ds, cat_vel, target):
    ds_bl = DS_BL_MAP.get(ds, ds.lower())
    baselines = {}
    tg_variants = [target, target.replace(" ", "_")]
    if ds_bl == "mood" and target.startswith("a "):
        # magus 的 mood 结果文件名是 "a typical happy music.wav" 这类改写
        tg_variants.append("a typical " + target[2:])
    for method, (root_sub, templates) in ML_METHODS.items():
        found = False
        for tmpl in templates:
            subs = SUB_DIRS if "{sub}" in tmpl else [""]
            for sub in subs:
                for t in T_STARTS:
                    for tg_var in tg_variants:
                        wav_path = Path(MELODIA_BL) / root_sub / tmpl.format(
                            ds=ds_bl, sub=sub, t=t, cv=cat_vel, tg=tg_var)
                        if wav_path.exists():
                            baselines[method] = str(wav_path)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
    return baselines


def select_musicdelta(md_rows):
    # 1) 每个 (cat, seed, target) 选 margin 最优的 ratio
    best = {}
    for r in md_rows:
        key = (r["cat"], r["seed"], r["target"])
        if key not in best or r["margin"] > best[key]["margin"]:
            best[key] = r
    best_list = sorted(best.values(), key=lambda x: x["margin"], reverse=True)

    # 2) 取 10 个不重复类别
    seen_cats = set()
    top = []
    for r in best_list:
        if r["cat"] not in seen_cats:
            seen_cats.add(r["cat"])
            top.append(r)
        if len(top) >= 10:
            break

    # 基线对齐校验：任一基线的目录名与 target 不完全一致（例如超长 prompt 被
    # 生成脚本截断）的案例，换成下一个对齐干净的候选（类别不重复）。
    used_cats = set(r["cat"] for r in top)
    for i, r in enumerate(top):
        if md_baselines_aligned(r):
            continue
        for alt in best_list:
            if alt is r or alt["cat"] in used_cats:
                continue
            if md_baselines_aligned(alt):
                print(f"  [替换] {r['cat']}: {r['target'][:50]}... (margin={r['margin']:.3f}) "
                      f"基线目录名与 target 不一致，换成 {alt['cat']}: {alt['target'][:50]}... "
                      f"(margin={alt['margin']:.3f})")
                used_cats.discard(r["cat"])
                used_cats.add(alt["cat"])
                top[i] = alt
                break
    top.sort(key=lambda x: -x["margin"])

    print(f"\nMusicDelta 候选 pair 数: {len(best)}")
    print("  Best ratio distribution:", dict(Counter(r["ratio"] for r in top)))
    md_captions = load_md_captions()
    results = []
    for r in top:
        baselines = find_md_baselines(r["cat"], r["seed"], r["target"])
        caption = md_captions.get(r["cat"], "")
        src_prompt = caption if caption else f"[MusicDelta] {r['cat']}"
        results.append({
            "title": r["target"].rstrip(".").strip(),
            "source_wav": f"{MUSICDELTA_SRC}/{r['cat']}/{r['seed']}.wav",
            "source_prompt": src_prompt,
            "target_prompt": r["target"].rstrip(".").strip(),
            "our_wav": r["wav"],
            "ratio": r["ratio"],
            "baselines": baselines,
            "metrics": {"clap_score": r["clap_score"],
                        "src_clap": r["src_clap"],
                        "margin": r["margin"]},
        })
        print(f"    #{len(results)} [{r['ratio']}] {r['cat']}: {r['target'][:45]}... "
              f"margin={r['margin']:.3f} (S_t={r['clap_score']:.3f} S_s={r['src_clap']:.3f})")
    return results


def md_baselines_aligned(r) -> bool:
    """校验一个 MusicDelta 候选的所有基线：目录名必须与 target 完全一致（去尾点）、
    文件名必须是 seed。防止超长 prompt 被截断导致错位。"""
    tgt = r["target"].rstrip(".").strip()
    bl = find_md_baselines(r["cat"], r["seed"], r["target"])
    for k, v in bl.items():
        dname = Path(v).parent.parent.name.rstrip(".").strip()
        if dname != tgt:
            return False
        if Path(v).stem != r["seed"]:
            return False
    return True


def select_melodia(ml_rows):
    # 1) 每个 (ds, cat, vel, target) 选 margin 最优的 rho
    best = {}
    for r in ml_rows:
        key = (r["ds"], r["cat"], r["vel"], r["target"])
        if key not in best or r["margin"] > best[key]["margin"]:
            best[key] = r

    # 2) 每个数据集先取 2 个不重复类别，再按 margin 补齐到 10
    picks = []
    for ds_name in sorted({r["ds"] for r in best.values()}):
        ds_list = sorted([r for r in best.values() if r["ds"] == ds_name],
                         key=lambda x: x["margin"], reverse=True)
        seen = set()
        for r in ds_list:
            if r["cat"] not in seen:
                seen.add(r["cat"])
                picks.append(r)
            if len([p for p in picks if p["ds"] == ds_name]) >= 2:
                break
    all_sorted = sorted(best.values(), key=lambda x: x["margin"], reverse=True)
    for r in all_sorted:
        if len(picks) >= 10:
            break
        if r not in picks and not any(p["ds"] == r["ds"] and p["cat"] == r["cat"]
                                     for p in picks):
            picks.append(r)
    picks.sort(key=lambda x: x["margin"], reverse=True)

    # 对照组完整性兜底：按页面显示名要求 6 个对照齐全
    # （SDEdit / DDPM / DDIM / MusicGen / MusicMagus / Melodia）。
    # 优先换同类别同目标的其他 vel；都没有再换其他候选（类别不重复）。
    def _bl_keys(r):
        return find_ml_baselines(r["ds"], f"{r['cat']}_{r['vel']}", r["target"])

    def _display_set(keys):
        s = set()
        for k in keys:
            if k in ("sdedit", "sdedit_worse"):
                s.add("SDEdit")
            elif k in ("ddpm", "ddpm_worse"):
                s.add("DDPM")
            elif k == "ddim_worse":
                s.add("DDIM")
            elif k == "musicgen":
                s.add("MusicGen")
            elif k == "magus":
                s.add("MusicMagus")
            elif k.startswith("ours_"):
                s.add("Melodia")
        return s

    REQUIRED = {"SDEdit", "DDPM", "DDIM", "MusicGen", "MusicMagus", "Melodia"}
    used = {(p["ds"], p["cat"]) for p in picks}
    for i, p in enumerate(picks):
        cur = _display_set(_bl_keys(p))
        if REQUIRED <= cur:
            continue
        chosen = None
        # 第一优先：同 cat+target 的其他 vel
        same_ct = sorted(
            [r for r in best.values()
             if r["ds"] == p["ds"] and r["cat"] == p["cat"]
             and r["target"] == p["target"] and r is not p],
            key=lambda x: -x["margin"])
        for alt in same_ct:
            if REQUIRED <= _display_set(_bl_keys(alt)):
                chosen = alt
                break
        # 第二优先：其他类别候选（按 margin）
        if chosen is None:
            for alt in all_sorted:
                if alt is p or alt in picks:
                    continue
                if (alt["ds"], alt["cat"]) in used:
                    continue
                if REQUIRED <= _display_set(_bl_keys(alt)):
                    chosen = alt
                    break
        if chosen is not None:
            print(f"  [替换] {p['ds']}/{p['cat']}_{p['vel']}→{p['target']} "
                  f"(margin={p['margin']:.3f}, 缺 {sorted(REQUIRED - cur)})，换成 "
                  f"{chosen['ds']}/{chosen['cat']}_{chosen['vel']}→{chosen['target']} "
                  f"(margin={chosen['margin']:.3f}, 对照组齐全)")
            used.discard((p["ds"], p["cat"]))
            used.add((chosen["ds"], chosen["cat"]))
            picks[i] = chosen
    picks.sort(key=lambda x: x["margin"], reverse=True)

    print(f"\nMelodia 候选 pair 数: {len(best)}")
    print("  Dataset distribution:", dict(Counter(p["ds"] for p in picks)))
    results = []
    for r in picks:
        ds_src = DS_SRC_MAP.get(r["ds"], r["ds"])
        src_wav = resolve_ml_source(ds_src, f"{r['cat']}_{r['vel']}")
        baselines = find_ml_baselines(r["ds"], f"{r['cat']}_{r['vel']}", r["target"])
        results.append({
            "title": f"{r['cat']} → {r['target']}",
            "source_wav": src_wav,
            "source_prompt": src_prompt_of(r["ds"], r["cat"]),
            "target_prompt": r["target"],
            "our_wav": r["wav"],
            "baselines": baselines,
            "metrics": {"clap_score": r["clap_score"],
                        "src_clap": r["src_clap"],
                        "margin": r["margin"]},
            "dataset": r["ds"],
        })
        print(f"    #{len(results)} [{r['ds']}] {r['cat']}→{r['target']} "
              f"margin={r['margin']:.3f} (S_t={r['clap_score']:.3f} S_s={r['src_clap']:.3f})")
    return results


def resolve_ml_source(ds_src: str, cat_vel: str) -> str:
    """不同数据集的源音频目录结构不同：
    Generated_Style 是 {cat}_{vel}/origin.wav，其余是 {cat}_{vel}/origin_audio/audio.wav。"""
    base = Path(MELODIA_SRC) / ds_src / cat_vel
    for cand in (base / "origin_audio" / "audio.wav", base / "origin.wav"):
        if cand.exists():
            return str(cand)
    return str(base / "origin_audio" / "audio.wav")


def src_prompt_of(ds, cat):
    if "Instrument" in ds:
        return f"a solo {cat} music"
    if "Style" in ds:
        return f"a typical {cat} music"
    if "Mood" in ds:
        return f"a {cat} music"
    return f"[{ds}] {cat}"


def main():
    rows = load_margins()
    print(f"加载 margin 记录: {len(rows)}")
    md_rows = [r for r in rows if r["group"] == "musicdelta"]
    ml_rows = [r for r in rows if r["group"] == "melodia"]
    print(f"  musicdelta: {len(md_rows)}, melodia: {len(ml_rows)}")

    md = select_musicdelta(md_rows)
    ml = select_melodia(ml_rows)

    # ---- 保存 demo_cases.json（旧文件备份） ----
    out_json = OUT_DIR / "demo_cases.json"
    backup = OUT_DIR / "demo_cases_zscore_backup.json"
    if out_json.exists() and not backup.exists():
        shutil.copy2(out_json, backup)
        print(f"\n旧 demo_cases.json 已备份到 {backup.name}")
    with open(out_json, "w") as f:
        json.dump({"MusicDelta": md, "Melodia": ml}, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_json}")

    # ---- 听评案例：2 个 MusicDelta + 3 个 Melodia（按 margin 最高，类别不重复） ----
    q_cases = []
    seen_cats = set()
    for r in md:
        if len([c for c in q_cases if c["_group"] == "musicdelta"]) >= 2:
            break
        if r["title"] in seen_cats:
            continue
        seen_cats.add(r["title"])
        q_cases.append({"_group": "musicdelta", **r})
    ml_seen = set()
    for r in ml:
        if len([c for c in q_cases if c["_group"] == "melodia"]) >= 3:
            break
        if r["dataset"] in ml_seen:
            continue
        ml_seen.add(r["dataset"])
        q_cases.append({"_group": "melodia", **r})
    # 听评 baseline 精简：MusicDelta 用 SDEdit/DDPM/DDIM/MusicGen/MAGUS；Melodia 用 SDEdit/MusicGen/MAGUS
    for c in q_cases:
        if c["_group"] == "musicdelta":
            keep = {"sdedit": "SDEdit", "ddpm": "DDPM", "ddim": "DDIM",
                    "musicgen": "MusicGen", "magus": "MAGUS"}
        else:
            keep = {"sdedit": "SDEdit", "musicgen": "MusicGen", "magus": "MAGUS"}
        c["baselines"] = {keep[k]: v for k, v in c["baselines"].items() if k in keep}

    q_out = OUT_DIR.parent / "questionnaire_cases.json"
    with open(q_out, "w") as f:
        json.dump(q_cases, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {q_out}")
    for i, c in enumerate(q_cases, 1):
        print(f"  #{i} [{c['_group']}] {c['title'][:60]} margin={c['metrics']['margin']:.3f}")


if __name__ == "__main__":
    main()
