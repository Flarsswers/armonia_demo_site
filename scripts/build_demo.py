#!/usr/bin/env python3
"""Build the Polyphonia-style demo HTML page from demo_cases.json."""

import json, shutil, os
from pathlib import Path

OUTPUT_DIR = Path("/DATA4_4T/yzqr/quiz/demo")
AUDIO_DIR = Path("/DATA4_4T/yzqr/quiz/demo/static/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_DIR / "demo_cases.json") as f:
    data = json.load(f)

# Baseline display names
BL_NAMES = {
    "sdedit": "SDEdit",
    "sdedit_worse": "SDEdit",
    "ddpm": "DDPM",
    "ddpm_worse": "DDPM",
    "ddim": "DDIM",
    "ddim_worse": "DDIM",
    "ddpm-100": "DDPM-Friendly",
    "ddim-1000": "DDIM Inversion",
    "musicgen": "MusicGen",
    "magus": "MusicMagus",
    "ours_melodia_(ours)": "Melodia",
    "ours_melodia_(ours_final)": "Melodia",
    "ours_ours": "Melodia",
    "ours_ours_final": "Melodia",
}

# For MusicDelta: skip ddim-1000 and ddpm plain, keep ddim and ddpm-100
# For Melodia: keep all
MD_SKIP = {"ddim-1000", "ddpm"}

# Preferred order
BL_DISPLAY_ORDER = ["sdedit", "ddpm", "ddpm-100", "ddim", "ddim-1000", "musicgen", "magus",
                    "ours_melodia_(ours)", "ours_melodia_(ours_final)",
                    "ours_ours", "ours_ours_final"]
# Fallback _worse methods
BL_FALLBACK = {"sdedit_worse": "sdedit", "ddpm_worse": "ddpm", "ddim_worse": "ddim"}

# Copy audio files（始终覆盖：文件名按案例位置编号，案例列表更新后必须替换旧文件）
def copy_audio(src, label):
    if not src or not os.path.exists(src):
        return None
    dst = AUDIO_DIR / f"{label}.wav"
    shutil.copy2(src, dst)
    return f"static/audio/{label}.wav"

copied = 0
for dataset, cases in data.items():
    for i, case in enumerate(cases):
        slug = f"{dataset}_{i+1}".lower().replace(" ", "_")
        case["_src_audio"] = copy_audio(case.get("source_wav"), f"{slug}_source")
        case["_our_audio"] = copy_audio(case.get("our_wav"), f"{slug}_ours")
        if case["_src_audio"]: copied += 1
        if case["_our_audio"]: copied += 1

        case["_bl_audio"] = {}
        for method, wav_path in case.get("baselines", {}).items():
            audio = copy_audio(wav_path, f"{slug}_{method}")
            case["_bl_audio"][method] = audio
            if audio: copied += 1

print(f"Copied {copied} audio files")

# Generate HTML
def make_case_card(case, dataset_idx, case_idx):
    title = case["title"][:100]
    src_audio = case.get("_src_audio", "")
    our_audio = case.get("_our_audio", "")

    # Source prompt / target prompt
    sp = case.get("source_prompt", "")
    tp = case.get("target_prompt", "")

    baseline_items = []
    # Our method first
    if our_audio:
        baseline_items.append(f"""<div class="audio-item is-highlight-red">
  <h5 class="audio-item-title"><i class="fas fa-star"></i>Armonia (Ours)</h5>
  <audio controls src="{our_audio}"></audio>
</div>""")

    # Then baselines — deduplicate: prefer non-worse, fallback to worse
    shown_names = set()
    shown_names.add("Armonia (Ours)")
    is_musicdelta = dataset_idx == 0
    for method in BL_DISPLAY_ORDER:
        # Skip ddim/ddpm plain for MusicDelta
        if is_musicdelta and method in MD_SKIP:
            continue
        bl_audio = case.get("_bl_audio", {})
        # Prefer regular method
        if method in bl_audio and bl_audio[method]:
            name = BL_NAMES[method]
            if name not in shown_names:
                shown_names.add(name)
                baseline_items.append(f"""<div class="audio-item">
  <h5 class="audio-item-title">{name}</h5>
  <audio controls src="{bl_audio[method]}"></audio>
</div>""")
        else:
            # Try fallback (_worse variant)
            for fw, reg in BL_FALLBACK.items():
                if reg == method and fw in bl_audio and bl_audio[fw]:
                    name = BL_NAMES[fw]
                    if name not in shown_names:
                        shown_names.add(name)
                        baseline_items.append(f"""<div class="audio-item">
  <h5 class="audio-item-title">{name}</h5>
  <audio controls src="{bl_audio[fw]}"></audio>
</div>""")
                    break

    bl_html = "\n".join(baseline_items)

    return f"""
      <div class="example-card">
        <h4 class="example-title"><code>{title}</code></h4>
        <div class="prompt-text">
          <p><strong>Source Description:</strong> {sp}</p>
          <p><strong>Target Prompt:</strong> {tp}</p>
        </div>
        <div class="source-audio">
          <h5>Source Audio</h5>
          <audio controls src="{src_audio}"></audio>
        </div>
        <div class="methods-grid">
          {bl_html}
        </div>
      </div>"""


# Build tab content
md_cards = "\n".join(make_case_card(c, 0, i) for i, c in enumerate(data.get("MusicDelta", [])))
ml_cards = "\n".join(make_case_card(c, 1, i) for i, c in enumerate(data.get("Melodia", [])))

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Armonia: Zero-Shot Timbre Transfer Demo</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fira+Code&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    :root {{
      --bg-color: #ffffff; --content-bg: #ffffff; --border-color: #e9ecef;
      --text-primary: #212529; --text-secondary: #6c757d;
      --accent-purple: #a29cff; --accent-purple-light: #cbbdff;
      --highlight-purple-bg: linear-gradient(145deg, rgba(162,156,255,0.05), rgba(162,156,255,0.01));
      --highlight-purple-border: #dcdcff;
      --accent-blue: #87CEEB; --accent-blue-light: #b0e0e6;
      --highlight-blue-bg: linear-gradient(145deg, rgba(135,206,235,0.05), rgba(135,206,235,0.01));
      --highlight-blue-border: #cfe8ff;
    }}
    html {{ scroll-behavior: smooth; }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg-color); color: var(--text-primary); line-height: 1.7; margin: 0;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 0 24px; }}
    .section {{ padding: 4.5rem 0; }}
    .hero {{ text-align: center; padding: 4rem 0 3rem 0; }}
    .hero h1 {{ font-size: 3rem; font-weight: 800; margin-bottom: 16px; }}
    .hero p {{ font-size: 1.5rem; font-weight: 400; margin-bottom: 32px; }}
    .project-title {{ font-size: 4.5rem; font-weight: 800; letter-spacing: -2px; margin-bottom: 0.5rem; background: linear-gradient(45deg, var(--text-primary), #495057); -webkit-background-clip: text; background-clip: text; color: transparent; }}
    .project-subtitle {{ font-size: 1.5rem; font-weight: 400; color: var(--text-secondary); margin-top: 1rem; margin-bottom: 2.5rem; max-width: 800px; margin-left: auto; margin-right: auto; }}
    .links {{ display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin-top: 2rem; }}
    .links a {{
      display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.8rem 1.5rem;
      border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 1rem;
      border: 1px solid var(--border-color); color: var(--text-primary);
      transition: all 0.3s ease;
    }}
    .links a:hover {{ background: #f8f9fa; border-color: var(--accent-purple); }}
    .abstract {{ max-width: 800px; margin: 2.5rem auto 0 auto; text-align: justify; font-size: 1.1rem; color: #343a40; }}
    .pipeline-figure {{ max-width: 1000px; margin: 2.5rem auto 0 auto; }}
    .pipeline-figure img {{ width: 100%; border-radius: 12px; border: 1px solid var(--border-color); box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
    .tabs {{ display: flex; justify-content: center; margin-bottom: 3rem; border-bottom: 1px solid var(--border-color); flex-wrap: wrap; }}
    .tab-button {{
      padding: 1rem 1.5rem; cursor: pointer; border: none; background: none;
      font-size: 1.1rem; font-weight: 600; color: var(--text-secondary);
      border-bottom: 3px solid transparent; transition: color 0.3s, border-color 0.3s; margin-bottom: -1px;
    }}
    .tab-button:hover {{ color: var(--text-primary); }}
    .tab-button.active {{ color: var(--accent-blue); border-bottom-color: var(--accent-blue); }}
    .tab-content {{ display: none; }}
    .tab-content.active {{ display: block; }}
    .example-card {{
      background-color: var(--content-bg); border: 1px solid var(--border-color);
      border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 3rem;
      box-shadow: 0 4px 12px rgba(0,0,0,0.03); transition: box-shadow 0.3s ease, transform 0.3s ease;
    }}
    .example-card:hover {{ box-shadow: 0 10px 30px rgba(0,0,0,0.06); transform: translateY(-4px); }}
    .example-title {{ font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem; }}
    .example-title code {{
      font-family: 'Fira Code', monospace; background-color: #e9ecef;
      padding: 3px 8px; border-radius: 6px; color: var(--accent-blue); font-weight: 500;
    }}
    .prompt-text {{ margin-bottom: 1.5rem; }}
    .prompt-text p {{ margin-bottom: 0.3rem; }}
    .source-audio {{
      background-color: #e9ecef; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;
    }}
    .source-audio h5 {{ margin-top: 0; }}
    .methods-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem;
    }}
    .audio-item {{
      display: flex; flex-direction: column; background-color: #f8f9fa;
      padding: 1rem; border-radius: 12px; border: 1px solid var(--border-color);
    }}
    .audio-item.is-highlight-red {{
      background: var(--highlight-purple-bg); border-color: var(--highlight-purple-border);
      box-shadow: 0 0 20px rgba(162,156,255,0.15);
    }}
    .audio-item-title {{ font-weight: 600; margin: 0 0 0.75rem 0; color: var(--text-primary); font-size: 0.9rem; }}
    audio {{ width: 100%; border-radius: 50px; }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="container">
        <h1 class="project-title">Armonia</h1>
        <p class="project-subtitle">Zero-Shot Timbre Transfer in Polyphonic Music</p>
        <div class="links">
          <a href="#"><i class="fas fa-file-pdf"></i> Paper</a>
          <a href="#"><i class="fas fa-code"></i> Code</a>
          <a href="#"><i class="fab fa-github"></i> GitHub</a>
        </div>
        <div class="abstract">
          <p>Text-guided music editing aims to modify specified attributes of a music while preserving musical content unrelated to the intended change, enabling flexible music creation.
          Inversion anchors this editing process to the source music by mapping it to a state that initializes generation under the target text.
          However, a state that enables faithful source reconstruction is not necessarily a suitable starting point for the requested edit.
          This distinction motivates us to move part of the editing control into inversion and prepare an initialization that is better suited to the requested edit.
          To this end, we propose <em>Armonia</em>, a training-free method that refines inversion states using output-level attention distillation derived from source- and target-conditioned model responses.
          To improve refinement efficiency and effectiveness while reducing the risk of off-manifold drift, Armonia uses temporal weighting to focus optimization on selected inversion steps and local projection to limit deviations from each DDIM inversion proposal.
          The resulting state is then used to initialize the original target-conditioned denoising process, leaving the pretrained diffusion model unchanged.
          Compared with SOTA, Armonia achieves an average relative improvement of 11.5% in CLAP across three datasets while retaining competitive structural preservation.</p>
        </div>
        <div class="pipeline-figure">
          <img src="static/pipeline.png" alt="Armonia pipeline">
        </div>
      </div>
    </section>

    <section class="section">
      <div class="container">
        <div class="tabs">
          <button class="tab-button active" onclick="openTab(event, 'musicdelta')">MusicDelta Dataset</button>
          <button class="tab-button" onclick="openTab(event, 'melodia')">Melodia Dataset</button>
        </div>

        <div id="musicdelta" class="tab-content active">
          {md_cards}
        </div>

        <div id="melodia" class="tab-content">
          {ml_cards}
        </div>
      </div>
    </section>
  </main>

  <script>
    function openTab(evt, tabName) {{
      document.querySelectorAll(".tab-content").forEach(t => t.style.display = "none");
      document.querySelectorAll(".tab-button").forEach(b => b.classList.remove("active"));
      document.getElementById(tabName).style.display = "block";
      evt.currentTarget.classList.add("active");
    }}
    document.addEventListener("DOMContentLoaded", () => {{
      document.querySelector(".tab-button").click();
    }});
  </script>
</body>
</html>"""

out = OUTPUT_DIR / "index.html"
out.write_text(html)
print(f"Generated: {out}")
print(f"  {len(data.get('MusicDelta',[]))} MusicDelta + {len(data.get('Melodia',[]))} Melodia cases")
print(f"  Audio: {AUDIO_DIR}")
