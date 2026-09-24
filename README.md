# Armonia — Zero-Shot Timbre Transfer Demo

Armonia 论文的 demo 网站（MusicDelta 10 案例 + Melodia 10 案例，A/B 对照听感）。

## 目录结构

```
armonia_demo_site/
├── index.html            # Demo 网站（静态页面，双击即可打开）
├── demo_cases.json       # 案例数据（含各方法的音频路径与指标）
├── 案例字典.md            # 每个案例的具体选取说明（音乐、prompt、margin、文件路径）
├── static/
│   ├── audio/            # 全部对比音频（source / ours / 各基线）
│   └── pipeline.png      # 方法示意图
├── scripts/              # 可复现脚本
│   ├── build_demo.py     # 由 demo_cases.json 生成 index.html
│   ├── collect_v3.py     # 按 margin = S_target − S_source 筛选案例
│   ├── compute_margins.py# 用 CLAP 计算每个候选的 margin
│   └── margins/          # 全部候选的 margin 明细（jsonl）
└── README.md
```

## 本地查看

直接双击 `index.html`，或用 HTTP 服务（避免部分浏览器对 file:// 下音频的限制）：

```bash
cd armonia_demo_site
python3 -m http.server 8900
# 浏览器打开 http://localhost:8900
```

## 部署到 GitHub Pages

1. 在 GitHub 新建仓库（如 `armonia-demo`）
2. 把本文件夹内容推到仓库主分支：

```bash
cd armonia_demo_site
git remote add origin https://github.com/<你的用户名>/armonia-demo.git
git branch -M main
git push -u origin main
```

3. 仓库 Settings → Pages → Source 选 `main` 分支、根目录 → Save
4. 几分钟后访问 `https://<你的用户名>.github.io/armonia-demo/`

## 案例筛选口径

- **筛选标准**：margin = S_target − S_source，即编辑后音频与 target prompt 的 CLAP 余弦相似度减去与 source prompt 的 CLAP 余弦相似度（margin 越大，目标属性转移越成功）
- **MusicDelta**：每个 (歌曲, 片段, target) 在 4 个 inversion ratio（0.4/0.5/0.6/0.8）中选 margin 最优者，再取 10 个不重复歌曲类别的最高 margin 案例
- **Melodia**：每个 (数据集, 类别, 样本, target) 在 rho sweep 中选 margin 最优者，每个数据集先取 2 个不重复类别，再按 margin 补齐到 10；对照组数据缺失的案例自动替换为 margin 次优且对照组齐全的候选
- 对照组：SDEdit / DDPM / DDIM / MusicGen / MusicMagus / Melodia（MusicDelta 另有 DDPM-Friendly、DDIM Inversion）

详见 `案例字典.md`。
