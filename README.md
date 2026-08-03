<!--
  Suhas Reddy · GitHub Profile README
  · primary accent: #64ffda (mint green)
  · vision → #22d3ee · agents → #c4b5fd · energy → #fb923c · data → #fda4af
  · numbered sections · year-prefixed projects · left-aligned · whitespace as structure
-->

<br/>

# Hi, I'm Suhas

<sub>Applied AI engineer. MS Data Science at Arizona State. Tempe, AZ.</sub>

<br/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=500&size=18&duration=4000&pause=2000&color=A7F3D0&width=320&lines=let+the+data+lead." alt="let the data lead." />

<br/>

[sdonthi4@asu.edu](mailto:sdonthi4@asu.edu) &nbsp;·&nbsp; [LinkedIn](https://linkedin.com/in/suhas-reddy-msds/) &nbsp;·&nbsp; [GitHub](https://github.com/Suhxs-Reddy)

<br/>
<br/>

## <sub><code>01.</code></sub> &nbsp;About

The data is always dirtier than you think. The deployment is always weirder than you imagined. Generic models miss the obvious because nobody told them what was obvious.

I work at the seam between AI and the world it has to live in — where 90 traffic cameras include eleven 320×240 relics from the early 2000s, where pipeline incident data from 1986 changes where you'd put a gigawatt plant in 2026, where the privacy policy nobody reads is also the one selling your location.

**The fun part isn't tuning the model. It's noticing what the model can't see.**

<br/>
<br/>

## <sub><code>02.</code></sub> &nbsp;Work

<br/>

<img src="https://capsule-render.vercel.app/api?type=transparent&color=22d3ee&height=2&fontColor=22d3ee&text=." width="100%" />

### [`'26` &nbsp; CATI](https://github.com/Suhxs-Reddy/sg-smart-city-analytics) &nbsp; <img src="https://img.shields.io/badge/computer%20vision-22d3ee?style=flat-square&labelColor=0d1117" /> &nbsp; <img src="https://img.shields.io/badge/live-22d3ee?style=flat-square&labelColor=0d1117" />

<sub>Context-Aware Traffic Intelligence · Singapore · solo build</sub>

Singapore runs 90 expressway cameras in rain, haze, and 4am dark — all piped through the same generic detector. CATI injects **FiLM layers** ([Perez et al., 2018](https://arxiv.org/abs/1709.07871)) into YOLOv11's backbone so the model re-wires itself at inference time based on live weather, PM2.5, hour, and camera ID. One model for all 90 cameras. 130K extra parameters on 9.4M.

<table><tr>
<td align="center"><h3>90</h3><sub>live LTA cameras</sub></td>
<td align="center"><h3>1.4%</h3><sub>parameter overhead</sub></td>
<td align="center"><h3>213K+</h3><sub>detection records on HF</sub></td>
<td align="center"><h3>90s</h3><sub>inference interval</sub></td>
</tr></table>

`PyTorch` &nbsp;·&nbsp; `YOLOv11` &nbsp;·&nbsp; `FiLM` &nbsp;·&nbsp; `Grounding DINO` &nbsp;·&nbsp; `BoT-SORT` &nbsp;·&nbsp; `Docker` &nbsp;·&nbsp; `HuggingFace Spaces`

[**Repo →**](https://github.com/Suhxs-Reddy/sg-smart-city-analytics) &nbsp;·&nbsp; [**Live demo →**](https://huggingface.co/spaces/SuhxsReddy/SingaporeAnalytics) &nbsp;·&nbsp; [**Dataset →**](https://huggingface.co/datasets/SuhxsReddy/cati-singapore-dataset) &nbsp;·&nbsp; [**Model →**](https://huggingface.co/SuhxsReddy/cati-singapore)

<br/>
<br/>

<img src="https://capsule-render.vercel.app/api?type=transparent&color=fb923c&height=2&fontColor=fb923c&text=." width="100%" />

### [`'26` &nbsp; COLLIDE](https://github.com/Suhxs-Reddy/EnergyHackathon) &nbsp; <img src="https://img.shields.io/badge/LLM%20agent-c4b5fd?style=flat-square&labelColor=0d1117" /> &nbsp; <img src="https://img.shields.io/badge/energy-fb923c?style=flat-square&labelColor=0d1117" />

<sub>AI siting for gigawatt data centers · ASU Energy Hackathon 2026 · solo build</sub>

Where do you build a 100 MW behind-the-meter gas plant when connecting to the grid takes 3–7 years? Three ML models score land, gas pipeline risk, and power market dynamics independently, then TOPSIS arbitrates. A 7-node LangGraph agent on top of Claude answers what-ifs — *"What if gas spikes 40%?"* — against live ERCOT/CAISO feeds. Monte Carlo gives P10/P50/P90 over 20 years.

<table><tr>
<td align="center"><h3>50–500</h3><sub>MW per site</sub></td>
<td align="center"><h3>10</h3><sub>live public data sources</sub></td>
<td align="center"><h3>10K</h3><sub>Monte Carlo scenarios</sub></td>
<td align="center"><h3>5 min</h3><sub>market data refresh</sub></td>
</tr></table>

`FastAPI` &nbsp;·&nbsp; `LangGraph` &nbsp;·&nbsp; `Claude` &nbsp;·&nbsp; `DuckDB` &nbsp;·&nbsp; `SHAP` &nbsp;·&nbsp; `Random Forest` &nbsp;·&nbsp; `KDE` &nbsp;·&nbsp; `GMM` &nbsp;·&nbsp; `Monte Carlo`

[**Repo →**](https://github.com/Suhxs-Reddy/EnergyHackathon)

<br/>
<br/>
<br/>

## <sub><code>03.</code></sub> &nbsp;Also built

<br/>

**[DataGuard](https://github.com/Suhxs-Reddy/KiroHackathon)** &nbsp; `'25` &nbsp; — &nbsp; Chrome extension that lets Llama 3.1 read the privacy policy for you. Breach-checks the domain and generates one-click GDPR/CCPA opt-outs.

**[AI Study Buddy](https://github.com/Suhxs-Reddy/CBC-Hackathon)** &nbsp; `'25` &nbsp; — &nbsp; RAG tutor that retrieves only from your real Canvas docs and flags out-of-syllabus questions instead of hallucinating answers.

<br/>
<br/>
<br/>

## <sub><code>04.</code></sub> &nbsp;Currently

**CATI Phase 3** — retraining on a 10-class Singapore vehicle taxonomy (car, motorcycle, scooter, bus, van, lorry, container truck, prime mover, tipper truck, taxi). The Phase 2 model was trained on COCO pseudo-labels; this fixes it. GDino auto-labelling pipeline is built, running now.

Research Success Data Assistant at ASU College of Health Solutions is the day job.

<br/>
<br/>
<br/>

## <sub><code>05.</code></sub> &nbsp;Stack

`Python` &nbsp;·&nbsp; `PyTorch` &nbsp;·&nbsp; `scikit-learn` &nbsp;·&nbsp; `HuggingFace` &nbsp;·&nbsp; `YOLO` &nbsp;·&nbsp; `LangGraph` &nbsp;·&nbsp; `FastAPI` &nbsp;·&nbsp; `DuckDB` &nbsp;·&nbsp; `Parquet` &nbsp;·&nbsp; `Pandas` &nbsp;·&nbsp; `SQL` &nbsp;·&nbsp; `MongoDB` &nbsp;·&nbsp; `AWS`

<br/>
<br/>
<br/>

## <sub><code>06.</code></sub> &nbsp;Background

**`2026 →`** &nbsp; Research Success Data Assistant — ASU College of Health Solutions

**`2025 → 27`** &nbsp; MS Data Science — Arizona State University

**`2025`** &nbsp; Data Science Intern — GlobalLogic

**`2024`** &nbsp; SWE Intern — GlobalLogic

**`2021 → 25`** &nbsp; B.Tech Computer Science — MIT Manipal

<br/>
<br/>
<br/>

[sdonthi4@asu.edu](mailto:sdonthi4@asu.edu) &nbsp;·&nbsp; [LinkedIn](https://linkedin.com/in/suhas-reddy-msds/)

<br/>
<br/>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Suhxs-Reddy/Suhxs-Reddy/output/github-contribution-grid-snake-dark.svg" />
  <img alt="contribution snake" src="https://raw.githubusercontent.com/Suhxs-Reddy/Suhxs-Reddy/output/github-contribution-grid-snake.svg" width="100%" />
</picture>
