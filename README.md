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

Every standard detector treats every frame the same. It doesn't know it's 3am. It doesn't know it's raining. It doesn't know camera 47 on the CTE sees through a degraded 320×240 lens that drifts darker in the wet. **It's flying blind when it doesn't have to be.**

CATI feeds the model what it's missing — live weather, PM2.5, time, camera identity — and uses **[FiLM layers](https://arxiv.org/abs/1709.07871)** to re-wire YOLOv11's backbone at every single inference step. The model that runs at 3am in a storm is not the same model that runs at noon on a clear highway. One set of weights. Ninety different behaviours. 130K extra parameters on 9.4M.

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

Every AI hyperscaler needs 100+ megawatts, around the clock. Connecting to the US grid takes **3 to 7 years**. So they build their own gas plants — and picking *where* is a three-body problem that breaks every existing tool, because land, gas, and power are not the same kind of data and can't be scored the same way.

Land viability is a **static multi-constraint problem** — parcel attributes, zoning, fiber, water, flood risk. A **Random Forest** handles feature interactions and explains each score via SHAP: *"site A ranks 47th because highway access (28%) and substation proximity (24%) dominate, but wildfire risk drags it."* Gas supply is a **spatial density problem** — 50 years of PHMSA pipeline failure records at exact coordinates, severity-weighted. A **GPU-accelerated KDE** estimates incident probability per location across 100k points in seconds. Power economics is a **temporal regime problem** — LMP swings ±$30/MWh intra-day across market regimes that aren't labeled. A **Moirai foundation model** handles 72-hour forecasting, a **GMM** discovers the three market states (normal / stress-scarcity / wind-curtailment), and a **Random Forest** predicts spread durability conditioned on the current regime.

**TOPSIS** combines the three axes (30/35/35 by default, user-adjustable). **10,000 Monte Carlo simulations** give P10/P50/P90 NPV over 20 years. A **LangGraph agent on Claude** sits on top, reasoning across the full scorecard — *"What if gas spikes 40%?"* re-runs the analysis live against ERCOT/CAISO feeds. What consulting firms charge six figures and take months to deliver.

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
