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

MS Data Science at ASU. I build ML systems that have to work in the real world — where data is messy, deployment is weird, and the interesting problems are the ones generic models can't see because nobody told them what to look for.

<br/>
<br/>

## <sub><code>02.</code></sub> &nbsp;Work

<br/>

<img src="https://capsule-render.vercel.app/api?type=transparent&color=22d3ee&height=2&fontColor=22d3ee&text=." width="100%" />

### [`'26` &nbsp; CATI](https://github.com/Suhxs-Reddy/sg-smart-city-analytics) &nbsp; <img src="https://img.shields.io/badge/computer%20vision-22d3ee?style=flat-square&labelColor=0d1117" /> &nbsp; <img src="https://img.shields.io/badge/live-22d3ee?style=flat-square&labelColor=0d1117" />

<sub>Context-Aware Traffic Intelligence · Singapore · solo build</sub>

Solo end-to-end build, live in production. Standard detectors treat every frame identically — same weights at 3am in a monsoon as at noon on a clear highway. Singapore's LTA network has 90 fixed cameras at known locations with live weather, PM2.5, and timestamp available at inference time. CATI uses all of it. **[FiLM layers](https://arxiv.org/abs/1709.07871)** inject environmental metadata directly into YOLOv11's backbone at P3/P4/P5, re-conditioning feature maps per inference step. One model, 90 context-specific behaviours, 130K extra parameters on 9.4M.

Deployed on HuggingFace Spaces — every 90 seconds it fetches all 90 LTA camera feeds, runs conditioned inference, and appends per-camera vehicle counts, directional splits, weather state, and road assignment to a public dataset. **213K+ records collected since April 2026**, structured for downstream traffic analysis: congestion modelling, peak-hour flow patterns, road-load estimation by vehicle class, multi-camera network analytics. Data pipeline is temporally split — train/val/test partitioned by date, no leakage. Grounding DINO auto-labelling (Phase 3, in progress) replaces COCO pseudo-labels with a 10-class Singapore-specific taxonomy so vehicle class counts are actually correct.

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

<sub>AI siting for BTM data center gas plants · ASU Energy Hackathon 2026 · group project · data architecture & pipeline</sub>

Every AI hyperscaler needs 100+ megawatts. Connecting to the US grid takes **3 to 7 years**. So they build their own gas plants — and siting one means solving three constraints simultaneously: **find viable land, confirm gas supply reliability, prove the power economics work**. The problem isn't that this is three problems. It's that these are three completely different *types* of data, and forcing them into a shared scoring framework without destroying what makes each one meaningful is where it actually gets hard.

Land viability is **static and multi-constraint** — parcel ownership, zoning, FEMA flood zone, distance to fiber, water, pipelines, substations. Static data calls for a model that handles feature interactions and is interpretable (developers need to know *why* a parcel ranks where it does). **Random Forest + SHAP**: highway access 28.5% of variance, substation proximity 23.7%, pipeline distance 19.9%. Gas supply is a **sparse spatial point process** — 50 years of PHMSA pipeline failure records, each a lat/lon coordinate with a severity weight. This isn't a distance problem; it's a density problem. You need a continuous risk surface, not discrete proximity rules. **GPU-accelerated Gaussian KDE** over 100k+ incident points, severity-weighted, falling back to CPU tensors gracefully. Power economics is a **volatile time series with hidden structure** — LMP swings ±$30/MWh intra-day through market regimes nobody labeled. Raw price is a bad feature; regime-conditioned durability is the right one. **GMM** on ERCOT state vectors discovers three unlabeled regimes (normal / stress-scarcity / wind-curtailment). A **Random Forest** then predicts spread durability conditioned on the current regime. **Moirai** handles 72-hour forecasting without per-node retraining.

Getting the data clean enough to score against was its own project — **13 live sources across 10 public APIs** ingested through a DuckDB lake with Pandera schema validation, freshness SLAs per source, natural-key deduplication, SHA-256 integrity manifests, and row-level lineage. Live zoning news and pipeline safety records flow in via Tavily, reasoned over by Claude Haiku, and adjust scores in real time. **TOPSIS** composites the three axes without flattening them into a meaningless average. **10,000 Monte Carlo simulations** give P10/P50/P90 NPV. A **LangGraph agent** on Claude answers what-ifs against live ERCOT/CAISO feeds. Only team at the hackathon to deliver all three sub-problems.

<table><tr>
<td align="center"><h3>13</h3><sub>live data sources</sub></td>
<td align="center"><h3>3</h3><sub>sub-problems solved (only team)</sub></td>
<td align="center"><h3>10K</h3><sub>Monte Carlo scenarios</sub></td>
<td align="center"><h3>5 min</h3><sub>market data refresh</sub></td>
</tr></table>

`FastAPI` &nbsp;·&nbsp; `LangGraph` &nbsp;·&nbsp; `Claude` &nbsp;·&nbsp; `DuckDB` &nbsp;·&nbsp; `Pandera` &nbsp;·&nbsp; `SHAP` &nbsp;·&nbsp; `Random Forest` &nbsp;·&nbsp; `GPU KDE` &nbsp;·&nbsp; `GMM` &nbsp;·&nbsp; `Moirai` &nbsp;·&nbsp; `React` &nbsp;·&nbsp; `Leaflet`

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
