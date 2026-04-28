<div align="center">

<br/>

# Context is the model.

<sub>**Suhas Reddy** &nbsp;·&nbsp; MS Data Science @ Arizona State University &nbsp;·&nbsp; `4.0 GPA`</sub>

<br/>

<a href="mailto:sdonthi4@asu.edu"><img src="https://img.shields.io/badge/email-sdonthi4%40asu.edu-39d353?style=flat-square&labelColor=0d1117" /></a>
&nbsp;
<a href="https://linkedin.com/in/suhas-reddy-msds/"><img src="https://img.shields.io/badge/linkedin-suhas--reddy--msds-39d353?style=flat-square&labelColor=0d1117" /></a>
&nbsp;
<a href="https://github.com/Suhxs-Reddy"><img src="https://img.shields.io/badge/Tempe-AZ-39d353?style=flat-square&labelColor=0d1117" /></a>

<br/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=18&duration=2400&pause=600&color=39D353&center=true&vCenter=true&width=820&lines=90+live+traffic+cameras.;50%E2%80%93500+MW+gigawatt+siting+decisions.;5%2C000-word+privacy+policies%2C+read+in+four+seconds.;the+interesting+part+is+never+the+model." />

<br/>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Platane/snk/output/github-contribution-grid-snake-dark.svg" />
  <img alt="snake animation eating contributions" src="https://raw.githubusercontent.com/Platane/snk/output/github-contribution-grid-snake.svg" width="100%" />
</picture>

</div>

<br/>

> I keep ending up at the seam between models and the messy world they have to live in. Every project I've shipped started because something obvious was being ignored — the camera that knows what camera it is, the gas-pipeline data that's already public, the privacy policy nobody reads. Less interested in benchmark SOTA. More interested in **what's already in the room with the model.**

<br/>

```
══════════════════════════════════════════════════════════════════════════
```

<br/>

## `/ 01` &nbsp;&nbsp; CATI &nbsp;<sub><sup>Singapore traffic intelligence</sup></sub>

> *A traffic detector that adapts in real time to weather, time of day, and the camera the frame came from.*

Singapore runs **90 live traffic cameras** around the clock — through monsoon rain, midnight glare, and on hardware that ranges from 1080p down to ancient 320×240 feeds. Generic detectors like YOLO treat every frame identically. A clear afternoon highway and a rain-soaked night get the exact same processing, and vehicles get missed in the conditions that matter most.

So I gave the model a **dial for context.** Same network, but it learns to behave differently when it's raining, when it's 2am, and when it's looking through Camera #47 versus Camera #12. No retraining. No separate models. One detector that knows where and when it is.

<details>
<summary><b>↳ &nbsp; how it works under the hood</b></summary>

<br/>

The dial is built from **FiLM layers** ([Perez et al., 2018](https://arxiv.org/abs/1709.07871)) — small modules that take an external signal and use it to scale and shift the network's internal features:

```
feature_out = γ(context) ⊙ feature_in + β(context)
```

A tiny context encoder (MLP) processes live signals from Singapore's national APIs — weather code, temperature, sin/cos hour-of-day, camera ID embedding, resolution, PM2.5 — and predicts `γ` and `β` for the P3 / P4 / P5 stages of YOLOv11's backbone. Each of the 90 cameras gets a learned **16-dim embedding** that captures its viewpoint and quirks. Initialization is identity (γ=1, β=0), so on day zero the model is exactly equivalent to vanilla YOLO and only specializes where it actually helps loss.

Total cost: **~130K extra parameters** on top of YOLO's 9.4M — a 1.4% overhead. Inference cost is negligible.

</details>

> 🏆 First published traffic detector to modulate detection features on real-time environmental metadata.

<sub>`PyTorch` &nbsp;·&nbsp; `YOLOv11` &nbsp;·&nbsp; `FiLM` &nbsp;·&nbsp; `BoT-SORT` &nbsp;·&nbsp; `data.gov.sg` &nbsp;·&nbsp; `Docker` &nbsp;·&nbsp; `HF Spaces`</sub>

→ &nbsp;[**github.com/Suhxs-Reddy/sg-smart-city-analytics**](https://github.com/Suhxs-Reddy/sg-smart-city-analytics)

<br/>

```
══════════════════════════════════════════════════════════════════════════
```

<br/>

## `/ 02` &nbsp;&nbsp; COLLIDE &nbsp;<sub><sup>AI siting for gigawatt data centers</sup></sub>

> *Picks where to build a 100 MW data center in seconds — work that takes consulting firms months.*

Hyperscale AI data centers need **50–500 MW of power, 24/7**. Connecting to the grid takes **3–7 years**. The fastest path is putting your own gas plant on-site (behind-the-meter), but that means evaluating *land* (zoning, fiber, water, flood risk), *gas supply* (pipeline reliability, hub distance), and *electricity markets* (LMP spreads, price regimes) — all together. Consulting firms charge six figures and take months.

I built an analyst that does this in seconds. Click any coordinate on a map. Get a scorecard with quantified risk, a 20-year cost forecast, and an AI that answers what-ifs in plain English.

<details>
<summary><b>↳ &nbsp; how it works under the hood</b></summary>

<br/>

Three specialized models, each scoring one axis:

| Axis | Model | What it does |
|---|---|---|
| **Land** | Random Forest (300 trees) | Predicts viability from highway access, substation proximity, pipeline distance, water access, flood zone. SHAP explainer for per-site attribution. |
| **Gas** | GPU Gaussian KDE | Severity-weighted density of real **PHMSA pipeline incidents** + interstate pipeline proximity + Waha hub distance. |
| **Power** | Random Forest + GMM | RF predicts spread durability from LMP and market state. GMM (3 components) classifies the live ERCOT market into **normal / wind-curtailment / scarcity** regimes. |

A **TOPSIS** composite blends the three (user-adjustable weights). On top of that sits a **LangGraph agent** running Claude — five intents (stress-test, compare, timing, explain, configure) that re-run the analysis on live ERCOT/CAISO feeds when you ask *"what if gas prices spike 30%?"*. **Monte Carlo (10,000 scenarios)** gives P10/P50/P90 NPV bands over a 20-year horizon at 8% WACC.

Live market refresh: every 5 minutes. 72-hour LMP forecast: Moirai foundation model with confidence bands.

</details>

> 🏆 Collapses a multi-month consulting workflow into seconds. Built solo for ASU Energy Hackathon 2026.

<sub>`FastAPI` &nbsp;·&nbsp; `LangGraph` &nbsp;·&nbsp; `Claude` &nbsp;·&nbsp; `DuckDB` &nbsp;·&nbsp; `Leaflet` &nbsp;·&nbsp; `SHAP` &nbsp;·&nbsp; `Random Forest` &nbsp;·&nbsp; `KDE` &nbsp;·&nbsp; `GMM` &nbsp;·&nbsp; `Monte Carlo`</sub>

→ &nbsp;[**github.com/Suhxs-Reddy/EnergyHackathon**](https://github.com/Suhxs-Reddy/EnergyHackathon)

<br/>

```
══════════════════════════════════════════════════════════════════════════
```

<br/>

## `/ 03` &nbsp;&nbsp; Other Work

<table>
<tr>
<td width="50%" valign="top">

#### 🛡️ [DataGuard](https://github.com/Suhxs-Reddy/KiroHackathon)

Privacy policies are 5,000 words of legal jargon nobody reads. So you click "Accept" without knowing if a site sells your location.

**A Chrome extension** that uses Llama 3.1 to parse any policy, surfaces a 14-category Apple-style privacy grid, checks **Have I Been Pwned** for breach history, and generates **one-click GDPR/CCPA opt-outs** with pre-filled emails and follow-up reminders.

<sub>`TypeScript` · `Chrome MV3` · `Llama 3.1 8B`</sub>

</td>
<td width="50%" valign="top">

#### 📚 [AI Study Buddy](https://github.com/Suhxs-Reddy/CBC-Hackathon)

Generic chatbots hallucinate when your course material isn't in their training data. They sound confident anyway.

**A Canvas LMS × Claude RAG tutor** that retrieves only from your actual course documents, with a **low-relevance threshold** that flags out-of-syllabus questions instead of inventing answers.

<sub>`React` · `FastAPI` · `Claude` · `Supermemory`</sub>

</td>
</tr>
<tr>
<td width="50%" valign="top">

#### 🏥 [Hospital Unsupervised Learning](https://github.com/Suhxs-Reddy/DShospitalproject)

Pattern discovery and patient clustering on hospital data — finding structure in records without labeled outcomes.

<sub>`Jupyter` · `scikit-learn`</sub>

</td>
<td width="50%" valign="top">

#### 🛒 [Walmart Demand Forecast](https://github.com/Suhxs-Reddy/prediction)

Store-level weekly demand prediction with lagged/rolling features, holiday flags, chronological splits. Random Forest vs linear baselines on MAE/RMSE.

<sub>`Python` · `time-series`</sub>

</td>
</tr>
</table>

<br/>

```
══════════════════════════════════════════════════════════════════════════
```

<br/>

## `/ 04` &nbsp;&nbsp; Stack &nbsp;&&nbsp; Background

<div align="center">

<img src="https://skillicons.dev/icons?i=python,pytorch,sklearn,fastapi,docker,aws,ts,react,nextjs,tailwind,vercel,git&theme=dark" />

</div>

<br/>

<details>
<summary><b>↳ &nbsp; full stack breakdown</b></summary>

<br/>

| Layer | Tools |
|---|---|
| **ML / Data** | Python · PyTorch · scikit-learn · Pandas · NumPy |
| **Backend** | FastAPI · Docker · AWS (S3, Athena) · MySQL · MongoDB |
| **Frontend** | TypeScript · React · Vite · Tailwind |
| **AI** | Anthropic Claude · LangGraph · HuggingFace · Llama · YOLO |
| **Observability** | Grafana · Tableau |

</details>

<br/>

```
2026 → present   Research Assistant · ASU College of Health Solutions
2026             ASU Energy Hackathon · COLLIDE
2025 → 2027      MS Data Science · Arizona State University · 4.0 GPA
2025             Data Science Intern · GlobalLogic · Auth/RBAC platform analytics
2024             SWE Intern · GlobalLogic · GenAI platform backend
2021 → 2025      B.Tech Computer Science (minor: Big Data) · MIT Manipal
```

<br/>

<div align="center">

<img src="https://github-readme-stats.vercel.app/api?username=Suhxs-Reddy&show_icons=true&theme=merko&hide_border=true&bg_color=0d1117&count_private=true&include_all_commits=true&rank_icon=github&card_width=480" height="170" />

<br/>
<br/>

<sub><code>built in colab notebooks · shipped on vercel · drank too much coffee</code></sub>

</div>
