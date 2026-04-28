<div align="center">

<br/>

# `Suhas Reddy`

<sub>applied AI engineer · MS Data Science @ Arizona State University · `4.0 GPA`</sub>

<br/>

<a href="mailto:sdonthi4@asu.edu"><img src="https://img.shields.io/badge/email-sdonthi4%40asu.edu-39d353?style=flat-square&labelColor=0d1117" /></a>
&nbsp;
<a href="https://linkedin.com/in/suhas-reddy-msds/"><img src="https://img.shields.io/badge/linkedin-suhas--reddy--msds-39d353?style=flat-square&labelColor=0d1117" /></a>
&nbsp;
<img src="https://img.shields.io/badge/Tempe-AZ-39d353?style=flat-square&labelColor=0d1117" />

<br/>
<br/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=18&duration=3500&pause=1000&color=39D353&center=true&vCenter=true&width=820&lines=the+interesting+part+is+never+the+model.;it's+the+context+the+model+has+been+ignoring." />

</div>

<br/>

> I keep ending up at the seam between models and the messy world they have to live in. Every project I've shipped started because something obvious was being ignored — the camera that knows what camera it is, the gas-pipeline data that's already public, the privacy policy nobody reads. Less interested in benchmark SOTA. More interested in what's already in the room with the model.

<br/>

```
─────────────────────────────────────────────────────────────────────────
```

<br/>

## `/ 01` &nbsp;&nbsp; CATI

> *Singapore traffic intelligence that adapts to weather, time, and the camera it came from.*

Singapore runs **90 live traffic cameras** around the clock — through monsoon rain, midnight glare, and on hardware that ranges from 1080p down to ancient 320×240 feeds. Generic detectors like YOLO treat every frame identically. A clear afternoon highway and a rain-soaked night get the exact same processing, and vehicles get missed in the conditions that matter most.

I built a detector that **learns its own playbook** for each combination of weather, time, camera, and air quality — without retraining for every condition.

The trick is small "context tuner" layers ([FiLM](https://arxiv.org/abs/1709.07871)) bolted onto YOLOv11's backbone. They read live signals from Singapore's national APIs — current weather, hour of day, camera ID, PM2.5 — and dial the network's internal features up or down accordingly. Each of the 90 cameras gets a learned 16-dim embedding that captures its viewpoint and quirks. Initialization is identity, so the model starts equivalent to vanilla YOLO and only specializes where it helps.

**~1.4% parameter overhead. Negligible inference cost.** First published traffic detector to modulate detection features on real-time environmental metadata.

<sub>`PyTorch` &nbsp;·&nbsp; `YOLOv11` &nbsp;·&nbsp; `FiLM` &nbsp;·&nbsp; `BoT-SORT` &nbsp;·&nbsp; `data.gov.sg` &nbsp;·&nbsp; `Docker` &nbsp;·&nbsp; `HF Spaces`</sub>

→ &nbsp;[**github.com/Suhxs-Reddy/sg-smart-city-analytics**](https://github.com/Suhxs-Reddy/sg-smart-city-analytics)

<br/>

```
─────────────────────────────────────────────────────────────────────────
```

<br/>

## `/ 02` &nbsp;&nbsp; COLLIDE

> *AI siting platform for gigawatt data centers.*

Hyperscale AI data centers need **50–500 MW of power, 24/7**. Connecting to the grid takes **3–7 years**. The fastest path is building your own natural-gas plant on-site — but that means evaluating land, gas pipelines, and electricity markets *all at once*. Consulting firms charge six figures and take months.

I built a platform that scores any candidate location across all three dimensions in **seconds**, with quantified risk and a 20-year cost forecast.

Three specialized ML models score each axis: a Random Forest for land viability, a GPU Gaussian KDE for gas-pipeline reliability (severity-weighted on real PHMSA incident data), and a Gaussian Mixture Model that classifies electricity market regimes — normal vs wind-curtailment vs scarcity. A TOPSIS composite blends them into one score. A LangGraph agent on top of Claude lets you ask things like *"what if gas prices spike 30%?"* and re-runs the analysis on live ERCOT and CAISO market feeds. Monte Carlo (10,000 scenarios) gives P10/P50/P90 NPV bands.

**Collapses a multi-month consulting workflow into seconds.** Built solo for the ASU Energy Hackathon 2026.

<sub>`FastAPI` &nbsp;·&nbsp; `LangGraph` &nbsp;·&nbsp; `Claude` &nbsp;·&nbsp; `DuckDB` &nbsp;·&nbsp; `Leaflet` &nbsp;·&nbsp; `SHAP` &nbsp;·&nbsp; `Random Forest` &nbsp;·&nbsp; `KDE` &nbsp;·&nbsp; `GMM` &nbsp;·&nbsp; `Monte Carlo`</sub>

→ &nbsp;[**github.com/Suhxs-Reddy/EnergyHackathon**](https://github.com/Suhxs-Reddy/EnergyHackathon)

<br/>

```
─────────────────────────────────────────────────────────────────────────
```

<br/>

## `/ 03` &nbsp;&nbsp; Other Work

<table>
<tr>
<td width="50%" valign="top">

#### 🛡️ [DataGuard](https://github.com/Suhxs-Reddy/KiroHackathon)

Privacy policies are 5,000 words of legal jargon nobody reads. A Chrome extension that uses **Llama 3.1** to parse any site's policy, surfaces a 14-category Apple-style privacy grid, checks Have I Been Pwned for breach history, and generates **one-click GDPR/CCPA opt-outs** with pre-filled emails and follow-up reminders.

<sub>`TypeScript` · `Chrome MV3` · `Llama 3.1 8B`</sub>

</td>
<td width="50%" valign="top">

#### 📚 [AI Study Buddy](https://github.com/Suhxs-Reddy/CBC-Hackathon)

Generic chatbots hallucinate when your course material isn't in their training data. A Canvas LMS × Claude RAG tutor that retrieves only from your actual course documents, with a **low-relevance threshold** that flags out-of-syllabus questions instead of making up answers.

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

Store-level weekly demand with lagged/rolling features, holiday flags, chronological splits. Random Forest vs linear baselines on MAE/RMSE.

<sub>`Python` · `time-series`</sub>

</td>
</tr>
</table>

<br/>

```
─────────────────────────────────────────────────────────────────────────
```

<br/>

## `/ 04` &nbsp;&nbsp; Stack

<div align="center">

<img src="https://skillicons.dev/icons?i=python,pytorch,sklearn,fastapi,docker,aws&theme=dark" />
<br/>
<img src="https://skillicons.dev/icons?i=ts,react,nextjs,tailwind,vercel,git&theme=dark" />

</div>

<br/>

<sub>**ML / Data** &nbsp;·&nbsp; Python · PyTorch · scikit-learn · Pandas · NumPy</sub><br/>
<sub>**Backend** &nbsp;·&nbsp; FastAPI · Docker · AWS (S3, Athena) · MySQL · MongoDB</sub><br/>
<sub>**Frontend** &nbsp;·&nbsp; TypeScript · React · Vite · Tailwind</sub><br/>
<sub>**AI** &nbsp;·&nbsp; Anthropic Claude · LangGraph · HuggingFace · Llama · YOLO</sub><br/>
<sub>**Observability** &nbsp;·&nbsp; Grafana · Tableau</sub>

<br/>

```
─────────────────────────────────────────────────────────────────────────
```

<br/>

## `/ 05` &nbsp;&nbsp; Background

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

</div>

<br/>
<br/>

<div align="center">
<sub>
<code>built in colab notebooks · shipped on vercel · drank too much coffee</code>
</sub>
</div>
