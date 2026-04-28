<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1f6feb,100:0d1117&height=180&section=header&text=Suhas%20Reddy&fontSize=58&fontColor=ffffff&fontAlignY=40&desc=Applied%20AI%20%E2%80%A2%20Computer%20Vision%20%E2%80%A2%20LLM%20Systems&descSize=16&descAlignY=64&descAlign=50&animation=fadeIn" width="100%" />

<a href="https://github.com/Suhxs-Reddy">
  <img src="https://readme-typing-svg.demolab.com?font=Inter&weight=500&size=20&duration=2800&pause=900&color=58A6FF&center=true&vCenter=true&width=720&lines=MS+Data+Science+%40+Arizona+State+University;Building+vision+systems+that+see+through+rain%2C+haze%2C+and+midnight;Shipping+LLM+agents+that+pick+sites+for+gigawatt+data+centers;Hackathon+builder+%E2%80%94+real+problems%2C+real+systems%2C+real+fast" alt="intro" />
</a>

<br/>

<a href="mailto:sdonthi4@asu.edu"><img src="https://img.shields.io/badge/Email-sdonthi4%40asu.edu-58a6ff?style=flat-square&labelColor=0d1117" /></a>
&nbsp;
<a href="https://linkedin.com/in/suhas-reddy-msds/"><img src="https://img.shields.io/badge/LinkedIn-suhas--reddy--msds-58a6ff?style=flat-square&labelColor=0d1117&logo=linkedin&logoColor=58a6ff" /></a>
&nbsp;
<img src="https://img.shields.io/badge/MS_Data_Science-ASU_·_4.0_GPA-58a6ff?style=flat-square&labelColor=0d1117" />

</div>

<br/>

> I build applied AI systems where the model meets the real world — fixed cameras in monsoon rain, gigawatt energy decisions, legal documents nobody reads. The interesting part is never the model. It's the **context** the model has been ignoring.

<br/>

---

<h2>🚦 &nbsp; CATI &nbsp;·&nbsp; <sub>Singapore traffic intelligence that adapts to weather, time, and the camera it came from</sub></h2>

<table border="0">
<tr><td valign="top">

**The problem.** Singapore runs **90 live traffic cameras** around the clock — through monsoon rain, midnight glare, and on hardware ranging from 1080p down to ancient 320×240. Generic object detectors like YOLO treat every frame identically. A clear afternoon highway and a rain-soaked night feed get the exact same processing. Vehicles get missed in the conditions that matter most.

**What I built.** A detector that **learns its own playbook** for each combination of weather, time, camera, and air quality — without retraining for every condition. Same model, different behavior depending on what's happening in the world right now.

**How it works (plain English).** I bolted small "context tuner" layers ([FiLM](https://arxiv.org/abs/1709.07871)) onto YOLOv11's vision backbone. The tuners read live signals from Singapore's national APIs — current weather, hour of day, camera ID, PM2.5 — and dial the network's internal features up or down accordingly. Each of the 90 cameras gets a learned 16-dim embedding that captures its viewpoint and quirks. Initialization is identity (γ=1, β=0), so the model starts equivalent to vanilla YOLO and only specializes where it helps.

**Achievement.** ~1.4% parameter overhead. Negligible inference cost. **First published traffic detector to modulate detection features on real-time environmental metadata.**

`PyTorch` `YOLOv11` `FiLM` `BoT-SORT` `data.gov.sg` `Docker` `HF Spaces`

→ [**View project**](https://github.com/Suhxs-Reddy/sg-smart-city-analytics)

</td></tr>
</table>

<br/>

---

<h2>⚡ &nbsp; COLLIDE &nbsp;·&nbsp; <sub>AI siting platform for gigawatt data centers</sub></h2>

<table border="0">
<tr><td valign="top">

**The problem.** Hyperscale AI data centers need **50–500 MW of power, 24/7**. Connecting to the grid takes **3–7 years**. The fastest path is building your own natural-gas plant on-site — but that means evaluating land, gas pipelines, and electricity markets *all at once*. Consulting firms charge six figures and take months.

**What I built.** An AI platform that scores any candidate location across all three dimensions in **seconds**, with quantified risk, a 20-year NPV forecast, and an analyst that answers what-if questions in natural language.

**How it works (plain English).** Three specialized ML models score each axis: a Random Forest for land viability, a GPU Gaussian KDE for gas-pipeline reliability, and a Gaussian Mixture Model that classifies electricity market regimes (normal vs wind-curtailment vs scarcity). A TOPSIS composite blends them into one score. A LangGraph agent layered on Claude lets you ask things like *"what if gas prices spike 30%?"* and re-runs the analysis on live ERCOT and CAISO data. Monte Carlo (10,000 scenarios) gives P10/P50/P90 NPV bands.

**Achievement.** Collapses a multi-month consulting workflow into seconds. Live market data refreshes every 5 minutes. Built solo for the **ASU Energy Hackathon 2026**.

`FastAPI` `LangGraph` `Claude` `DuckDB` `Leaflet` `SHAP` `Random Forest` `KDE` `GMM` `Monte Carlo`

→ [**View project**](https://github.com/Suhxs-Reddy/EnergyHackathon)

</td></tr>
</table>

<br/>

---

<h2>🛠️ &nbsp; Other Work</h2>

<br/>

<table>
<tr>
<td width="50%" valign="top">

#### 🛡️ [DataGuard](https://github.com/Suhxs-Reddy/KiroHackathon)
**Privacy policies are 5,000 words of legal jargon nobody reads.** A Chrome extension that uses Llama 3.1 to parse any site's policy, surfaces a 14-category Apple-style privacy grid, checks Have I Been Pwned for breach history, and generates **one-click GDPR/CCPA opt-outs** with pre-filled emails and follow-up calendar reminders.
<br/><br/>
<sub>`TypeScript` · `Chrome MV3` · `Llama 3.1 8B`</sub>

</td>
<td width="50%" valign="top">

#### 📚 [AI Study Buddy](https://github.com/Suhxs-Reddy/CBC-Hackathon)
**Generic chatbots hallucinate when your course material isn't in their training data.** A Canvas LMS × Claude RAG tutor that retrieves only from your actual course documents, with a **low-relevance threshold** that flags out-of-syllabus questions instead of making up answers.
<br/><br/>
<sub>`React` · `FastAPI` · `Claude` · `Supermemory`</sub>

</td>
</tr>
<tr>
<td width="50%" valign="top">

#### 🏥 [Hospital Unsupervised Learning](https://github.com/Suhxs-Reddy/DShospitalproject)
Pattern discovery and patient clustering on hospital data — finding structure in records without labeled outcomes.
<br/><br/>
<sub>`Jupyter` · `scikit-learn`</sub>

</td>
<td width="50%" valign="top">

#### 🛒 [Walmart Demand Forecast](https://github.com/Suhxs-Reddy/prediction)
Store-level weekly demand prediction with lagged/rolling features, holiday flags, and chronological splits. Random Forest vs linear baselines, evaluated on MAE/RMSE.
<br/><br/>
<sub>`Python` · `time-series`</sub>

</td>
</tr>
</table>

<br/>

---

<h2>🧰 &nbsp; Stack</h2>

<div align="center">

<img src="https://skillicons.dev/icons?i=python,pytorch,sklearn,fastapi,docker,aws&theme=dark" />
<br/>
<img src="https://skillicons.dev/icons?i=ts,react,nextjs,tailwind,vercel,git&theme=dark" />

<br/><br/>

<sub>
<b>ML / Data</b> &nbsp; Python · PyTorch · scikit-learn · Pandas · NumPy &nbsp;·&nbsp; <b>Backend</b> &nbsp; FastAPI · Docker · AWS (S3, Athena) · MySQL · MongoDB
<br/>
<b>Frontend</b> &nbsp; TypeScript · React · Vite · Tailwind &nbsp;·&nbsp; <b>AI</b> &nbsp; Claude · LangGraph · HuggingFace · Llama · YOLO &nbsp;·&nbsp; <b>Observability</b> &nbsp; Grafana · Tableau
</sub>

</div>

<br/>

---

<h2>🎓 &nbsp; Background</h2>

- **MS Data Science** — Arizona State University · `4.0 GPA` · May 2027
- **B.Tech Computer Science** (minor: Big Data Analytics) — Manipal Institute of Technology · 2025
- **Research Assistant** — ASU College of Health Solutions · Jan 2026 → present
- **Data Science Intern** — GlobalLogic · Platform Analytics for Auth/RBAC · Grafana, Athena, S3
- **SWE Intern** — GlobalLogic · GenAI Platform Backend · FastAPI, Docker, async S3 audit logging

<br/>

<div align="center">

<img src="https://github-readme-stats.vercel.app/api?username=Suhxs-Reddy&show_icons=true&theme=github_dark&hide_border=true&bg_color=0d1117&title_color=58a6ff&icon_color=58a6ff&text_color=c9d1d9&count_private=true&include_all_commits=true&rank_icon=github&card_width=480" height="170" />

<br/><br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1f6feb,100:0d1117&height=80&section=footer" width="100%" />

</div>
