<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:4ade80,100:0d1117&height=240&section=header&text=Suhas%20Reddy&fontSize=68&fontColor=ffffff&fontAlignY=35&desc=MS+Data+Science+%C2%B7+Arizona+State+University&descSize=16&descAlignY=56&animation=fadeIn" width="100%" />

<br/>

<a href="mailto:sdonthi4@asu.edu"><img src="https://img.shields.io/badge/sdonthi4@asu.edu-4ade80?style=flat-square&logo=gmail&logoColor=4ade80&labelColor=0d1117" /></a>
&nbsp;
<a href="https://linkedin.com/in/suhas-reddy-msds/"><img src="https://img.shields.io/badge/linkedin-suhas--reddy--msds-818cf8?style=flat-square&logo=linkedin&logoColor=818cf8&labelColor=0d1117" /></a>
&nbsp;
<a href="https://github.com/Suhxs-Reddy"><img src="https://img.shields.io/badge/github-Suhxs--Reddy-e5e7eb?style=flat-square&logo=github&logoColor=e5e7eb&labelColor=0d1117" /></a>

<br/>
<br/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=20&duration=2800&pause=900&color=22c55e&center=true&vCenter=true&width=900&lines=let+the+data+lead.;context+changes+everything.;the+signal+is+already+in+the+room.;ship+it%2C+then+write+the+paper." />

</div>

<br/>

> The data is always dirtier than you think. The deployment is always weirder than you imagined. Generic models miss the obvious because nobody told them what was obvious. I work at the seam between AI and the world it has to live in — where 90 traffic cameras include eleven 320×240 relics from the early 2000s, where pipeline incident data from 1986 changes where you'd put a gigawatt plant in 2026, where the privacy policy nobody reads is also the one selling your location. **The fun part isn't tuning the model. It's noticing what the model can't see.**

<br/>

---

<br/>

<div align="center">

## <img src="https://api.iconify.design/fluent-emoji-flat/sparkles.svg" width="24" /> &nbsp; What I'm good at

</div>

<br/>

<table>
<tr>
<td valign="top" width="33%" align="center">

<img src="https://api.iconify.design/fluent-emoji-flat/eye.svg" width="40" />

### Computer vision in the wild

Noisy cameras, bad weather, ancient hardware. I build detectors that know where and when they are.

</td>
<td valign="top" width="33%" align="center">

<img src="https://api.iconify.design/fluent-emoji-flat/robot.svg" width="40" />

### LLM agents over real data

RAG, tool-use, LangGraph orchestration — grounded in live APIs, not toy demos.

</td>
<td valign="top" width="33%" align="center">

<img src="https://api.iconify.design/fluent-emoji-flat/chart-increasing.svg" width="40" />

### Applied ML & DS

Time-series, clustering, geospatial scoring. The boring stuff that makes the interesting stuff work.

</td>
</tr>
</table>

<br/>

---

<br/>

<div align="center">

## <img src="https://api.iconify.design/fluent-emoji-flat/rocket.svg" width="24" /> &nbsp; Featured Projects

</div>

<br/>

### `01` &nbsp; CATI &nbsp;<sub><sup>Singapore traffic intelligence</sup></sub>

<a href="https://github.com/Suhxs-Reddy/sg-smart-city-analytics"><img src="https://img.shields.io/badge/View%20on%20GitHub-0d1117?style=for-the-badge&logo=github&logoColor=14b8a6" /></a>

Singapore runs ninety traffic cameras around the clock. Some are 1080p. Eleven are 320×240 hardware from the early 2000s. They look out over monsoon storms, midnight glare, peak-hour chaos, and the dead silence at 4am. And every single frame — every camera, every condition — gets piped through the same generic YOLO that treats them all identically. Which is exactly when vehicles get missed.

I kept staring at this and thinking: **the model already has access to everything it needs to do better.** The camera knows which camera it is. The system has live weather. The clock is right there. The PM2.5 air-quality index is one public API call away. So why is a clear afternoon highway and a rain-soaked midnight feed processed the same way?

The answer turned out to live in a 2018 paper I came across — **FiLM** (*Feature-wise Linear Modulation*, [Perez et al.](https://arxiv.org/abs/1709.07871)). It's a small, beautiful idea: tiny modules that take an external signal and use it to **scale and shift the network's internal features**. Identity at initialization, so the model starts as plain YOLO and only specializes where it actually helps. No retraining. No ensemble. One model, with a dial for context.

The way to picture it: imagine the vision network as layers of pattern detectors stacked on top of each other. With FiLM, I can dial each detector's volume up or down based on the current weather, time, camera, and air quality. In monsoon rain, the detectors looking for crisp edges get turned down. The ones looking for blurry motion blobs get turned up. At 2am on Camera #47, the network behaves differently than at 2pm on Camera #12. The model figures out the dialing on its own — I just feed it the current state of the world.

```mermaid
%%{init: {'theme':'dark', 'themeVariables': {'primaryColor':'#14b8a6','primaryBorderColor':'#14b8a6','lineColor':'#4ade80','primaryTextColor':'#e5e7eb','background':'#0d1117'}}}%%
flowchart LR
    subgraph CTX["live context · data.gov.sg"]
        W[weather]
        T[time]
        C[camera ID]
        R[resolution]
        A[PM2.5]
    end
    CTX --> ENC["context encoder<br/>(MLP → γ, β)"]
    IMG["camera frame"] --> YOLO["YOLOv11<br/>backbone"]
    YOLO --> F1["FiLM<sub>P3</sub>"]
    YOLO --> F2["FiLM<sub>P4</sub>"]
    YOLO --> F3["FiLM<sub>P5</sub>"]
    ENC --> F1 & F2 & F3
    F1 & F2 & F3 --> H[detection head]
    H --> OUT["vehicles<br/>plates<br/>tracks"]

    style CTX fill:#0d1117,stroke:#14b8a6
    style ENC fill:#14b8a6,color:#fff,stroke:#14b8a6
    style F1 fill:#14b8a6,color:#fff,stroke:#14b8a6
    style F2 fill:#14b8a6,color:#fff,stroke:#14b8a6
    style F3 fill:#14b8a6,color:#fff,stroke:#14b8a6
    style OUT fill:#4ade80,color:#0d1117,stroke:#4ade80
```

<table>
<tr><td>💡</td><td><strong>~130K extra parameters</strong> on top of YOLO's 9.4M — a 1.4% overhead. Inference cost is rounding-error.</td></tr>
</table>

<details>
<summary>🔬 &nbsp;<b>the math, the numbers, what I actually built</b></summary>

<br/>

The FiLM transform is just two affine parameters per channel:

```
feature_out = γ(context) ⊙ feature_in + β(context)
```

A tiny encoder (MLP) processes weather code, temperature, sin/cos hour-of-day, a 16-dim camera ID embedding (learned per camera), resolution, and PM2.5, then predicts `γ` and `β` for the P3/P4/P5 stages of YOLOv11's backbone. Identity initialization (γ=1, β=0) means day zero is exactly vanilla YOLO — the model only specializes where it helps loss.

Total cost: ~130K extra parameters on YOLO's 9.4M. Inference cost is rounding-error.

</details>

<sub>`PyTorch` &nbsp;·&nbsp; `YOLOv11` &nbsp;·&nbsp; `FiLM` &nbsp;·&nbsp; `BoT-SORT` &nbsp;·&nbsp; `data.gov.sg` &nbsp;·&nbsp; `Docker` &nbsp;·&nbsp; `HuggingFace Spaces`</sub>

<br/>

---

<br/>

### `02` &nbsp; COLLIDE &nbsp;<sub><sup>AI siting for gigawatt data centers</sup></sub>

<a href="https://github.com/Suhxs-Reddy/EnergyHackathon"><img src="https://img.shields.io/badge/View%20on%20GitHub-0d1117?style=for-the-badge&logo=github&logoColor=f59e0b" /></a>

Every AI hyperscaler needs power — gigawatts of it, twenty-four seven. Connecting a new data center to the grid takes three to seven years. Your competitors aren't waiting. So you build your own natural-gas plant on-site, behind the meter. The catch is picking *where*, and the catch is brutal: it's a three-body problem. Land has its own constraints (zoning, fiber, water, flood). Gas supply has its own (pipeline reliability, hub distance, curtailment risk). Electricity markets have their own (LMP spreads, price regimes, scarcity events). Real consultants charge six figures and take months because they evaluate these axes one at a time.

I wanted the three to **argue with each other in real time** instead of sequentially. A great parcel with bad gas economics shouldn't beat a decent parcel with stellar gas economics, but spreadsheets can't tell you that. So I gave each axis its own ML model — a Random Forest for land (interpretable, gives me SHAP), a GPU Gaussian KDE for gas pipeline reliability (because incidents cluster spatially and KDE reads that distribution out cleanly), and a Random Forest plus GMM for power markets (because electricity really does have distinct regimes — normal, wind-curtailment, scarcity — and the GMM finds them in unlabeled ERCOT data without me having to tag anything). They all feed into TOPSIS, a multi-criteria scoring method from 1981 that's still the cleanest way to combine apples and oranges with adjustable weights.

The interesting layer sits on top: a LangGraph agent running Claude. Five intents — stress-test, compare, timing, explain, configure. You ask *"what if gas prices spike 30%?"* in plain English and the system actually re-runs the analysis on live ERCOT and CAISO feeds.

The whole thing is wired to ten public data sources updating every five minutes. Click any coordinate. Get a scorecard with quantified risk, a 20-year cost forecast from 10,000 Monte Carlo simulations, and a 72-hour LMP forecast from the **Moirai foundation model**. What used to take consulting firms months happens while you're still scrolling.

```mermaid
%%{init: {'theme':'dark', 'themeVariables': {'primaryColor':'#f59e0b','primaryBorderColor':'#f59e0b','lineColor':'#4ade80','primaryTextColor':'#e5e7eb','background':'#0d1117'}}}%%
flowchart LR
    subgraph SRC["10 public data sources · live"]
        S1[PHMSA pipelines]
        S2[ERCOT / CAISO]
        S3[county GIS]
        S4[Waha hub]
    end
    SRC --> M1["Land<br/>Random Forest"]
    SRC --> M2["Gas<br/>GPU KDE"]
    SRC --> M3["Power<br/>RF + GMM"]
    M1 & M2 & M3 --> TOP["TOPSIS<br/>composite"]
    TOP --> AGT["LangGraph<br/>agent · Claude"]
    AGT --> UI["map UI<br/>scorecard + NPV"]

    style SRC fill:#0d1117,stroke:#f59e0b
    style M1 fill:#1f2937,color:#f59e0b,stroke:#f59e0b
    style M2 fill:#1f2937,color:#f59e0b,stroke:#f59e0b
    style M3 fill:#1f2937,color:#f59e0b,stroke:#f59e0b
    style TOP fill:#f59e0b,color:#0d1117,stroke:#f59e0b
    style AGT fill:#4ade80,color:#0d1117,stroke:#4ade80
    style UI fill:#4ade80,color:#0d1117,stroke:#4ade80
```

<table>
<tr><td>💡</td><td><strong>Built solo for ASU Energy Hackathon 2026.</strong> Collapses a multi-month consulting workflow into seconds.</td></tr>
</table>

<details>
<summary>🧮 &nbsp;<b>model specs and numbers</b></summary>

<br/>

**Land — Random Forest, 300 trees, max depth 8.** Top features by importance: highway access (28.5%), substation proximity (23.7%), pipeline distance (19.9%), water (14.6%), fiber (8.1%). SHAP for per-site attribution.

**Gas — GPU Gaussian KDE, bandwidth 0.5.** Trained on PHMSA pipeline incident coordinates (severity-weighted). Score: incident density 40% + interstate proximity 35% + Waha distance 25%.

**Power — Random Forest (200 trees, depth 6) + GMM (3 components, full covariance).** GMM trained on five ERCOT features. Cluster labels: `normal` (0), `wind_curtailment` (1), `stress_scarcity` (2). Training mean: LMP $78/MWh, wind 31%, demand 54.9 GW.

**Composite — TOPSIS** with default weights land 30% / gas 35% / power 35%, fully user-adjustable.

**NPV — Monte Carlo, 10,000 scenarios, 8% WACC, 100 MW plant assumption.** Returns P10/P50/P90 over 20 years.

**Agent — LangGraph orchestration, Claude Haiku + Sonnet.** Five intents: stress-test, compare, timing, explain, configure. Live market refresh every 5 minutes. 72-hour LMP forecast via Moirai with confidence bands.

</details>

<sub>`FastAPI` &nbsp;·&nbsp; `LangGraph` &nbsp;·&nbsp; `Claude` &nbsp;·&nbsp; `DuckDB` &nbsp;·&nbsp; `Leaflet` &nbsp;·&nbsp; `SHAP` &nbsp;·&nbsp; `Random Forest` &nbsp;·&nbsp; `KDE` &nbsp;·&nbsp; `GMM` &nbsp;·&nbsp; `Monte Carlo`</sub>

<br/>

---

<br/>

<div align="center">

## <img src="https://api.iconify.design/fluent-emoji-flat/package.svg" width="24" /> &nbsp; Other things I've shipped

</div>

<br/>

<table>
<tr>
<td width="50%" valign="top">

#### <img src="https://api.iconify.design/lucide/shield-check.svg?color=%23818cf8" width="20" /> &nbsp;[DataGuard](https://github.com/Suhxs-Reddy/KiroHackathon)
*Chrome extension · Llama 3.1*

Privacy policies are 5,000 words of legal jargon nobody reads. So you click "Accept" without knowing if a site sells your location. **I let an LLM do the reading for you** — Llama 3.1 parses the policy, breach-checks the domain on Have I Been Pwned, and generates one-click GDPR/CCPA opt-outs with pre-filled emails and follow-up reminders.

<sub>`TypeScript` · `Chrome MV3` · `Llama 3.1 8B`</sub>

</td>
<td width="50%" valign="top">

#### <img src="https://api.iconify.design/lucide/graduation-cap.svg?color=%23fb7185" width="20" /> &nbsp;[AI Study Buddy](https://github.com/Suhxs-Reddy/CBC-Hackathon)
*Canvas LMS × Claude RAG*

Generic chatbots hallucinate when your course material isn't in their training data. **I built a RAG tutor that knows when to shut up** — retrieves only from your real Canvas docs, with a low-relevance threshold that flags out-of-syllabus questions instead of inventing answers.

<sub>`React` · `FastAPI` · `Claude` · `Supermemory`</sub>

</td>
</tr>
<tr>
<td width="50%" valign="top">

#### <img src="https://api.iconify.design/lucide/heart-pulse.svg?color=%23f87171" width="20" /> &nbsp;[Hospital Unsupervised Learning](https://github.com/Suhxs-Reddy/DShospitalproject)
*Patient clustering*

Pattern discovery on hospital records — finding structure in unlabeled patient data when you don't yet know what the labels should be.

<sub>`Jupyter` · `scikit-learn`</sub>

</td>
<td width="50%" valign="top">

#### <img src="https://api.iconify.design/lucide/shopping-cart.svg?color=%2338bdf8" width="20" /> &nbsp;[Walmart Demand Forecast](https://github.com/Suhxs-Reddy/prediction)
*Time-series · 45 stores*

Store-level weekly demand with lagged/rolling features, holiday flags, and chronological splits — the boring forecasting that actually keeps shelves stocked.

<sub>`Python` · `time-series`</sub>

</td>
</tr>
</table>

<br/>

---

<br/>

<div align="center">

## <img src="https://api.iconify.design/fluent-emoji-flat/satellite-antenna.svg" width="24" /> &nbsp; Currently

</div>

<br/>

<table>
<tr>
<td width="50%" valign="top">

📖 &nbsp; **Reading**
- Perez et al. — *FiLM: Visual Reasoning with a General Conditioning Layer* (2018)
- Salesforce — *Moirai* time-series foundation model
- Anything on **applying foundation models to physical infrastructure**

</td>
<td width="50%" valign="top">

🛠️ &nbsp; **Building**
- CATI training on Colab A100 — moving toward HF Spaces deploy
- COLLIDE — Vercel deploy + scaling beyond ERCOT to CAISO + PJM
- Research Assistant work at ASU College of Health Solutions

</td>
</tr>
</table>

<br/>

---

<br/>

<div align="center">

## <img src="https://api.iconify.design/fluent-emoji-flat/hammer-and-wrench.svg" width="24" /> &nbsp; Stack

<br/>

<img src="https://skillicons.dev/icons?i=python,pytorch,sklearn,fastapi,docker,aws,ts,react,nextjs,tailwind,vercel,git&perline=6&theme=dark" />

<br/>

<details>
<summary>📋 &nbsp;<b>full stack breakdown</b></summary>

<br/>

| Layer | Tools |
|---|---|
| **ML / Data** | Python · PyTorch · scikit-learn · Pandas · NumPy |
| **Backend** | FastAPI · Docker · AWS (S3, Athena) · MySQL · MongoDB |
| **Frontend** | TypeScript · React · Vite · Tailwind |
| **AI** | Anthropic Claude · LangGraph · HuggingFace · Llama · YOLO |
| **Observability** | Grafana · Tableau |

</details>

</div>

<br/>

---

<br/>

<div align="center">

## <img src="https://api.iconify.design/fluent-emoji-flat/bar-chart.svg" width="24" /> &nbsp; Activity

<br/>

<img src="https://github-readme-stats.vercel.app/api?username=Suhxs-Reddy&show_icons=true&theme=github_dark&hide_border=true&bg_color=0d1117&title_color=4ade80&icon_color=fbbf24&text_color=e5e7eb&count_private=true&include_all_commits=true&rank_icon=github" height="180" />
&nbsp;&nbsp;
<img src="https://github-readme-streak-stats.herokuapp.com/?user=Suhxs-Reddy&theme=github-dark-blue&hide_border=true&background=0d1117&ring=4ade80&fire=fb7185&currStreakLabel=4ade80&sideLabels=e5e7eb&dates=9ca3af" height="180" />

<br/>
<br/>

<img src="https://github-readme-stats.vercel.app/api/top-langs/?username=Suhxs-Reddy&layout=compact&theme=github_dark&hide_border=true&bg_color=0d1117&title_color=4ade80&text_color=e5e7eb&langs_count=8" height="160" />

<br/>
<br/>

<img src="https://github-readme-activity-graph.vercel.app/graph?username=Suhxs-Reddy&theme=react-dark&hide_border=true&bg_color=0d1117&color=4ade80&line=4ade80&point=fbbf24&area=true&area_color=4ade80" width="100%" />

</div>

<br/>

---

<br/>

<div align="center">

## <img src="https://api.iconify.design/fluent-emoji-flat/milky-way.svg" width="24" /> &nbsp; The path here

</div>

<br/>

<table>
<tr>
<td align="center" width="100"><b>When</b></td>
<td><b>What</b></td>
</tr>
<tr>
<td align="center">2026 →</td>
<td>🔬 &nbsp; <b>Research Assistant</b> · ASU College of Health Solutions</td>
</tr>
<tr>
<td align="center">2026</td>
<td>⚡ &nbsp; <b>ASU Energy Hackathon</b> · COLLIDE — built the whole pipeline solo</td>
</tr>
<tr>
<td align="center">2025 → 27</td>
<td>🎓 &nbsp; <b>MS Data Science</b> · Arizona State University</td>
</tr>
<tr>
<td align="center">2025</td>
<td>📊 &nbsp; <b>Data Science Intern</b> · GlobalLogic · Auth/RBAC platform analytics</td>
</tr>
<tr>
<td align="center">2024</td>
<td>💻 &nbsp; <b>SWE Intern</b> · GlobalLogic · GenAI platform backend</td>
</tr>
<tr>
<td align="center">2021 → 25</td>
<td>🎓 &nbsp; <b>B.Tech CS</b> (Big Data minor) · MIT Manipal</td>
</tr>
</table>

<br/>

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Suhxs-Reddy/Suhxs-Reddy/output/github-contribution-grid-snake-dark.svg" />
  <img alt="contribution snake" src="https://raw.githubusercontent.com/Suhxs-Reddy/Suhxs-Reddy/output/github-contribution-grid-snake.svg" width="100%" />
</picture>

<br/>
<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:4ade80,100:0d1117&height=120&section=footer" width="100%" />

</div>
