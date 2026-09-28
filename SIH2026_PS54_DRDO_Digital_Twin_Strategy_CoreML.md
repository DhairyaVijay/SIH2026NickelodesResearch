# SIH 2026 — DRDO PS54: AI-Enabled Real-Time Digital Twin for Aero Piston Engine Health Monitoring

**Constraint envelope (hard-locked):** React + Vite + Mapbox (frontend) · FastAPI + WebSockets + PostgreSQL + Docker/Docker Compose (backend) · Scikit-Learn (Pipelines, KNNImputer), XGBoost, SVM, SMOTE (ML — classical/core ML only, no deep learning) · Groq API + Instructor (GenAI). No Kafka/MQTT/RabbitMQ, no Redis, no NoSQL, no cloud PaaS.

---

## 1. The Build & Solve Strategy

### 1.1 The core problem framing
DRDO PS54 asks for a **digital twin** — not just a classifier. A digital twin has three properties your architecture must explicitly demonstrate to judges: (a) a live, streaming virtual replica of engine state, (b) a predictive model of future degradation (RUL — Remaining Useful Life), and (c) closed-loop decision output (fault → recommendation). Build all three layers visibly, not just a dashboard with numbers.

### 1.2 Data reality check (do this first, Day 1)
No public real-world piston aero-engine telemetry dataset exists at the fidelity DRDO wants — so drop the plan to hand-build a synthetic physics simulator. Instead, **train and test exclusively on NASA C-MAPSS** (turbofan RUL dataset): it's the only dataset with real, physically-grounded, run-to-failure, multivariate sensor telemetry publicly available at the fidelity this problem needs, and it structurally *is* your problem (multivariate sensor streams → RUL/fault class), not a stand-in for it.

**What this means concretely:**
- `train_FD001–FD004.txt` are your **only** training data — no fabricated CHT/EGT/RPM curves, no injected noise you invented, no hand-picked degradation slopes. Every number your models see is real recorded turbofan sensor data.
- `test_FD001–FD004.txt` + `RUL_FD001–FD004.txt` are your **only** evaluation data — RUL scoring is against NASA's ground truth, not a self-generated one.
- Start with **FD001** (single operating condition, single fault mode — HPC degradation) to get a clean baseline, then progress to FD002/FD003/FD004 for multi-condition and multi-fault robustness, exactly the way the published benchmarks structure the problem.
- Column schema: col 1 = `unit_number`, col 2 = `time_cycle`, cols 3–5 = operational settings (altitude, Mach number, throttle resolver angle), cols 6–26 = 21 sensor channels (T2, T24, T30, T50, P2, P15, P30, Nf, Nc, epr, Ps30, phi, NRf, NRc, BPR, farB, htBleed, Nf_dmd, PCNfR_dmd, W31, W32). Sensors 1, 5, 6, 10, 16, 18, 19 are near-constant in FD001 and should be dropped during feature selection.
- RUL labeling: use the standard **piecewise-linear RUL clipping** convention (cap raw cycles-to-failure at 125 cycles) — this is what makes your reported numbers directly comparable to published C-MAPSS benchmarks, which matters when a DRDO panel checks your claims.

**Framing for the judges (say this explicitly, don't dodge it):** the problem statement targets aero piston engines, and no public piston-engine dataset exists. Rather than fabricate one — which is unverifiable and easy for a domain-engineer judge to poke holes in — you're demonstrating the full prognostics pipeline (streaming ingestion → feature engineering → fault classification → RUL regression → GenAI decision layer) end-to-end on the best publicly available, real, physically-accurate aero-engine degradation dataset, and the entire architecture is sensor-schema-agnostic: swapping in real piston-engine telemetry later requires no pipeline redesign, only a config change to the sensor list. This is a stronger, more defensible claim than "trust our synthetic data."

### 1.3 End-to-end architecture (strictly in-stack)

```
[C-MAPSS Replay Streamer (Python asyncio)]
        │  reads rows from train/test_FD00X.txt sequentially per engine_id and
        │  streams them as JSON frames over WS at 1–20Hz (configurable), replaying
        │  real recorded sensor trajectories to simulate live onboard telemetry —
        │  no values are invented, only the playback timing is simulated
        ▼
[FastAPI WebSocket Ingestion Gateway]
        │  Pydantic schema validation → reject malformed frames
        │  writes raw frame to PostgreSQL (append-only telemetry table, BRIN-indexed on timestamp)
        │  pushes frame into an in-process asyncio.Queue (this IS your "message broker" —
        │  a single-process pub/sub substitute for Kafka/MQTT, fully justified for a
        │  single-instance hackathon deployment)
        ▼
[Preprocessing Worker (asyncio background task, same FastAPI process)]
        │  Scikit-Learn Pipeline: KNNImputer (sensor dropout) → StandardScaler →
        │  rolling-window feature extraction (rate-of-change, rolling mean/std over
        │  last N frames, held in a local Python deque — this replaces Redis as your
        │  "hot state cache": bounded in-memory ring buffer per engine ID, rebuilt
        │  from PostgreSQL on restart)
        ▼
[Prediction Router (FastAPI service layer)]
        │  Tier 1 (every frame, <50ms):  XGBoost / SVM fault classifier → engine state
        │                                 {NOMINAL, DEGRADING, CRITICAL}
        │  Tier 2 (every N seconds):      XGBoost / Gradient-Boosted Tree regressor → RUL
        │                                 estimate, fed on the rolling-window engineered
        │                                 features (rate-of-change, rolling mean/std/slope
        │                                 over the last N frames, sensor-vs-physics-baseline
        │                                 residuals) — this IS your "digital twin": a
        │                                 continuously-updated, feature-engineered replica
        │                                 of engine degradation state, built on interpretable
        │                                 tree ensembles instead of a learned black-box
        │                                 temporal model
        │  Router logic: Tier 1 result gates whether Tier 2 runs at high frequency
        │  (escalates inference rate under DEGRADING/CRITICAL states — a resource-aware
        │  design point worth calling out to judges)
        ▼
[PostgreSQL — predictions table]   [Groq + Instructor GenAI Layer]
        │  stores every prediction        │  consumes structured prediction JSON
        │  with model version + timestamp │  (Pydantic schema, NOT free text) →
        │                                 │  generates structured mission-report JSON:
        │                                 │  {risk_level, root_cause_hypothesis,
        │                                 │   recommended_action, mission_go_no_go,
        │                                 │   confidence, human_readable_summary}
        ▼                                 ▼
[FastAPI REST + WS broadcast layer] ──────┘
        ▼
[React + Vite Frontend]
        │  WebSocket client: live sensor charts, engine health gauge
        │  Mapbox: UAV position + route, marker color-coded by live health state
        │  GenAI report panel: renders the Instructor-validated JSON as a
        │  structured "Flight Engineer" card, not raw LLM text
```

### 1.4 Why the "no Kafka/no Redis" constraint is a feature, not a limitation
Frame this explicitly in your report/demo: a MALE UAV ground control station is a **single deployable unit** in the field, not a distributed cloud system. A monolithic, Docker-Composed FastAPI service with in-process queuing and PostgreSQL as the single source of truth is *more* defensible for a defense/field deployment than a Kafka/Redis microservice sprawl — lower attack surface, no network dependency between broker and consumer, easier to air-gap. This reframes your constraint as a deliberate architectural decision for the DRDO context.

### 1.5 Database schema sketch (PostgreSQL only)
- `telemetry_raw(id, engine_id, ts, cycle, op_setting_1, op_setting_2, op_setting_3, sensors JSONB)` — `sensors` holds the 21 C-MAPSS sensor readings (`sensor_1`…`sensor_21`) keyed by name so dropped/dead sensors don't force schema migrations; BRIN index on `ts`
- `predictions(id, engine_id, ts, model_tier, fault_class, rul_estimate, confidence, model_version)`
- `genai_reports(id, engine_id, ts, risk_level, root_cause, recommended_action, mission_go_no_go, raw_json)`
- `engine_registry(engine_id, dataset_source, fd_subset, install_date, total_flight_cycles)` — for fleet-level view; `dataset_source`/`fd_subset` track which C-MAPSS file (FD001–FD004) and unit_number each `engine_id` maps to

---

## 2. The Unique X-Factors

### X-Factor 1 — Physics-Informed Residual Twin (hybrid model, not a black box)
Don't let judges see "we trained XGBoost on numbers." Implement a lightweight physics baseline computed analytically in Python: standard turbofan gas-path relations (e.g., expected T30/T50 rise and Ps30/P30 pressure ratio given the operational settings — altitude, Mach number, TRA — using published compressor/turbine thermodynamic approximations consistent with the C-MAPSS engine model documentation) for the sensors you're actually training on. Feed the **residual** (actual sensor value − physics-predicted value) into your ML models as a feature. This is the actual definition of a digital twin used in aerospace literature (physics model + data-driven correction), and because it's grounded in the same published C-MAPSS/turbofan relations your dataset comes from, it's directly defensible to a DRDO panel that will contain domain engineers.

### X-Factor 2 — GenAI "Digital Flight Engineer" with Instructor-Enforced Schema
Most SIH GenAI integrations are a chatbot bolted on the side. Yours should be a **structured decision system**: Groq + Instructor forces every LLM call into a strict Pydantic schema (`risk_level: Literal[...]`, `root_cause_hypothesis: str`, `mission_go_no_go: bool`, `evidence: list[SensorEvidence]`). Because Instructor guarantees valid JSON, this output can safely drive UI state and even simulated mission-abort triggers live in the demo — "LLM output directly gates a mission-critical decision path" is a strong, safe-to-demo claim because the schema makes it deterministic and auditable, not a hallucination risk.

### X-Factor 3 — Real-Data Replay Engine as a Standalone, Dataset-Agnostic Deliverable
Present your C-MAPSS replay streamer itself as a product artifact, not just a data-loading script: a parameterized live-telemetry replayer that reads any FD00X subset, streams it engine-by-engine at a configurable rate, and can support fault-severity filtering (e.g., replay only engines whose degradation trajectory crosses into DEGRADING/CRITICAL within N cycles) for repeatable demo scenarios. Frame it explicitly as solving DRDO's real bottleneck — **the absence of labeled failure data for indigenous aero piston engines** — not by fabricating numbers, but by proving the entire ingestion-to-decision pipeline against real, physically accurate run-to-failure data, with a sensor schema abstraction layer that makes swapping in real piston-engine telemetry (once available) a configuration change, not a rebuild. This directly targets the "no real dataset exists" gap judges will otherwise flag against you, without the credibility risk of self-generated data.

### X-Factor 4 — Mission-Context-Aware Geospatial Health Overlay (Mapbox)
Most competing teams will show an isolated engine dashboard. Use Mapbox to render the UAV's simulated flight path with markers color-coded to live engine health state (green/amber/red), and correlate flight-profile events (climb, throttle change, loiter) with degradation onset in a synchronized timeline scrubber. This ties engine health to **mission reliability** — the exact phrase in the problem statement — rather than presenting engine health in isolation.

---

## 3. The 40-Day Prep & 35-Hour Hackathon Master Plan

### 3.1 Forty-Day Preparation Plan

**Week 1 (Days 1–7) — Foundations & Data**
- Days 1–2: Deep-dive predictive maintenance theory — RUL estimation, sensor degradation curves, piecewise-linear RUL labeling (the C-MAPSS convention). Read NASA C-MAPSS documentation and 2–3 reference papers on gradient-boosted-tree/feature-engineered RUL prediction.
- Days 3–4: Download and fully EDA all four C-MAPSS subsets (FD001–FD004) in a notebook; identify and drop near-constant sensors per subset; replicate a baseline RUL regressor (XGBoost) on FD001 to internalize the windowing/labeling pattern.
- Days 5–7: Build v1 of the **C-MAPSS Replay Streamer** (Python, no infra yet) — reads `train_FD00X.txt`/`test_FD00X.txt` row-by-row per engine, emits JSON frames at a configurable rate, CSV/stdout output for now. No fabricated data — this is a pure real-data playback utility.

**Week 2 (Days 8–14) — ML Core**
- Days 8–9: Build the Scikit-Learn preprocessing Pipeline (KNNImputer, scaling, rolling-window feature extraction) against C-MAPSS training data.
- Days 10–11: Train XGBoost + SVM fault classifiers on C-MAPSS (derive NOMINAL/DEGRADING/CRITICAL bins from RUL thresholds); apply SMOTE for rare-failure class imbalance; evaluate with precision/recall (not just accuracy — failure classes are rare).
- Days 12–14: Engineer the rolling-window feature set for RUL regression (rolling mean/std/slope, rate-of-change, cycles-since-last-window, physics-residual features from X-Factor 1) and train a core-ML regressor (XGBoost Regressor or Gradient-Boosted/Random-Forest ensemble) for sequence-aware RUL estimation, trained and validated entirely on C-MAPSS (FD001 baseline, then FD002–FD004 for robustness). Tune tree depth/estimators against a held-out validation split to avoid overfitting the engineered windows. Establish a model-versioning convention now (you'll need it for the predictions table).

**Week 3 (Days 15–21) — Backend Skeleton**
- Days 15–16: FastAPI project scaffold, PostgreSQL schema migration, Docker Compose (FastAPI + Postgres services) — get `docker compose up` working end-to-end early.
- Days 17–18: WebSocket ingestion endpoint + asyncio.Queue pattern; wire the C-MAPSS Replay Streamer to stream live over WS instead of static CSV output.
- Days 19–21: Wire the prediction router (Tier 1/Tier 2 logic) to consume from the queue and write to `predictions`. This is your riskiest integration point — de-risk it early.

**Week 4 (Days 22–28) — GenAI + Frontend Bring-up**
- Days 22–23: Groq + Instructor prototyping — design the Pydantic report schema, test prompt reliability against varied fault scenarios.
- Days 24–25: React + Vite scaffold; WebSocket client hook; live sensor line charts.
- Days 26–28: Mapbox integration — static demo route first, then live marker color-binding to health state.

**Week 5 (Days 29–35) — Integration**
- Days 29–31: Full pipeline integration test — C-MAPSS Replay Streamer → WS → DB → models → GenAI → frontend, single command via Docker Compose.
- Days 32–33: Select and replay real degrading-to-failure engine trajectories from the C-MAPSS test set end-to-end (e.g., an FD001 unit crossing DEGRADING → CRITICAL); verify GenAI report quality and latency (Groq should keep this comfortably sub-2s).
- Days 34–35: Fix integration bugs found under sustained streaming load (memory growth in the deque buffers, WS reconnect handling).

**Week 6 (Days 36–40) — Polish, Rehearsal, Narrative**
- Day 36: Performance pass — DB indices, batched writes, Tier-2 inference throttling.
- Day 37: Build the pitch deck; nail the framing for X-Factors 1–4.
- Day 38: Full dry-run demo, timed, with the real C-MAPSS demo-trajectory replay script rehearsed.
- Day 39: Buffer day — fix whatever broke in the dry run.
- Day 40: Rest. Pack devices, chargers, offline fallback video of the working demo.

### 3.2 Thirty-Five-Hour Hackathon Plan

| Hours | Focus | Deliverable |
|---|---|---|
| 0–3 | Repo setup, Docker Compose skeleton (Postgres + FastAPI containers healthy), DB schema applied | `docker compose up` works, empty tables exist |
| 3–8 | Port pre-trained models (XGBoost/SVM fault classifier, XGBoost Regressor RUL model) from prep phase into the FastAPI service; load the C-MAPSS Replay Streamer, confirm WS streaming end-to-end | Live telemetry frames hitting DB |
| 8–14 | Wire Tier 1/Tier 2 prediction router; confirm predictions land in `predictions` table with correct latency profile | Fault classification + RUL live |
| 14–18 | Integrate Groq + Instructor GenAI layer against live prediction stream; validate schema robustness against edge-case fault combos | Structured mission reports generating live |
| 18–20 | **Sleep/rotate shifts** — keep one person on overnight log monitoring only | System stable overnight |
| 20–26 | React frontend: sensor charts, health gauge, GenAI report card, Mapbox route + color-coded markers | Working end-to-end UI |
| 26–29 | Demo-trajectory rehearsal: pre-select 2–3 real C-MAPSS test engines whose recorded degradation crosses into CRITICAL near the truncation point (e.g., an HPC-degradation unit → CRITICAL → mission-abort GenAI report) and script their replay for the live demo | Repeatable demo script |
| 29–32 | Bug fixing, edge-case hardening, record a fallback demo video (mandatory safety net) | Fallback video in hand |
| 32–34 | Deck finalization, X-Factor talking points rehearsed per team member | Deck + speaking roles locked |
| 34–35 | Final buffer, submission upload, team rest | Submitted |

---

## Instructions for Gemini Visuals

Copy-paste each block below into Google Gemini to generate the corresponding interactive widget.

**1. Architecture Flow Diagram**
```
Generate an interactive flowchart widget showing this system architecture as a top-to-bottom data flow diagram: a Python asyncio C-MAPSS replay streamer reads real NASA turbofan sensor trajectories from file and streams them as JSON sensor frames over WebSocket to a FastAPI ingestion gateway; the gateway validates frames with Pydantic and writes them to a PostgreSQL telemetry table while also pushing them into an in-process asyncio.Queue; a preprocessing worker consumes the queue and runs a Scikit-Learn pipeline (KNNImputer, StandardScaler, rolling-window feature extraction using a local deque buffer); a prediction router then splits into two parallel tiers — Tier 1 runs XGBoost/SVM fault classification on every frame, Tier 2 runs an XGBoost/Gradient-Boosted-Tree regressor on rolling-window engineered features for RUL estimation every few seconds — both writing to a PostgreSQL predictions table; a Groq + Instructor GenAI layer consumes structured prediction JSON and produces a structured mission report JSON; finally a FastAPI REST/WebSocket broadcast layer sends live data to a React + Vite frontend with Mapbox geospatial visualization and a GenAI report panel. Make each stage a distinct labeled node with arrows showing data flow direction, and use color coding to distinguish the frontend, backend, ML, and GenAI layers.
```

**2. 40-Day + 35-Hour Gantt/Timeline Chart**
```
Generate an interactive Gantt chart widget covering two phases. Phase 1, "40-Day Prep," spans Day 1 to Day 40 with these sequential week-long tracks: Week 1 "Foundations & Data" (Days 1-7), Week 2 "ML Core" (Days 8-14), Week 3 "Backend Skeleton" (Days 15-21), Week 4 "GenAI + Frontend Bring-up" (Days 22-28), Week 5 "Integration" (Days 29-35), Week 6 "Polish, Rehearsal, Narrative" (Days 36-40). Phase 2, "35-Hour Hackathon," spans Hour 0 to Hour 35 with these tracks: "Setup & Docker" (0-3), "Model Porting & Streaming" (3-8), "Prediction Router" (8-14), "GenAI Integration" (14-18), "Overnight Monitoring" (18-20), "Frontend Build" (20-26), "Demo-Trajectory Rehearsal" (26-29), "Bug Fixing & Fallback Video" (29-32), "Deck Finalization" (32-34), "Submission Buffer" (34-35). Use two visually separated timeline sections, one per phase, with distinct color bands per track and hover tooltips showing the deliverable for each.
```

**3. System Architecture / Tech Stack Map**
```
Generate an interactive architecture map widget organized into four labeled layers, left to right or top to bottom: "Frontend Layer" containing React, Vite, and Mapbox; "Backend Layer" containing FastAPI, WebSockets, PostgreSQL, and Docker Compose; "ML Layer" containing XGBoost Regressor (rolling-window feature-based RUL estimation), XGBoost and SVM (fault classification), Scikit-Learn Pipelines with KNNImputer, and SMOTE for class imbalance; "GenAI Layer" containing Groq API and Instructor for structured JSON mission reports. Connect the layers with directional arrows showing that Frontend receives from Backend, Backend orchestrates ML and GenAI, and ML/GenAI both feed back into Backend. Each technology should be a clickable node showing a one-line description of its specific role in this UAV aero-engine digital twin system when hovered or clicked.
```
