# SIH 2026 — DRDO PS54: AI-Enabled Real-Time Digital Twin for Aero Piston Engine Health Monitoring

Constraint envelope (hard-locked): React + Vite + Mapbox (frontend) · FastAPI + WebSockets + PostgreSQL + Docker/Docker Compose (backend) · Scikit-Learn (Pipelines, KNNImputer), XGBoost, SVM, SMOTE (ML — classical/core ML only, no deep learning) · Groq API + Instructor (GenAI). No Kafka/MQTT/RabbitMQ, no Redis, no NoSQL, no cloud PaaS.
---

## 1. The Build & Solve Strategy

### 1.1 The core problem framing
DRDO PS54 asks for a digital twin — not just a classifier. A digital twin has three properties your architecture must explicitly demonstrate to judges: (a) a live, streaming virtual replica of engine state and (b) a predictive model of future degradation (RUL — Remaining Useful Life) (c) closed-loop decision output (fault → recommendation).

### 1.2 Data reality check (do this first, Day 1)
No public real-world piston aero-engine telemetry dataset exists at the fidelity DRDO wants — so drop the plan to hand-build a synthetic physics simulator. Instead, train and test exclusively on NASA C-MAPSS (turbofan RUL dataset): it's the only dataset with real, physically-grounded, run-to-failure, multivariate sensor telemetry publicly available at the fidelity this problem needs, and it structurally is your problem (multivariate sensor streams → RUL/fault class), not a stand-in for it.

What this means concretely:
- `train_FD001–FD004.txt` are your only training data — no fabricated CHT/EGT/RPM curves, no injected noise you invented, no hand-picked degradation slopes
- `test_FD001–FD004.txt` + `RUL_FD001–FD004.txt` are your only evaluation data — RUL scoring is against NASA's ground truth
- Start with FD001 (single operating condition, single fault mode — HPC degradation) to get a clean baseline, then progress to FD002/FD003/FD004 for multi-condition and multi-fault robustness
- Column schema: col 1 = `unit_number`, col 2 = `time_cycle`, cols 3–5 = operational settings (altitude, Mach number, throttle resolver angle), cols 6–26 = 21 sensor channels (T2, T24, T30, T50, P2, P15, P30, Nf, Nc, epr, Ps30, phi, NRf, NRc, BPR, farB, htBleed, Nf_dmd, PCNfR_dmd, W31, W32). Sensors 1, 5, 6, 10, 16, 18, 19 are near-constant in FD001 and should be dropped during feature selection
- RUL labeling: use the standard piecewise-linear RUL clipping convention (cap raw cycles-to-failure at 125 cycles)

Framing for the judges (say this explicitly, don't dodge it): the problem statement targets aero piston engines, and no public piston-engine dataset exists. Rather than fabricate one — which is unverifiable and easy for a domain-engineer judge to poke holes in — you're demonstrating the full prognostics pipeline (streaming ingestion → feature engineering → fault classification → RUL regression → GenAI decision layer) end-to-end on the best publicly available, real, physically-accurate aero-engine degradation dataset, and the entire architecture is sensor-schema-agnostic: swapping in real piston-engine telemetry later requires no pipeline redesign, only a config change to the sensor list. This is a stronger, more defensible claim than "trust our synthetic data."

### 1.3 End-to-end architecture (strictly in-stack)
```

[C-MAPSS Replay Streamer (Python asyncio)]
│ reads rows from train/test_FD00X.txt sequentially per engine_id and
│ streams them as JSON frames over WS at 1–20Hz (configurable), replaying
│ real recorded sensor trajectories to simulate live onboard telemetry —
│ no values are invented, only the playback timing is simulated
▼
[FastAPI WebSocket Ingestion Gateway]
│ Pydantic schema validation → reject malformed frames
│ writes raw frame to PostgreSQL (append-only telemetry table, BRIN-indexed on timestamp)
│ pushes frame into an in-process asyncio.Queue (this IS your "message broker" —
│ a single-process pub/sub substitute for Kafka/MQTT, fully justified for a
│ single-instance hackathon deployment)
▼
[Preprocessing Worker (asyncio background task, same FastAPI process)]
│ Scikit-Learn Pipeline: KNNImputer (sensor dropout) → StandardScaler →
│ rolling-window feature extraction (rate-of-change, rolling mean/std over
│ last N frames, held in a local Python deque — this replaces Redis as your
│ "hot state cache": bounded in-memory ring buffer per engine ID, rebuilt
│ from PostgreSQL on restart)
▼
[Prediction Router (FastAPI service layer)]
│ Tier 1 (every frame, <50ms): XGBoost / SVM fault classifier → engine state
│                 {NOMINAL, DEGRADING, CRITICAL}
│ Tier 2 (every N seconds):   XGBoost / Gradient-Boosted Tree regressor → RUL
│                 estimate, fed on the rolling-window engineered
│                 features (rate-of-change, rolling mean/std/slope
│                 over the last N frames, sensor-vs-physics-baseline
│                 residuals) — this IS your "digital twin": a
│                 continuously-updated, feature-engineered replica
│                 of engine degradation state, built on interpretable
│                 tree ensembles instead of a learned black-box
│                 temporal model
│ Router logic: Tier 1 result gates whether Tier 2 runs at high frequency
│ (escalates inference rate under DEGRADING/CRITICAL states — a resource-aware
│ design point worth calling out to judges)
▼
[PostgreSQL — predictions table]  [Groq + Instructor GenAI Layer]
│ stores every prediction    │ consumes structured prediction JSON
│ with model version + timestamp │ (Pydantic schema, NOT free text) →
│                 │ generates structured mission-report JSON:
│                 │ {risk_level, root_cause_hypothesis,
│                 │  recommended_action, mission_go_no_go,
│                 │  confidence, human_readable_summary}
▼                 ▼
[FastAPI REST + WS broadcast layer] ──────┘
▼
[React + Vite Frontend]
│ WebSocket client: live sensor charts, engine health gauge
│ Mapbox: UAV position + route, marker color-coded by live health state
│ GenAI report panel: renders the Instructor-validated JSON as a
│ structured "Flight Engineer" card, not raw LLM text
```

### 1.4 Why the "no Kafka/no Redis" constraint is a feature, not a limitation
Frame this explicitly in your report/demo: a MALE UAV ground control station is a single deployable unit in the field, not a distributed cloud system. A monolithic, Docker-Composed FastAPI service with in-process queuing and PostgreSQL as the single source of truth is more defensible for a defense/field deployment than a Kafka/Redis microservice sprawl — lower attack surface, no network dependency between broker and consumer, easier to air-gap. This reframes your constraint as a deliberate architectural decision for the DRDO context.

### 1.5 Database schema sketch (PostgreSQL only)
- `telemetry_raw(id, engine_id, ts, cycle, op_setting_1, op_setting_2, op_setting_3, sensors JSONB)` — `sensors` holds the 21 C-MAPSS sensor readings (`sensor_1`…`sensor_21`) keyed by name so dropped/dead sensors don't force schema migrations; BRIN index on `ts`
- `predictions(id, engine_id, ts, model_tier, fault_class, rul_estimate, confidence, model_version)`
- `genai_reports(id, engine_id, ts, risk_level, root_cause, recommended_action, mission_go_no_go, raw_json)`
- `engine_registry(engine_id, dataset_source, fd_subset, install_date, total_flight_cycles)` — for fleet-level view; `dataset_source`/`fd_subset` track which C-MAPSS file (FD001–FD004) and unit_number each `engine_id` maps to
---

## 2. The Unique X-Factors

### X-Factor 1 — Physics-Informed Residual Twin (hybrid model, not a black box)
Don't let judges see "we trained XGBoost on numbers." Implement a lightweight physics baseline computed analytically in Python: standard turbofan gas-path relations (e.g., expected T30/T50 rise and Ps30/P30 pressure ratio given the operational settings — altitude, Mach number, TRA — using published compressor/turbine thermodynamic approximations consistent with the C-MAPSS engine model documentation) for the sensors you're actually training on. Feed the residual (actual sensor value − physics-predicted value) into your ML models as a feature. This is the actual definition of a digital twin used in aerospace literature (physics model + data-driven correction), and because it's grounded in the same published C-MAPSS/turbofan relations your dataset comes from, it's directly defensible to a DRDO panel that will contain domain engineers.

### X-Factor 2 — GenAI "Digital Flight Engineer" with Instructor-Enforced Schema
Most SIH GenAI integrations are a chatbot bolted on the side. Yours should be a structured decision system: Groq + Instructor forces every LLM call into a strict Pydantic schema (`risk_level: Literal[...]`, `root_cause_hypothesis: str`, `mission_go_no_go: bool`, `evidence: list[SensorEvidence]`). Because Instructor guarantees valid JSON, this output can safely drive UI state and even simulated mission-abort triggers live in the demo — "LLM output directly gates a mission-critical decision path" is a strong, safe-to-demo claim because the schema makes it deterministic and auditable, not a hallucination risk.

### X-Factor 3 — Real-Data Replay Engine as a Standalone, Dataset-Agnostic Deliverable
Present your C-MAPSS replay streamer itself as a product artifact, not just a data-loading script: a parameterized live-telemetry replayer that reads any FD00X subset, streams it engine-by-engine at a configurable rate, and can support fault-severity filtering (e.g., replay only engines whose degradation trajectory crosses into DEGRADING/CRITICAL within N cycles) for repeatable demo scenarios. Frame it explicitly as solving DRDO's real bottleneck — the absence of labeled failure data for indigenous aero piston engines — not by fabricating numbers, but by proving the entire ingestion-to-decision pipeline against real, physically accurate run-to-failure data, with a sensor schema abstraction layer that makes swapping in real piston-engine telemetry (once available) a configuration change, not a rebuild. This directly targets the "no real dataset exists" gap judges will otherwise flag against you, without the credibility risk of self-generated data.

### X-Factor 4 — Mission-Context-Aware Geospatial Health Overlay (Mapbox)
Most competing teams will show an isolated engine dashboard. Use Mapbox to render the UAV's simulated flight path with markers color-coded to live engine health state (green/amber/red), and correlate flight-profile events (climb, throttle change, loiter) with degradation onset in a synchronized timeline scrubber. This ties engine health to mission reliability — the exact phrase in the problem statement — rather than presenting engine health in isolation.
---
 and GenAI, and ML/GenAI both feed back into Backend. Each technology should be a clickable node showing a one-line description of its specific role in this UAV aero-engine digital twin system when hovered or clicked.
```
