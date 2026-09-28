# SIH2026NickelodesResearch

**Smart India Hackathon (SIH) 2026 — DRDO PS54**

This repository contains the end-to-end architecture, C-MAPSS damage propagation modeling, and implementation strategy for DRDO PS54. AeroTwin is a real-time, streaming digital twin designed to monitor UAV engine health, predict Remaining Useful Life (RUL), and autonomously generate mission-critical decisions using structured Generative AI. 

To ensure defense-grade reliability and ease of field deployment, this solution strictly adheres to a constrained, monolithic stack: **No cloud PaaS, no distributed message brokers (Kafka/RabbitMQ), no Redis, and no Deep Learning black boxes.**

---

## 🎯 Core Objectives & Deliverables

1.  **Live Virtual Replica:** A streaming telemetry pipeline that mirrors real-time engine state.
2.  **Predictive RUL Modeling:** Sequence-aware degradation forecasting using classical ML (XGBoost/Gradient Boosted Trees).
3.  **Closed-Loop Decision Output:** A GenAI-driven "Digital Flight Engineer" that outputs structured, schema-enforced mission recommendations (Fault → Action).

## 📊 Data Strategy: The NASA C-MAPSS Approach

**The Challenge:** There is no publicly available, high-fidelity multivariate sensor telemetry dataset for run-to-failure aero piston engines. 
**The Solution:** We train, validate, and demonstrate our pipeline using the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset. 

Rather than fabricating synthetic piston-engine data (which cannot be verified), we demonstrate the full prognostics pipeline on the most rigorous turbofan degradation dataset available. Our architecture is **sensor-schema-agnostic**. Transitioning to indigenous DRDO piston-engine telemetry requires only a configuration change to the sensor list, not a pipeline rebuild.

*   **Training/Evaluation:** Strictly using `train_FD00X.txt` and `test_FD00X.txt` (RUL clipped to 125 cycles).
*   **Sensor Streaming:** A custom Replay Engine streams real C-MAPSS trajectories at configurable frequencies (1–20Hz) to simulate live onboard UAV telemetry.

## 🏗️ System Architecture & Tech Stack

The system is deployed as a single, field-deployable unit via **Docker Compose**.

### Tech Stack
*   **Frontend:** React + Vite, Mapbox (Geospatial routing & health overlays)
*   **Backend:** FastAPI (REST & WebSockets), PostgreSQL (Single source of truth)
*   **Machine Learning (Core ML Only):** Scikit-Learn (Pipelines, KNNImputer), XGBoost, SVM, SMOTE (Imbalance handling)
*   **GenAI Layer:** Groq API + Instructor (Strict JSON schema enforcement)

### Data Flow Pipeline
1.  **Ingestion:** Python asyncio Replay Streamer sends JSON frames over WebSocket to FastAPI.
2.  **State Management:** FastAPI validates (Pydantic), writes to PostgreSQL, and pushes to an in-process `asyncio.Queue` (replacing Kafka/Redis).
3.  **Preprocessing Worker:** Scikit-Learn pipeline handles imputation, scaling, and rolling-window feature extraction in a local Python `deque` ring buffer.
4.  **Prediction Router:** 
    *   **Tier 1 (High Frequency):** XGBoost/SVM Classifier → `[NOMINAL, DEGRADING, CRITICAL]`
    *   **Tier 2 (Low Frequency):** XGBoost Regressor → Estimates RUL based on engineered windows.
5.  **Decision Layer:** Groq + Instructor consumes prediction JSON to output a validated mission report (Risk Level, Root Cause, Go/No-Go).
6.  **Broadcast:** FastAPI streams the live state and GenAI reports back to the React UI.

## 🚀 Key Innovations (X-Factors)

*   **Physics-Informed Residual Twin:** We don't just feed raw data to an XGBoost black box. We compute an analytical physics baseline (standard gas-path relations) and feed the *residuals* (actual vs. expected) into our ML models.
*   **Structured GenAI Flight Engineer:** Using Groq + Instructor, our LLM is forced into a strict Pydantic schema (e.g., `mission_go_no_go: bool`). This makes the GenAI output deterministic, auditable, and safe to use for automated mission-abort triggers.
*   **Dataset-Agnostic Replay Engine:** A standalone utility that solves the lack of real piston data by providing a robust, repeatable live-telemetry playback system for demo scenarios.
*   **Mission-Context Geospatial Overlay:** Engine health isn't viewed in isolation. Mapbox renders the UAV's flight path, binding engine degradation states (Green/Amber/Red) to specific mission phases and geospatial coordinates.

## 📂 Repository Structure

```text
├── README.md                  # Project documentation
├── strategy_doc.md            # Detailed SIH 40-day/35-hour execution strategy
├── docker-compose.yml         # Single-command infrastructure deployment
├── data/
│   ├── raw/                   # C-MAPSS FD001-FD004 datasets
│   └── replay_engine/         # Asyncio streaming playback utility
├── backend/                   # FastAPI application
│   ├── main.py                # WS & REST endpoints
│   ├── database/              # PostgreSQL schemas and models
│   ├── ml_pipeline/           # Scikit-Learn preprocessing, XGBoost models
│   └── genai/                 # Groq + Instructor Pydantic schemas
├── models/                    # Pickled/ONNX classical ML models & Scalers
├── notebooks/                 # EDA, Feature Engineering, Damage Propagation Modeling
└── frontend/                  # React + Vite application
    ├── src/
    │   ├── components/        # Sensor charts, GenAI Report Cards
    │   └── mapbox/            # Geospatial routing and health markers
```
5. **Research and References**
   NASA C-MAPSS-1 Turbofan Engine Degradation Dataset: https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation
   
