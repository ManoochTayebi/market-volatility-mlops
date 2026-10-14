# Market Volatility Forecasting — MLOps Platform

**A cloud-integrated, end-to-end MLOps engineering project for financial time-series forecasting.**

[![CI](https://github.com/ManoochTayebi/volatility-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/ManoochTayebi/volatility-mlops/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/ML-PyTorch%20GRU-orange)
![Azure](https://img.shields.io/badge/Cloud-Microsoft%20Azure-0078D4)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2)

This repository demonstrates how to connect **market-data ingestion, cloud storage, time-series model training, experiment tracking, artifact delivery, automated workflows, and a user-facing prediction service**. It uses a PyTorch gated recurrent unit (GRU) network to forecast stock-return volatility, with a FastAPI backend and browser-based asset/portfolio dashboards.

**Project scope:** An end-to-end *MLOps architecture and implementation*, rather than a claim of an always-healthy production service. The deployment workflow has previously succeeded; scheduled Azure retraining is currently blocked by cloud-access configuration. See [Engineering status](#engineering-status) for verified evidence and remaining gaps.

## Architecture

`mermaid
flowchart TB
    GH["GitHub Actions<br/>CI · scheduled/manual training · deployment"]
    TD["Twelve Data API<br/>Daily OHLCV"]
    ING["Incremental ingestion<br/>Validation/preflight"]
    SQL[("Azure SQL<br/>Historical market data")]
    AZML["Azure Machine Learning<br/>Command job"]
    PREP["Feature pipeline<br/>Log returns → rolling realized volatility → sequences"]
    TRAIN["PyTorch GRU<br/>Per-asset training"]
    EVAL["Chronological holdout<br/>MSE · RMSE · MAE"]
    MLF["MLflow<br/>Experiments + model artifacts"]
    GATE{"Optional RMSE gate<br/>for registration"}
    REG["MLflow Model Registry<br/>(opt-in)"]
    BLOB[("Azure Blob Storage<br/>Latest model + volatility artifacts")]
    ACR["Azure Container Registry<br/>Versioned application image"]
    ACA["Azure Container Apps<br/>FastAPI + static web UI"]
    USERS["Browser / API clients"]

    GH -->|submit| AZML
    TD --> ING --> SQL
    AZML --> ING
    SQL --> PREP --> TRAIN --> EVAL
    TRAIN --> MLF
    EVAL --> MLF
    EVAL --> GATE
    GATE -->|enabled and passes| REG
    TRAIN --> BLOB
    GH -->|Docker build + push| ACR --> ACA
    BLOB -->|download when predictor initializes| ACA
    USERS <-->|REST API| ACA
`

The **training/data plane** runs as an Azure ML command job initiated by GitHub Actions. Separately, the **serving/deployment plane** builds a container in Azure Container Registry and deploys it to Azure Container Apps. Model files are transferred through Azure Blob Storage; the current serving code does **not** load approved model versions from the registry.

## What this project demonstrates

| MLOps layer | Implementation | Status |
| --- | --- | --- |
| Data ingestion | Twelve Data daily OHLCV API, incremental updates and Azure SQL upserts | Implemented |
| Data preparation | Log returns, rolling realized volatility, GRU input sequences | Implemented |
| Model training | Configurable PyTorch GRU, per-symbol retraining, optional warm start from local weights | Implemented |
| Evaluation | Chronological train/test split; MSE, RMSE and MAE | Implemented; horizon correction needed |
| Experiment tracking | MLflow nested runs, parameters, metrics and artifacts | Implemented |
| Model registry | Optional MLflow registration with an RMSE threshold | Available; disabled in Azure job |
| Artifact storage | Models and realized-volatility histories uploaded to Azure Blob | Implemented |
| Continuous integration | Pytest + Docker build via GitHub Actions | Implemented; previously passing |
| Training orchestration | Scheduled/manual GitHub Actions → Azure ML job | Implemented; current cloud run failing |
| Container deployment | Docker → Azure Container Registry → Azure Container Apps | Implemented; previously deployed |
| Model serving | FastAPI + web interface with on-demand artifact download | Implemented; requires available artifacts |
| Monitoring / promotion | Basic health endpoint and logs; no enforced production model approval or drift monitoring | Further engineering required |

## ML methodology

The modeling flow is built around **realized volatility**, calculated as a rolling standard deviation of log returns from historical closing prices. For each symbol (for example, `AAPL`, `GOOGL`, or `MSFT`):

1. Retrieve time-ordered OHLCV records from Azure SQL.
2. Compute log returns and rolling realized volatility.
3. Assemble fixed-length sequences (`WINDOW_SIZE=21` by default).
4. Split sequences chronologically into training and test partitions.
5. Train a compact PyTorch GRU with Adam and MSE loss.
6. Report holdout MSE, RMSE and MAE, log the experiment in MLflow, and save artifacts.
7. Upload model checkpoints and volatility histories to Blob Storage when cloud artifact upload is enabled.

The API provides iterative multi-step forecasts using saved GRU checkpoints. **Known correctness issue:** the current training sequence target is indexed two observations ahead, while the API presents forecasts as one step ahead. Correcting this alignment and retesting performance is a priority before interpreting reported metrics as one-day-ahead forecasting accuracy.

The portfolio endpoint computes an **equal-weight, zero-correlation approximation** to portfolio volatility; it is not a full covariance-aware portfolio risk model.

## Repository map

`text
.
├── .github/workflows/
│   ├── ci.yml                          # Pytest + Docker build
│   ├── azure-mlops-pipeline.yml        # Scheduled Azure ML retraining
│   └── azure-container-app-deploy.yml # ACR build + Container Apps deployment
├── azure/
│   ├── ml/                             # Azure ML job + Conda environment
│   ├── sql/                            # Database schema
│   └── setup_low_cost_resources.sh     # Infrastructure provisioning helper
├── backend/
│   ├── app.py                          # FastAPI routes + frontend hosting
│   ├── compute.py                      # Asset/portfolio API logic
│   ├── dataset.py                      # Returns, realized volatility, sequences
│   ├── nn_trainer.py                   # GRU and training/evaluation
│   ├── predictor.py                    # Model loading and inference
│   └── azure_artifacts.py              # Blob artifact download
├── frontend/                           # HTML / CSS / JavaScript dashboards
├── scripts/
│   ├── preflight_check.py              # Configuration + Azure SQL checks
│   ├── ingest_market_data.py           # Historical/incremental ingestion
│   ├── retrain_with_mlflow.py          # Per-asset training + tracking
│   ├── run_daily_pipeline.py           # Pipeline entry point
│   ├── upload_artifacts_to_azure_blob.py
│   └── smoke_test_api.py               # Deployed /api/health check
├── src/
│   ├── twelve_data_client.py
│   └── azure_sql_connect.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── .env.example
`

## Running locally

**Prerequisites:** Python 3.11. To train against the configured cloud data source, you also need a Twelve Data API key, an accessible Azure SQL database, and corresponding credentials. To serve predictions, trained artifacts must exist locally or be accessible from configured Azure Blob Storage.

`bash
git clone https://github.com/ManoochTayebi/volatility-mlops.git
cd volatility-mlops

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Set your own configuration in .env (never commit credentials).

python -m pytest -q
uvicorn backend.app:app --host 0.0.0.0 --port 8000
`

Open `http://localhost:8000` for the frontend and `http://localhost:8000/docs` for interactive API documentation.

**Try the API:**

`bash
curl http://localhost:8000/api/health

# Requires a trained AAPL model and realized-volatility artifact:
curl "http://localhost:8000/api/asset_volatility?asset=AAPL&days=5&historical_period=30"

# Requires corresponding model artifacts for both assets:
curl -X POST "http://localhost:8000/api/portfolio_volatility" \
  -H "Content-Type: application/json" \
  -d '{"assets":["AAPL","MSFT"],"days":5,"historical_period":30}'
`

You can also start the app and a local MLflow UI with `docker compose up --build` after configuring `.env`. The UI runs at `http://localhost:8000` and MLflow at `http://localhost:5001`.

### Data and training commands

`bash
make preflight      # Verify API/SQL configuration and SQL access
make ingest-full    # Load historical prices
make ingest-daily   # Incremental daily OHLCV update
make retrain        # Train GRU models and track in MLflow
make pipeline       # Preflight → ingestion → retraining → optional Blob upload
`

The default market data source is `azure_sql`. For a different source, see the `MARKET_DATA_SOURCE` configuration and `backend/dataset.py`. The training commands are not intended to work without the associated data source and credentials.

## CI, retraining and deployment

| Workflow | Trigger | Responsibility |
| --- | --- | --- |
| [CI](.github/workflows/ci.yml) | Push, PR, manual | Install dependencies, run tests, build Docker image |
| [Azure MLOps Pipeline](.github/workflows/azure-mlops-pipeline.yml) | Scheduled (Tue–Sat UTC), push, manual | Authenticate to Azure, provision/check ML compute, submit/stream data ingestion and GRU training |
| [Azure Container App Deploy](.github/workflows/azure-container-app-deploy.yml) | Push, manual | Build/push image to ACR, update Container App, inject storage secret, health smoke test |

The Azure ML job definition is [`azure/ml/daily_pipeline_job.yml`](azure/ml/daily_pipeline_job.yml). Its command runs preflight checks, daily market ingestion, training/MLflow logging and Blob artifact upload. The deployment workflow builds the FastAPI + frontend container separately. A successful `GET /api/health` smoke test checks the running application, **not** the availability or predictive validity of model artifacts.

### Cloud configuration

Use [`.env.example`](.env.example) for configuration names and [the Azure setup guide](docs/azure_mlops_setup.md) for infrastructure instructions.

**GitHub Actions secrets:** `AZURE_CREDENTIALS`, `TWELVE_DATA_API_KEY`, `AZURE_SQL_SERVER`, `AZURE_SQL_DATABASE`, `AZURE_SQL_USERNAME`, `AZURE_SQL_PASSWORD`, `AZURE_STORAGE_CONNECTION_STRING`.

**Repository variables:** `AZURE_RESOURCE_GROUP`, `AZURE_LOCATION`, `AZURE_ML_WORKSPACE`, `AZURE_COMPUTE_NAME`, `AZURE_ACR_NAME`, `AZURE_CONTAINER_APP`, `AZURE_CONTAINER_APP_ENV`, `AZURE_SQL_TABLE`, `SYMBOLS`, `AZURE_MODEL_ARTIFACTS_CONTAINER`, `AZURE_MODEL_ARTIFACTS_PREFIX`.

No real credentials should appear in Git. The Azure setup script uses scale-down-conscious development defaults (Azure ML compute minimum zero, Container App replicas minimum zero), but deployed services may still incur charges.

### Model registration and quality gates

`scripts/retrain_with_mlflow.py` supports **optional** MLflow model registration and an RMSE threshold:

`dotenv
MLFLOW_REGISTER_MODELS=true
MODEL_MAX_RMSE=0.05
ENFORCE_MODEL_QUALITY_GATE=true
`

This is an example threshold; it must be selected from a sound validation procedure for the target and units. In the checked-in Azure ML configuration, registration is **disabled**, and artifact upload to the shared `latest/` Blob prefix is **not gated by model approval**. Production-style candidate/champion evaluation, immutable version promotion and rollback remain to be added.

## Engineering status

This is an implemented full-lifecycle **portfolio project** with historical cloud execution, not a currently verified production service.

- **CI:** Tests and Docker build passed on [June 29, 2026](https://github.com/ManoochTayebi/volatility-mlops/actions/runs/28361009891).
- **Cloud serving:** ACR build, Azure Container Apps deployment and the health smoke test passed on [June 26, 2026](https://github.com/ManoochTayebi/volatility-mlops/actions/runs/28235049696); the following [June 29 deployment attempt](https://github.com/ManoochTayebi/volatility-mlops/actions/runs/28361009917) failed at image build/push.
- **Azure ML retraining:** The latest checked run, on [August 28, 2026](https://github.com/ManoochTayebi/volatility-mlops/actions/runs/33180593073), stopped at `azure/login` with `No subscriptions found`. Older project notes also record an Azure SQL login blocker. Fixing subscription/service-principal access comes before another cloud end-to-end verification.
- **Limitations to address:** One-step target alignment; model-vs-baseline validation; enforced registry promotion; tested model-serving readiness; trading-calendar dates; correlation-aware portfolio risk; drift/quality monitoring and rollback.

These limitations are documented intentionally so the architecture, implementation and empirical verification are not conflated.

## Next engineering milestones

1. **Correct forecasting semantics:** align training targets with serving horizons and add time-series backtesting and baseline comparisons.
2. **Restore cloud execution:** verify service-principal subscription permissions, Azure SQL connectivity and current ACR deployment.
3. **Make model promotion explicit:** version candidate artifacts; evaluate against baseline/champion; promote approved versions only; make serving load a pinned approved model.
4. **Strengthen operational quality:** test `asset_volatility` with real artifacts, add readiness and data freshness checks, monitor prediction quality/drift, and support rollback.

---

**Stack:** Python · Pandas · PyTorch · MLflow · FastAPI · Docker · GitHub Actions · Azure Machine Learning · Azure SQL · Azure Blob Storage · Azure Container Registry · Azure Container Apps · HTML/CSS/JavaScript

**Purpose:** Demonstrate practical ML engineering and cloud MLOps design through a real financial forecasting application, including the operational challenges of automating the data-to-serving lifecycle.
