<div align="center">

# 📊 RetailPulse
### AI-Powered Customer Analytics & Demand Forecasting Platform
**Predictive Demand • Customer Segmentation • Churn Analysis • Inventory Optimization**

**Project made by Santosh Kumar** · Zidio Development – Data Science & Analytics · March 2026 · v2.0 Industry Edition

![banner](docs/images/00_project_banner.png)
</div>

## 1. Project summary
RetailPulse ingests retail sales, customer and inventory data and delivers **(1)** 30-day demand forecasts (Prophet + LSTM hybrid), **(2)** customer segmentation (RFM, K-Means, DBSCAN), **(3)** churn prediction (XGBoost + SHAP) and **(4)** inventory optimisation (safety stock, reorder point, EOQ, recommended order quantity).
Results are served through a 5-page **Streamlit** dashboard, a **FastAPI** service and a 12-page **Power BI** report, with MLOps: **MLflow** tracking, **PSI/KS drift detection**, an **Airflow** retraining DAG, **Docker**, **Kubernetes** and **GitHub Actions** CI/CD.

![architecture](docs/images/00_architecture_diagram.png)

## 2. Quick start

### A) Google Colab (recommended – no installation)
1. Upload `notebooks/00_RetailPulse_MAIN.ipynb` to Colab (or open from GitHub) → **Runtime → Run all**.
2. **No dataset zip is needed.** `DATASET="auto"` uses (1) a CSV/XLSX you upload (`UPLOAD_MY_FILE=True`) → (2) Kaggle download via `kagglehub` → (3) built-in demo data (never fails).
3. At the end, the reports (`.docx`) are generated and downloaded automatically.

Datasets: [Online Retail II (UCI)](https://www.kaggle.com/datasets/cgrymn/online-retail-ii-uci-dataset) · [Synthetic Retail 10M](https://www.kaggle.com/datasets/mrayyanshehzad/synthetic-retail-dataset-10-million-transactions) (huge → sampled by customer, `MAX_ROWS=400000`).

### B) Local PC
```bash
git clone https://github.com/<your-username>/RetailPulse.git && cd RetailPulse
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py                        # dashboard  -> http://localhost:8501
uvicorn api.main:app --reload                             # API docs   -> http://localhost:8000/docs
pytest -q                                                 # tests
python scripts/retrain_pipeline.py --force                # retrain everything
```
Light install for just the dashboard: `pip install -r requirements-app.txt`.

### C) Open the dashboard from Colab
Run notebook `15_streamlit_dashboard_and_report.ipynb`, set `LAUNCH = True` – it starts Streamlit and opens a free *localtunnel* URL (password = the IP printed by the cell).

### D) Free public deployment (Zidio “Live Demo URL”)
1. Push this repo to GitHub. 2. Go to **share.streamlit.io** → *New app* → repo, branch `main`, file `app/streamlit_app.py`, requirements file `requirements-app.txt`. 3. Deploy → HTTPS URL. (Hugging Face Spaces → Streamlit SDK works the same way.)

## 3. Notebooks (`notebooks/`)
| Notebook | Day | Content |
|---|---|---|
| `00_RetailPulse_MAIN` | all | **Everything end-to-end in one notebook** |
| `01_eda_exploration` | 1 | EDA – 19 charts |
| `02_data_cleaning_feature_engineering` | 2 | cleaning waterfall, validation, RFM / lag / rolling features |
| `03_customer_segmentation` | 3 | RFM, K-Means, DBSCAN, cohorts |
| `04_timeseries_preparation` | 4 | stationarity, decomposition, ACF/PACF, split |
| `05_prophet_forecasting` | 5 | Prophet baseline |
| `06_lstm_forecasting` | 6 | PyTorch LSTM |
| `07_mlflow_experiment_tracking` | 7 | Week-1 MLflow runs |
| `08_hybrid_ensemble` | 8 | Prophet + LSTM ensemble, 30-day production forecast |
| `09_churn_prediction` | 9 | XGBoost + SHAP |
| `10_inventory_optimization` | 10 | ROP / SS / EOQ + back-test |
| `11_optuna_tuning` | 11 | Optuna tuning |
| `12_drift_detection` | 12 | PSI/KS + real drift simulation |
| `13_model_refinement` | 13 | calibration, thresholds, ablation, Airflow DAG |
| `14_mlflow_week2` | 14 | Week-2 runs, acceptance scorecard, model card |
| `15_streamlit_dashboard_and_report` | 15-28 | dashboard launch + auto-generated DOCX reports |

Every notebook is **standalone** (first 3 cells install missing packages, write `retailpulse_utils.py`, import it). `notebooks/executed_demo_output/` holds a fully executed run (on demo data) with all outputs.

## 4. Data
* **Real data columns (UCI):** `Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country` (any similar schema is auto-mapped).
* **Cleaning:** duplicates, cancellations (`C…`) → returns table, qty/price ≤ 0, non-product codes (POST, D, M…), extreme outliers, guest checkouts kept for sales but excluded from customer models.
* **Two columns added to the cleaned CSV:** `invoice_no` (numeric order key → exact distinct-order counts, clean joins) and `order_value` (invoice total on every line → AOV and order-value bands without re-grouping).
* `data/sample/online_retail_II_demo_raw.csv` – synthetic demo data in the UCI schema (so everything runs offline). `data/processed/` – pipeline outputs (clean transactions, forecasts, segments, churn scores, inventory plan, Power BI file).

## 5. Power BI
`powerbi/` contains the Power-BI-ready CSV, Power Query script, **DAX measures**, theme and a 12-page build guide. (A `.pbix` binary can only be created by Power BI Desktop – follow `powerbi/PowerBI_Build_Guide.md`, ~30 min.)

## 6. Repository structure
```
RetailPulse/
├── notebooks/        00 MAIN + 01-15 Colab notebooks
├── src/              retailpulse_utils.py (all logic) · retailpulse_report.py (DOCX generator)
├── app/              streamlit_app.py (5 pages)      ├── api/ main.py (FastAPI)
├── data/             sample/ (raw demo) · processed/ (outputs)
├── models/           churn_model.joblib · best_params.json · model_card.json
├── powerbi/          CSV helpers · DAX · Power Query · theme · guide
├── airflow/dags/     retraining DAG       ├── scripts/ retrain_pipeline.py + notebook builder
├── k8s/              deployment.yaml      ├── Dockerfile · docker-compose.yml
├── .github/workflows/ci.yml               ├── tests/ test_pipeline.py
├── docs/             charts (115 PNG) · images · metrics.json
└── reports/          FULL report + Zidio submission report (DOCX)
```

## 7. Results on the bundled demo data (`docs/metrics.json`)
| Requirement | Result |
|---|---|
| F-02 segments 6–8 | **7** (silhouette 0.34) – met |
| F-03 MAPE ≤ 12 % | **10.4 % weekly totals** (daily MAPE 18.0 %) – met on weekly basis |
| F-04 AUC ≥ 0.88 | **0.850 – not met** on demo data (Logistic Regression 0.885) · precision@top20 % 0.981 – met |
| F-05 inventory | back-test cost −79 %, lost sales −92 % (inventory level +93 %) |
| Drift | injected shock detected → retrain triggered |

 These numbers come from **synthetic  data** in an environment * PyTorch* (a gradient-boosting lag model stood in for the LSTM). Run the notebooks in Colab on the **real Kaggle data** to get your real results – the DOCX reports are regenerated automatically from them. Results are never faked: unmet targets are reported as unmet.
Stock-on-hand is **simulated** (public retail data has no inventory table).

## 8. Git workflow (as required by Zidio)
```bash
git init && git add . && git commit -m "feat: add RetailPulse end-to-end pipeline"
git branch -M main && git remote add origin https://github.com/<your-username>/RetailPulse.git && git push -u origin main
git checkout -b feat/dashboard-polish   # feature branch → pull request, even if solo
```
Semantic commits: `feat:` `fix:` `docs:` `test:` `refactor:`. No secrets are committed (`.gitignore`).

## 9. Security & privacy
Customer IDs only (no names); recommended: anonymise IDs, role-based access, JWT on API endpoints, audit logging.

---
*Made  by **Santosh Kumar** – Zidio Development Data Science & Analytics, oct 2026.*
