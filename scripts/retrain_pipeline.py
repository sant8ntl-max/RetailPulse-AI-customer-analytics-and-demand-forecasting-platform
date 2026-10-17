"""
Retraining pipeline (used by Airflow DAG, GitHub Actions and manual runs).
Usage:  python scripts/retrain_pipeline.py [--force]
Author: Santosh Kumar
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joblib  # noqa: E402
import retailpulse_utils as ru  # noqa: E402


def drift_needs_retrain(psi_limit: float = 0.2) -> bool:
    sales = ru.get_sales()
    cut = sales["date"].min() + (sales["date"].max() - sales["date"].min()) * 0.7
    ref, cur = sales[sales["date"] <= cut], sales[sales["date"] > cut]
    d = ru.drift_table(ref, cur, ["quantity", "unit_price", "revenue", "hour"])
    dec = ru.retrain_decision(d, current_mape=0.0, psi_limit=psi_limit)
    print("Drift table:\n", d.round(3))
    print("Decision:", dec)
    return dec["retrain"]


def run_retrain():
    ru.setup()
    bundle = ru.get_data(force=True)
    sales, returns = bundle["sales"], bundle["returns"]
    daily = ru.daily_series(sales)
    fb = ru.forecast_bundle(daily, "quantity", force=True, lstm_epochs=40)
    cs = ru.get_customer_sales()
    X, y, _ = ru.build_churn_dataset(cs, returns)
    res = ru.train_churn_models(X, y)
    model = res["models"][res["xgb_name"]]
    ru.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ru.MODELS / "churn_model.joblib")
    fb["future"].to_csv(ru.DATA_PROC / "forecast_future.csv", index=False)
    m = res["metrics"].loc[res["xgb_name"]].to_dict()
    hy = fb["metrics"].loc["Hybrid (Prophet+LSTM)"].to_dict()
    ru.log_metrics("retrain", {"churn": m, "forecast": hy})
    ru.mlflow_log("RetailPulse-Retraining", "scheduled_retrain", {"engine": res["xgb_name"]}, {**{f"churn_{k}": v for k, v in m.items()}, **{f"fc_{k}": v for k, v in hy.items()}})
    print("Retraining finished. Churn AUC:", round(m["AUC-ROC"], 3), "| Hybrid weekly MAPE:", round(hy["MAPE_weekly_%"], 1))


if __name__ == "__main__":
    if "--force" in sys.argv or drift_needs_retrain():
        run_retrain()
    else:
        print("No retraining required.")
