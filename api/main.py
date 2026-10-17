"""
RetailPulse - FastAPI serving layer.
Run:  uvicorn api.main:app --reload --port 8000     (docs at http://localhost:8000/docs)
Author: Santosh Kumar
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import retailpulse_utils as ru  # noqa: E402

app = FastAPI(title="RetailPulse API", version="2.0", description="Demand forecast, churn and RFM segment endpoints - by Santosh Kumar")
PROC = ROOT / "data" / "processed"


class RFMInput(BaseModel):
    recency_days: float = Field(..., ge=0, description="Days since last purchase")
    frequency: float = Field(..., ge=0, description="Number of orders")
    monetary: float = Field(..., ge=0, description="Total spend")


class ChurnInput(RFMInput):
    avg_gap_days: float = Field(30.0, ge=0, description="Average days between orders")


@app.get("/health")
def health():
    return {"status": "ok", "service": "RetailPulse", "author": "Santosh Kumar"}


@app.get("/forecast")
def forecast(horizon: int = Query(30, ge=1, le=60)):
    f = PROC / "forecast_future.csv"
    if not f.exists():
        raise HTTPException(404, "forecast_future.csv not found - run notebook 08 first")
    df = pd.read_csv(f).head(horizon)
    return {"horizon": len(df), "total_units": float(df["hybrid"].sum()), "forecast": df[["ds", "hybrid", "lower", "upper"]].round(2).to_dict(orient="records")}


@app.post("/segment")
def segment(x: RFMInput):
    """Rule-based RFM segment from absolute values (uses quintile thresholds of the shipped customer table when available)."""
    f = PROC / "rfm_segments.csv"
    if f.exists():
        t = pd.read_csv(f)
        r = int(5 - np.searchsorted(np.quantile(t["recency"], [0.2, 0.4, 0.6, 0.8]), x.recency_days, side="right")) if True else 3
        fq = int(1 + np.searchsorted(np.quantile(t["frequency"], [0.2, 0.4, 0.6, 0.8]), x.frequency, side="right"))
        mq = int(1 + np.searchsorted(np.quantile(t["monetary"], [0.2, 0.4, 0.6, 0.8]), x.monetary, side="right"))
    else:
        r, fq, mq = (5 if x.recency_days < 30 else 3 if x.recency_days < 120 else 1), min(5, 1 + int(x.frequency // 3)), min(5, 1 + int(x.monetary // 800))
    fm = (fq + mq) / 2
    if r >= 4 and fm >= 4.5: name = "Champions"
    elif r == 1 and fm >= 4: name = "Can't Lose Them"
    elif r <= 2 and fm >= 3: name = "At Risk"
    elif r >= 3 and fm >= 3.5: name = "Loyal Customers"
    elif r >= 4 and fm >= 2.5: name = "Potential Loyalists"
    elif r >= 4: name = "New Customers"
    elif r == 3 and fm >= 2.5: name = "Need Attention"
    elif r == 3: name = "Promising"
    elif r == 2: name = "About To Sleep"
    elif r == 1 and fm <= 2: name = "Lost"
    else: name = "Hibernating"
    return {"R": r, "F": fq, "M": mq, "segment": name}


@app.post("/churn/predict")
def churn(x: ChurnInput):
    """Uses the trained model in models/churn_model.joblib if present; otherwise a transparent heuristic."""
    mp = ROOT / "models" / "churn_model.joblib"
    if mp.exists():
        try:
            import joblib
            model = joblib.load(mp)
            names = list(getattr(model, "feature_names_in_", []))
            if names:
                row = {n: 0.0 for n in names}
                row.update({"recency": x.recency_days, "frequency": x.frequency, "monetary": x.monetary, "avg_gap_days": x.avg_gap_days,
                            "avg_order_value": x.monetary / max(x.frequency, 1), "recency_gap_ratio": x.recency_days / max(x.avg_gap_days, 1)})
                p = float(model.predict_proba(pd.DataFrame([row])[names])[0, 1])
                return {"churn_probability": round(p, 4), "risk_band": "High" if p > 0.7 else "Medium" if p > 0.4 else "Low", "model": "XGBoost (trained)"}
        except Exception:
            pass
    ratio = x.recency_days / max(x.avg_gap_days, 1)
    p = float(1 / (1 + np.exp(-(1.2 * np.log1p(ratio) - 0.25 * np.log1p(x.frequency) - 0.6))))
    return {"churn_probability": round(p, 4), "risk_band": "High" if p > 0.7 else "Medium" if p > 0.4 else "Low", "model": "heuristic (no trained model file found)"}
