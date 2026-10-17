"""
RetailPulse - Streamlit Dashboard (5 pages)
Run:  streamlit run app/streamlit_app.py
Author: Santosh Kumar | Zidio Development - Data Science & Analytics (March 2026)

Data order: (1) file uploaded in sidebar  (2) env RETAILPULSE_DATA folder  (3) ./retailpulse_work/data/processed
            (4) ./data/processed (shipped demo)  (5) built-in demo data.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "src", Path.cwd(), Path.cwd() / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import retailpulse_utils as ru  # noqa: E402

st.set_page_config(page_title="RetailPulse | Santosh Kumar", page_icon="📊", layout="wide")

CANDIDATE_DIRS = [Path(os.environ["RETAILPULSE_DATA"])] if os.environ.get("RETAILPULSE_DATA") else []
CANDIDATE_DIRS += [Path.cwd() / "retailpulse_work" / "data" / "processed", HERE.parent / "data" / "processed", Path.cwd() / "data" / "processed"]


def find_dir():
    for d in CANDIDATE_DIRS:
        if (d / "transactions_clean.csv.gz").exists():
            return d
    return None


@st.cache_data(show_spinner="Loading transactions ...")
def load_transactions(path, uploaded_bytes=None, uploaded_name=None):
    if uploaded_bytes is not None:
        import io
        buf = io.BytesIO(uploaded_bytes)
        raw = pd.read_csv(buf) if uploaded_name.lower().endswith(".csv") else pd.read_excel(buf)
        sales, _, _ = ru.clean(ru.standardize(raw), verbose=False)
        return sales
    if path:
        df = pd.read_csv(Path(path) / "transactions_clean.csv.gz", parse_dates=["invoice_date"])
        d = df["invoice_date"]
        df["date"] = d.dt.normalize()
        df["year_month"] = d.dt.to_period("M").astype(str)
        df["month_name"] = d.dt.strftime("%b")
        df["dow"], df["hour"] = d.dt.dayofweek, d.dt.hour
        df["month"], df["year"], df["day"] = d.dt.month, d.dt.year, d.dt.day
        df["is_weekend"] = (df["dow"] >= 5).astype(int)
        df["week"] = d.dt.isocalendar().week.astype(int)
        df["quarter"] = d.dt.quarter
        return df
    sales, _, _ = ru.clean(ru.standardize(ru.generate_demo_data()), verbose=False)
    return sales


@st.cache_data(show_spinner="Segmenting customers ...")
def customer_tables(_sales, key):
    cs = _sales[_sales["customer_id"].notna()]
    rfm = ru.rfm_scores(ru.rfm_table(cs))
    ks, X = ru.kmeans_search(rfm, range(6, 9))
    K = int(ks.loc[ks["silhouette"].idxmax(), "k"])
    seg, _, _ = ru.fit_kmeans(rfm, X, K)
    return rfm, seg.reset_index(), K


@st.cache_data(show_spinner="Scoring churn risk ...")
def churn_table(_sales, key):
    cs = _sales[_sales["customer_id"].notna()]
    X, y, _ = ru.build_churn_dataset(cs, None)
    res = ru.train_churn_models(X, y)
    model = res["models"][res["xgb_name"]]
    return ru.score_all_customers(model, cs, None), res["metrics"].loc[res["xgb_name"]]


@st.cache_data(show_spinner="Forecasting ...")
def live_forecast(_sales, key, horizon):
    daily = ru.daily_series(_sales)
    fut_dates = pd.date_range(daily["ds"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    p = ru.holt_winters_forecast(daily, fut_dates, "quantity")
    sd = float(daily["quantity"].tail(60).std())
    return pd.DataFrame({"ds": fut_dates, "hybrid": p, "lower": np.clip(p - 1.28 * sd, 0, None), "upper": p + 1.28 * sd})


# ------------------------------------------------------------------ sidebar
st.sidebar.title("📊 RetailPulse")
st.sidebar.caption("AI-Powered Customer Analytics & Demand Forecasting\n\nMade by **Santosh Kumar**")
page = st.sidebar.radio("Dashboard page", ["1 · Sales", "2 · Customers", "3 · Forecast", "4 · Inventory", "5 · Monitoring", "ℹ️ About"])
up = st.sidebar.file_uploader("Optional: upload your retail CSV / XLSX", type=["csv", "xlsx", "xls"])
data_dir = find_dir()
try:
    if up is not None:
        sales = load_transactions(None, up.getvalue(), up.name)
    else:
        sales = load_transactions(str(data_dir) if data_dir else None)
except Exception as e:  # noqa
    st.sidebar.error(f"Upload could not be read ({e}). Showing demo data.")
    up = None
    sales = load_transactions(str(data_dir) if data_dir else None)
key = f"{len(sales)}_{sales['revenue'].sum():.0f}"
st.sidebar.success(f"{len(sales):,} rows | {sales['date'].min().date()} → {sales['date'].max().date()}")

st.sidebar.markdown("### Filters")
d0, d1 = sales["date"].min().date(), sales["date"].max().date()
date_rng = st.sidebar.date_input("Date range", (d0, d1), min_value=d0, max_value=d1)
countries = st.sidebar.multiselect("Country", sorted(sales["country"].unique()), default=[])
cats = st.sidebar.multiselect("Category", sorted(sales["category"].unique()), default=[])
f = sales.copy()
if isinstance(date_rng, tuple) and len(date_rng) == 2:
    f = f[(f["date"].dt.date >= date_rng[0]) & (f["date"].dt.date <= date_rng[1])]
if countries:
    f = f[f["country"].isin(countries)]
if cats:
    f = f[f["category"].isin(cats)]

# ------------------------------------------------------------------ page 1
if page.startswith("1"):
    st.title("🛒 Sales Dashboard")
    if f.empty:
        st.warning("No rows for the selected filters.")
        st.stop()
    c = st.columns(5)
    orders = f["invoice"].nunique()
    c[0].metric("Revenue", f"{f['revenue'].sum():,.0f}")
    c[1].metric("Orders", f"{orders:,}")
    c[2].metric("Avg order value", f"{f['revenue'].sum() / max(orders, 1):,.1f}")
    c[3].metric("Customers", f"{f['customer_id'].nunique():,}")
    c[4].metric("Units sold", f"{int(f['quantity'].sum()):,}")
    m = f.groupby("year_month").agg(revenue=("revenue", "sum"), orders=("invoice", "nunique")).reset_index()
    fig = go.Figure()
    fig.add_bar(x=m["year_month"], y=m["revenue"], name="Revenue", marker_color="#0EA5E9")
    fig.add_scatter(x=m["year_month"], y=m["orders"], name="Orders", yaxis="y2", line=dict(color="#F59E0B", width=3))
    fig.update_layout(title="Monthly revenue and orders", yaxis2=dict(overlaying="y", side="right"), height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    top = f.groupby("description")["revenue"].sum().nlargest(10).reset_index()
    a.plotly_chart(px.bar(top, x="revenue", y="description", orientation="h", title="Top 10 products", color_discrete_sequence=["#14B8A6"]).update_layout(yaxis=dict(autorange="reversed")), width="stretch")
    cat = f.groupby("category")["revenue"].sum().reset_index()
    b.plotly_chart(px.pie(cat, names="category", values="revenue", hole=0.5, title="Revenue by category"), width="stretch")
    a, b = st.columns(2)
    ct = f.groupby("country")["revenue"].sum().nlargest(10).reset_index()
    a.plotly_chart(px.bar(ct, x="country", y="revenue", title="Top countries", color_discrete_sequence=["#6366F1"]), width="stretch")
    hm = pd.crosstab(f["dow"], f["hour"]).reindex(range(7), fill_value=0)
    hm.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    b.plotly_chart(px.imshow(hm, aspect="auto", color_continuous_scale="YlGnBu", title="Order lines: weekday × hour"), width="stretch")
    st.download_button("⬇️ Download filtered data (CSV)", f.to_csv(index=False).encode(), "retailpulse_filtered_sales.csv", "text/csv")

# ------------------------------------------------------------------ page 2
elif page.startswith("2"):
    st.title("👥 Customer Dashboard — Segmentation & Churn")
    if sales["customer_id"].nunique() < 50:
        st.warning("Too few identified customers in this dataset.")
        st.stop()
    rfm, seg, K = customer_tables(sales, key)
    scores, met = churn_table(sales, key)
    c = st.columns(4)
    c[0].metric("Customers", f"{len(rfm):,}")
    c[1].metric("Segments (K-Means)", K)
    c[2].metric("High-risk customers", f"{int((scores['churn_probability'] > 0.7).sum()):,}")
    c[3].metric("Churn model AUC", f"{met['AUC-ROC']:.3f}")
    t1, t2, t3 = st.tabs(["RFM segments", "Churn risk", "Customer lookup"])
    with t1:
        prof = seg.groupby("segment").agg(customers=("monetary", "size"), recency=("recency", "mean"), frequency=("frequency", "mean"), monetary=("monetary", "mean"), revenue=("monetary", "sum")).reset_index()
        a, b = st.columns(2)
        a.plotly_chart(px.bar(prof.sort_values("customers"), x="customers", y="segment", orientation="h", title="Customers per segment", color_discrete_sequence=["#14B8A6"]), width="stretch")
        b.plotly_chart(px.pie(prof, names="segment", values="revenue", hole=0.45, title="Revenue share by segment"), width="stretch")
        st.plotly_chart(px.scatter(seg, x="recency", y="monetary", size="frequency", color="segment", log_y=True, title="Recency vs monetary (bubble = frequency)", opacity=0.7), width="stretch")
        st.dataframe(prof.round(1), width="stretch")
        st.download_button("⬇️ Download segments (CSV)", seg.to_csv(index=False).encode(), "retailpulse_segments.csv", "text/csv")
    with t2:
        a, b = st.columns(2)
        a.plotly_chart(px.histogram(scores, x="churn_probability", nbins=40, color_discrete_sequence=["#EF4444"], title="Churn probability distribution"), width="stretch")
        rb = scores["risk_band"].astype(str).value_counts().reset_index()
        rb.columns = ["risk_band", "customers"]
        b.plotly_chart(px.pie(rb, names="risk_band", values="customers", hole=0.5, title="Risk bands", color="risk_band", color_discrete_map={"Low": "#22C55E", "Medium": "#F59E0B", "High": "#EF4444"}), width="stretch")
        th = st.slider("Show customers with churn probability above", 0.0, 1.0, 0.7, 0.05)
        top = scores[scores["churn_probability"] >= th].sort_values("churn_probability", ascending=False)
        st.write(f"**{len(top):,} customers** above {th:.2f}")
        st.dataframe(top.head(200).round(3), width="stretch")
        st.download_button("⬇️ Download at-risk list (CSV)", top.to_csv(index=False).encode(), "retailpulse_at_risk_customers.csv", "text/csv")
    with t3:
        cid = st.selectbox("Customer ID", scores["customer_id"].astype(int).tolist()[:2000])
        row = scores[scores["customer_id"] == cid].iloc[0]
        s_ = seg[seg["customer_id"] == cid]
        st.json({"customer_id": int(cid), "churn_probability": round(float(row["churn_probability"]), 3), "risk_band": str(row["risk_band"]),
                 "recency_days": int(row["recency"]), "orders": int(row["frequency"]), "spend": round(float(row["monetary"]), 2),
                 "segment": None if s_.empty else str(s_["segment"].iloc[0])})

# ------------------------------------------------------------------ page 3
elif page.startswith("3"):
    st.title("🔮 Demand Forecast Dashboard")
    fut_file = (data_dir / "forecast_future.csv") if data_dir else None
    test_file = (data_dir / "forecast_test.csv") if data_dir else None
    daily = ru.daily_series(sales)
    horizon = st.slider("Forecast horizon (days)", 7, 60, 30)
    if fut_file is not None and fut_file.exists() and up is None:
        fut = pd.read_csv(fut_file, parse_dates=["ds"]).head(horizon)
        st.success("Showing the **Prophet + LSTM hybrid** forecast produced by the notebooks.")
    else:
        fut = live_forecast(sales, key, horizon)
        st.info("Showing a fast **Holt-Winters** forecast computed live. Run the notebooks for the Prophet + LSTM hybrid.")
    st.subheader("What-if analysis")
    a, b, c_ = st.columns(3)
    promo = a.slider("Promotion uplift %", -30, 80, 0)
    price = b.slider("Price change %", -20, 30, 0)
    elasticity = c_.slider("Price elasticity", -3.0, 0.0, -1.2, 0.1)
    mult = (1 + promo / 100) * (1 + price / 100) ** elasticity
    adj = fut.copy()
    for col in ("hybrid", "lower", "upper"):
        adj[col] = adj[col] * mult
    hist = daily.tail(90)
    fig = go.Figure()
    fig.add_scatter(x=hist["ds"], y=hist["quantity"], name="History", line=dict(color="#1F3A5F"))
    fig.add_scatter(x=fut["ds"], y=fut["upper"], line=dict(width=0), showlegend=False)
    fig.add_scatter(x=fut["ds"], y=fut["lower"], fill="tonexty", fillcolor="rgba(239,68,68,0.15)", line=dict(width=0), name="80% interval")
    fig.add_scatter(x=fut["ds"], y=fut["hybrid"], name="Baseline forecast", line=dict(color="#EF4444", width=3))
    if abs(mult - 1) > 1e-6:
        fig.add_scatter(x=adj["ds"], y=adj["hybrid"], name="What-if scenario", line=dict(color="#14B8A6", width=3, dash="dash"))
    fig.update_layout(title="Demand forecast (units per day)", height=470, legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    c = st.columns(3)
    c[0].metric("Baseline demand", f"{fut['hybrid'].sum():,.0f} units")
    c[1].metric("Scenario demand", f"{adj['hybrid'].sum():,.0f} units", f"{(mult - 1) * 100:+.1f}%")
    c[2].metric("Demand multiplier", f"{mult:.3f}")
    if test_file is not None and test_file.exists() and up is None:
        tt = pd.read_csv(test_file, parse_dates=["ds"])
        st.subheader("Back-test on the last 30 days")
        cols = [x for x in tt.columns if x != "ds"]
        st.plotly_chart(px.line(tt, x="ds", y=cols, title="Actual vs model forecasts"), width="stretch")
        rows = [{"model": x, "WAPE %": ru.wape(tt["actual"], tt[x]), "MAPE weekly %": ru.block_mape(tt["actual"], tt[x]), "MAPE daily %": ru.mape(tt["actual"], tt[x])} for x in cols if x != "actual"]
        st.dataframe(pd.DataFrame(rows).round(2), width="stretch")
    st.download_button("⬇️ Download forecast (CSV)", adj.to_csv(index=False).encode(), "retailpulse_forecast.csv", "text/csv")

# ------------------------------------------------------------------ page 4
elif page.startswith("4"):
    st.title("📦 Inventory Optimization")
    st.caption("Current stock is **simulated** because public retail data has no stock-on-hand column. Connect your real stock table in production.")
    a, b, c_, d = st.columns(4)
    L = a.slider("Lead time (days)", 1, 21, ru.CONFIG["LEAD_TIME_DAYS"])
    R = b.slider("Review period (days)", 1, 14, ru.CONFIG["REVIEW_PERIOD_DAYS"])
    SL = c_.slider("Service level %", 80, 99, 95)
    growth = d.slider("Demand growth vs history", 0.5, 2.0, 1.0, 0.05)
    plan = ru.inventory_plan(sales, growth_ratio=growth, lead_time=L, review=R, service_level=SL / 100, top_n=40)
    c = st.columns(4)
    c[0].metric("SKUs analysed", len(plan))
    c[1].metric("Reorder now", int((plan["status"] == "Reorder Now").sum()))
    c[2].metric("Overstocked", int((plan["status"] == "Overstock").sum()))
    c[3].metric("Recommended order value", f"{plan['order_value'].sum():,.0f}")
    cmap = {"Reorder Now": "#EF4444", "Healthy": "#22C55E", "Overstock": "#F59E0B"}
    a, b = st.columns(2)
    a.plotly_chart(px.pie(plan, names="status", hole=0.5, title="Stock status", color="status", color_discrete_map=cmap), width="stretch")
    b.plotly_chart(px.scatter(plan, x="reorder_point", y="current_stock_simulated", color="status", hover_name="description", title="Stock vs reorder point", color_discrete_map=cmap), width="stretch")
    st.plotly_chart(px.bar(plan.nlargest(15, "order_value"), x="order_value", y="description", orientation="h", title="Top recommended purchase orders", color_discrete_sequence=["#14B8A6"]).update_layout(yaxis=dict(autorange="reversed")), width="stretch")
    st.dataframe(plan.round(1), width="stretch")
    st.download_button("⬇️ Download reorder plan (CSV)", plan.to_csv(index=False).encode(), "retailpulse_inventory_plan.csv", "text/csv")

# ------------------------------------------------------------------ page 5
elif page.startswith("5"):
    st.title("🛰️ Monitoring — Data Drift")
    cut = sales["date"].min() + (sales["date"].max() - sales["date"].min()) * 0.7
    ref, cur = sales[sales["date"] <= cut], sales[sales["date"] > cut]
    st.write(f"Reference: **{ref['date'].min().date()} → {ref['date'].max().date()}** | Current: **{cur['date'].min().date()} → {cur['date'].max().date()}**")
    sim = st.checkbox("Simulate drift (price +35%, quantity −40%)")
    cur2 = cur.copy()
    if sim:
        cur2["unit_price"] = cur2["unit_price"] * 1.35
        cur2["quantity"] = np.maximum(1, (cur2["quantity"] * 0.6).round())
        cur2["revenue"] = cur2["quantity"] * cur2["unit_price"]
    d = ru.drift_table(ref, cur2, ["quantity", "unit_price", "revenue", "hour"])
    st.plotly_chart(px.bar(d, x="feature", y="PSI", color="status", title="Population Stability Index", color_discrete_map={"OK": "#22C55E", "WARNING": "#F59E0B", "DRIFT": "#EF4444"}), width="stretch")
    st.dataframe(d.round(4), width="stretch")
    dec = ru.retrain_decision(d, 0.0)
    if dec["retrain"]:
        st.error("RETRAIN RECOMMENDED: " + "; ".join(dec["reasons"]))
    else:
        st.success("Model is stable - no retraining required.")

# ------------------------------------------------------------------ about
else:
    st.title("ℹ️ About RetailPulse")
    st.markdown("""
**RetailPulse – AI-Powered Customer Analytics & Demand Forecasting Platform**
*Made by **Santosh Kumar** · Zidio Development – Data Science & Analytics · March 2026*

**Pages:** 1 Sales · 2 Customers (RFM + churn) · 3 Forecast (Prophet + LSTM, what-if) · 4 Inventory (ROP / safety stock / EOQ) · 5 Monitoring (PSI drift).

**How to use:** pick filters in the sidebar, or upload your own retail file with columns like *Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country*.

**Stack:** Python · Pandas · Scikit-learn · XGBoost · Prophet · PyTorch LSTM · MLflow · Streamlit · FastAPI · Docker · Kubernetes.
""")
