"""Notebook stage definitions 05-08."""
def M(t): return ("md", t.strip("\n"))
def C(t): return ("code", t.strip("\n"))

LOAD = '''
bundle = ru.get_data()
sales, returns, rep = bundle["sales"], bundle["returns"], bundle["report"]
print("DATA SOURCE :", bundle["source"], "|", f"{len(sales):,} clean rows")
daily = ru.daily_series(sales)
TARGET = "quantity"                       # demand = units sold per day
H = ru.CONFIG["FORECAST_HORIZON"]         # 30-day horizon (F-03)
tr, va, te = ru.split_series(daily, H)
trv = pd.concat([tr, va])                 # train + validation, used to score the final test window
print(f"Train {len(tr)} d | Validation {len(va)} d | Test {len(te)} d  ({te['ds'].min().date()} -> {te['ds'].max().date()})")
'''

STAGES = {}

# ---------------------------------------------------------------- 05 PROPHET
STAGES["05_prophet_forecasting"] = dict(
    title="05 - Baseline Demand Forecasting with Prophet",
    week="Week 1 - Day 5", needs_load=True, cells=[
M('''
## Introduction
**Prophet** (Meta) is an additive / multiplicative time-series model: `y(t) = trend(t) x seasonality(t) x holidays(t) + noise`.
It handles weekly and yearly seasonality, holidays and changepoints out-of-the-box, which makes it a strong **baseline** for retail demand.

In this notebook we fit Prophet on train + validation, forecast the **next 30 days (test window)**, and compare it with two simple baselines (Seasonal Naive, Holt-Winters).

**Metrics used:** MAPE (daily), MAPE on weekly totals, WAPE, sMAPE, RMSE, MAE and Bias.
> Daily MAPE is naturally high for lumpy wholesale data, so we also report weekly-total MAPE which is what a replenishment planner actually uses.
'''),
C(LOAD),
M("### Step 1 - Fit Prophet and forecast the test window"),
C('''
res = ru.prophet_forecast(trv, te["ds"], TARGET, return_model=True)
y = te[TARGET].values
print("Engine:", res["engine"])
met = ru.forecast_metrics(y, res["pred"])
display(pd.Series(met, name="Prophet").round(2).to_frame())
'''),
C('''
hist = trv.tail(90)
fig, ax = plt.subplots(figsize=(13, 5.2))
ax.plot(hist["ds"], hist[TARGET], color="#94A3B8", label="History (train+val)")
ax.plot(te["ds"], y, color=ru.PALETTE["navy"], lw=2.5, marker="o", ms=4, label="Actual (test)")
ax.plot(te["ds"], res["pred"], color=ru.PALETTE["orange"], lw=2.5, marker="s", ms=4, label=f"{res['engine']} forecast")
ax.fill_between(te["ds"], res["lower"], res["upper"], color=ru.PALETTE["orange"], alpha=0.18, label="80% interval")
ax.axvline(te["ds"].min(), color="grey", ls="--"); ax.legend(); ax.set_title("Prophet: 30-day forecast vs actual demand (units)")
ru.save_fig(fig, "05_01_prophet_forecast", "Prophet 30-day forecast vs actual", f"Daily MAPE {met['MAPE_daily_%']:.1f}%, weekly-total MAPE {met['MAPE_weekly_%']:.1f}%, WAPE {met['WAPE_%']:.1f}%.", "Forecasting - Prophet")
'''),
M("### Step 2 - Model components (trend, weekly, yearly, holidays)"),
C('''
m = res.get("model")
if m is not None:
    full = m.predict(m.make_future_dataframe(periods=H))
    fig = m.plot_components(full); fig.set_size_inches(11, 11)
    ru.save_fig(fig, "05_02_prophet_components", "Prophet components", "Trend, weekly seasonality, yearly seasonality and UK holiday effects learned by the model.", "Forecasting - Prophet")

    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(full["ds"], full["trend"], color=ru.PALETTE["red"], lw=2.5, label="Trend")
    ax.scatter(m.history["ds"], m.history["y"], s=4, color="#CBD5E1", label="Observed")
    for cp in m.changepoints: ax.axvline(cp, color=ru.PALETTE["purple"], alpha=0.15)
    ax.legend(); ax.set_title("Prophet trend and automatically detected changepoints")
    ru.save_fig(fig, "05_03_prophet_changepoints", "Prophet trend and changepoints", "Vertical lines = candidate changepoints where the growth rate may change.", "Forecasting - Prophet")
else:
    print("Prophet model object not available (fallback engine used) - component plots skipped.")
'''),
M("### Step 3 - Residual diagnostics"),
C('''
from scipy import stats
resid = y - res["pred"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].bar(te["ds"], resid, color=np.where(resid >= 0, ru.PALETTE["green"], ru.PALETTE["red"])); axes[0].set_title("Residuals over time (actual - forecast)"); axes[0].tick_params(axis="x", rotation=45)
sns.histplot(resid, bins=15, kde=True, ax=axes[1], color=ru.PALETTE["navy"]); axes[1].set_title("Residual distribution")
stats.probplot(resid, dist="norm", plot=axes[2]); axes[2].set_title("Q-Q plot")
ru.save_fig(fig, "05_04_prophet_residuals", "Prophet residual diagnostics", f"Mean residual (bias) = {resid.mean():.1f} units; residuals should be centred near zero.", "Forecasting - Prophet")
'''),
M("### Step 4 - Baselines comparison"),
C('''
nv = ru.naive_forecast(trv, te["ds"], TARGET)
hw = ru.holt_winters_forecast(trv, te["ds"], TARGET)
comp = pd.DataFrame({"Seasonal Naive": ru.forecast_metrics(y, nv), "Holt-Winters": ru.forecast_metrics(y, hw), "Prophet": met}).T
display(comp.round(2))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, c in zip(axes, ["MAPE_daily_%", "MAPE_weekly_%", "WAPE_%"]):
    comp[c].plot(kind="bar", ax=ax, color=ru.COLORS[:3]); ax.set_title(c); ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(comp[c]): ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
ru.save_fig(fig, "05_05_baseline_comparison", "Prophet vs simple baselines", "Any serious model must beat Seasonal Naive; weekly-total MAPE is the planning metric.", "Forecasting - Prophet")
'''),
C('''
dfp = pd.DataFrame({"ds": te["ds"].values, "actual": y, "pred": res["pred"]}); dfp["dow"] = pd.to_datetime(dfp["ds"]).dt.dayofweek
bd = dfp.groupby("dow")[["actual", "pred"]].mean().reindex(range(7)).fillna(0)
fig, ax = plt.subplots(figsize=(10, 4.5)); x = np.arange(7); ax.bar(x - 0.2, bd["actual"], 0.4, label="Actual", color=ru.PALETTE["navy"]); ax.bar(x + 0.2, bd["pred"], 0.4, label="Prophet", color=ru.PALETTE["orange"])
ax.set_xticks(x); ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]); ax.legend(); ax.set_title("Average actual vs forecast demand by weekday (test window)")
ru.save_fig(fig, "05_06_prophet_by_weekday", "Forecast accuracy by weekday", "Checks that the weekly pattern (including closed days) is captured.", "Forecasting - Prophet")
'''),
M("### Step 5 - Hyper-parameter sensitivity (changepoint_prior_scale)"),
C('''
rows = []
for cps in [0.01, 0.05, 0.2, 0.5]:
    r = ru.prophet_forecast(tr, va["ds"], TARGET, changepoint_prior_scale=cps)
    rows.append({"changepoint_prior_scale": cps, "val_WAPE_%": ru.wape(va[TARGET].values, r["pred"]), "val_MAPE_weekly_%": ru.block_mape(va[TARGET].values, r["pred"])})
sens = pd.DataFrame(rows); display(sens.round(2))
fig, ax = plt.subplots(figsize=(8, 4.5)); ax.plot(sens["changepoint_prior_scale"].astype(str), sens["val_WAPE_%"], marker="o", color=ru.PALETTE["purple"], lw=2.5)
ax.set_xlabel("changepoint_prior_scale"); ax.set_ylabel("Validation WAPE %"); ax.set_title("Prophet sensitivity to trend flexibility")
ru.save_fig(fig, "05_07_prophet_sensitivity", "Prophet hyper-parameter sensitivity", "Too small = under-fits trend changes, too large = over-fits noise.", "Forecasting - Prophet")
'''),
C('''
target_ok = met["MAPE_weekly_%"] <= 12
print(f"F-03 acceptance (MAPE <= 12% on weekly totals): {'PASS' if target_ok else 'NOT MET'}  -> {met['MAPE_weekly_%']:.1f}%")
print(f"Daily MAPE (strict, lumpy data): {met['MAPE_daily_%']:.1f}%")
ru.log_metrics("prophet", {**met, "engine": res["engine"], "target_weekly_mape_met": bool(target_ok)})
'''),
M('''
## Summary of Notebook 05
* Prophet baseline with UK holidays, weekly + yearly seasonality, closed-day handling.
* Components, changepoints, residual diagnostics, baseline comparison, weekday accuracy, hyper-parameter sensitivity (**7 charts**).
* Be honest in reporting: daily MAPE is high on lumpy wholesale data; weekly-total MAPE is the planning metric. **Next -> Notebook 06:** LSTM.
'''),
])

# ---------------------------------------------------------------- 06 LSTM
STAGES["06_lstm_forecasting"] = dict(
    title="06 - Deep Learning Forecasting with LSTM (PyTorch)",
    week="Week 1 - Day 6", needs_load=True, cells=[
M('''
## Introduction
An **LSTM** (Long Short-Term Memory) network learns non-linear temporal patterns from the last 28 days of demand. Our design:
* Input window: 28 days of `[log-demand, sin/cos weekday, sin/cos month, closed-day flag]`.
* **Known-future covariates** (weekday / month / closed flag of the day being predicted) are concatenated to the LSTM output.
* Training: Adam, MSE, early stopping on a chronological validation slice, **recursive 30-day forecast**.
* Implemented in plain PyTorch (stable on every Colab runtime); if PyTorch is missing the code automatically falls back to a gradient-boosting lag model so the notebook never fails.
'''),
C(LOAD),
M("### Step 1 - Network architecture"),
C('''
fig, ax = plt.subplots(figsize=(13, 3.8)); ax.set_xlim(0, 13); ax.set_ylim(0, 3.8); ax.axis("off")
ru._box(ax, 0.2, 1.4, 2.2, 1.0, "Input window\\n28 days x 6 features", "#DBEAFE", "#1D4ED8", 9, True)
ru._box(ax, 3.0, 1.4, 2.2, 1.0, "LSTM\\n(hidden = 32)", "#FEF3C7", "#B45309", 9, True)
ru._box(ax, 5.8, 1.4, 2.0, 1.0, "Last hidden\\nstate (32)", "#FCE7F3", "#BE185D", 9, True)
ru._box(ax, 5.8, 0.15, 2.0, 0.85, "Future calendar\\nfeatures (5)", "#DCFCE7", "#15803D", 9, True)
ru._box(ax, 8.4, 1.4, 2.0, 1.0, "Dense 16 + ReLU", "#EDE9FE", "#6D28D9", 9, True)
ru._box(ax, 11.0, 1.4, 1.8, 1.0, "Dense 1\\nlog-demand t+1", "#FEE2E2", "#B91C1C", 9, True)
for a, b in [(2.4, 3.0), (5.2, 5.8), (7.8, 8.4), (10.4, 11.0)]:
    ax.annotate("", xy=(b, 1.9), xytext=(a, 1.9), arrowprops=dict(arrowstyle="-|>", lw=2, color="#334155"))
ax.annotate("", xy=(8.4, 1.55), xytext=(7.8, 0.6), arrowprops=dict(arrowstyle="-|>", lw=2, color="#15803D"))
ax.text(6.5, 3.3, "Recursive multi-step forecasting: each prediction is fed back into the window", ha="center", fontsize=11, style="italic")
ax.set_title("LSTM architecture used in RetailPulse")
ru.save_fig(fig, "06_01_lstm_architecture", "LSTM architecture", "LSTM encoder + known-future calendar covariates + dense head, applied recursively for 30 days.", "Forecasting - LSTM")
'''),
M("### Step 2 - Train and forecast"),
C('''
res = ru.lstm_forecast(trv, te["ds"], TARGET, window=28, epochs=60, hidden=32)
y = te[TARGET].values
met = ru.forecast_metrics(y, res["pred"])
print("Engine:", res["engine"], "| epochs run:", len(res["history"]))
display(pd.Series(met, name="LSTM").round(2).to_frame())
'''),
C('''
h = res["history"]
if h:
    h = np.array(h)
    fig, ax = plt.subplots(figsize=(9, 4.8)); ax.plot(h[:, 0], label="Train loss", color=ru.PALETTE["navy"], lw=2.5); ax.plot(h[:, 1], label="Validation loss", color=ru.PALETTE["red"], lw=2.5)
    ax.axvline(int(np.argmin(h[:, 1])), color="grey", ls="--", label="best epoch"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE (scaled)"); ax.legend(); ax.set_title("LSTM training curves (early stopping)")
    ru.save_fig(fig, "06_02_lstm_training_curve", "LSTM training and validation loss", "Validation loss stops improving -> early stopping restores the best weights.", "Forecasting - LSTM")
else:
    print("No training curve (fallback model in use).")

hist = trv.tail(90)
fig, ax = plt.subplots(figsize=(13, 5.2)); ax.plot(hist["ds"], hist[TARGET], color="#94A3B8", label="History")
ax.plot(te["ds"], y, color=ru.PALETTE["navy"], lw=2.5, marker="o", ms=4, label="Actual"); ax.plot(te["ds"], res["pred"], color=ru.PALETTE["teal"], lw=2.5, marker="s", ms=4, label=res["engine"])
ax.axvline(te["ds"].min(), color="grey", ls="--"); ax.legend(); ax.set_title("LSTM: 30-day recursive forecast vs actual")
ru.save_fig(fig, "06_03_lstm_forecast", "LSTM 30-day forecast vs actual", f"Daily MAPE {met['MAPE_daily_%']:.1f}%, weekly MAPE {met['MAPE_weekly_%']:.1f}%, WAPE {met['WAPE_%']:.1f}%.", "Forecasting - LSTM")
'''),
C('''
from sklearn.metrics import r2_score
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
axes[0].scatter(y, res["pred"], color=ru.PALETTE["purple"], s=30); lim = [0, max(y.max(), res["pred"].max()) * 1.05]; axes[0].plot(lim, lim, "k--"); axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
axes[0].set_title(f"Actual vs predicted (R2 = {r2_score(y, res['pred']):.2f})")
resid = y - res["pred"]
axes[1].bar(te["ds"], resid, color=np.where(resid >= 0, ru.PALETTE["green"], ru.PALETTE["red"])); axes[1].set_title("Residuals over time"); axes[1].tick_params(axis="x", rotation=45)
sns.histplot(resid, bins=15, kde=True, ax=axes[2], color=ru.PALETTE["navy"]); axes[2].set_title("Residual distribution")
ru.save_fig(fig, "06_04_lstm_diagnostics", "LSTM diagnostics: scatter and residuals", "Points close to the diagonal and centred residuals indicate a well-calibrated model.", "Forecasting - LSTM")
'''),
M("### Step 3 - Sensitivity to the input window length"),
C('''
rows = []
for w in [14, 28, 56]:
    r = ru.lstm_forecast(tr, va["ds"], TARGET, window=w, epochs=30)
    rows.append({"window_days": w, "val_WAPE_%": ru.wape(va[TARGET].values, r["pred"]), "val_MAPE_weekly_%": ru.block_mape(va[TARGET].values, r["pred"])})
sens = pd.DataFrame(rows); display(sens.round(2))
fig, ax = plt.subplots(figsize=(8, 4.5)); ax.bar(sens["window_days"].astype(str), sens["val_WAPE_%"], color=ru.COLORS[:3])
for i, v in enumerate(sens["val_WAPE_%"]): ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
ax.set_xlabel("Input window (days)"); ax.set_ylabel("Validation WAPE %"); ax.set_title("LSTM window-length sensitivity")
ru.save_fig(fig, "06_05_lstm_window_sensitivity", "LSTM window-length sensitivity", "Chooses the look-back window using validation data only.", "Forecasting - LSTM")
'''),
M("### Step 4 - LSTM vs Prophet"),
C('''
pr = ru.prophet_forecast(trv, te["ds"], TARGET)
cmp_ = pd.DataFrame({"Prophet": ru.forecast_metrics(y, pr["pred"]), "LSTM": met}).T; display(cmp_.round(2))
fig, ax = plt.subplots(figsize=(13, 5)); ax.plot(te["ds"], y, color=ru.PALETTE["navy"], lw=3, label="Actual")
ax.plot(te["ds"], pr["pred"], color=ru.PALETTE["orange"], lw=2, label="Prophet"); ax.plot(te["ds"], res["pred"], color=ru.PALETTE["teal"], lw=2, label="LSTM")
ax.legend(); ax.set_title("Prophet vs LSTM on the test window")
ru.save_fig(fig, "06_06_prophet_vs_lstm", "Prophet vs LSTM", "The two models make different errors - which is exactly why an ensemble helps (Notebook 08).", "Forecasting - LSTM")
ru.log_metrics("lstm", {**met, "engine": res["engine"]})
'''),
M('''
## Summary of Notebook 06
* LSTM with calendar covariates, early stopping, recursive 30-day forecast, window-length study, comparison to Prophet (**6 charts**).
* **Next -> Notebook 07:** log baseline experiments in MLflow.
'''),
])

# ---------------------------------------------------------------- 07 MLFLOW (week 1)
STAGES["07_mlflow_experiment_tracking"] = dict(
    title="07 - MLflow Experiment Tracking (Week 1 Baselines)",
    week="Week 1 - Day 7 (checkpoint)", needs_load=True, cells=[
M('''
## Introduction
**MLflow** gives us reproducibility: every run stores parameters, metrics, tags and artifacts (charts, CSVs). In the Week-1 checkpoint we log all baseline models:
data cleaning, Seasonal Naive, Holt-Winters, Prophet, LSTM and K-Means segmentation into the experiment **RetailPulse-Week1** (local SQLite backend, no paid service or key needed).
'''),
C(LOAD),
C('''
y = te[TARGET].values
runs = []
def log(name, params, metrics, artifacts=None):
    rid = ru.mlflow_log("RetailPulse-Week1", name, params, metrics, artifacts, tags={"stage": "baseline"})
    runs.append({"run": name, **{k: v for k, v in metrics.items()}}); print(f"logged {name:<22} run_id={str(rid)[:8]}")

log("data_cleaning", {"source": bundle["source"], "dedupe": True, "outlier_pct": 99.95},
    {"raw_rows": rep["raw_rows"], "clean_rows": rep["final_rows"], "retention_pct": rep["final_rows"] / rep["raw_rows"] * 100})
nv = ru.naive_forecast(trv, te["ds"], TARGET)
log("seasonal_naive", {"weeks": 4}, ru.forecast_metrics(y, nv))
hw = ru.holt_winters_forecast(trv, te["ds"], TARGET)
log("holt_winters", {"trend": "damped additive", "seasonal_periods": 7}, ru.forecast_metrics(y, hw))
pr = ru.prophet_forecast(trv, te["ds"], TARGET)
log("prophet_baseline", {"engine": pr["engine"], "changepoint_prior_scale": 0.05, "seasonality_mode": "multiplicative"}, ru.forecast_metrics(y, pr["pred"]))
ls = ru.lstm_forecast(trv, te["ds"], TARGET, epochs=40)
log("lstm_baseline", {"engine": ls["engine"], "window": 28, "hidden": 32}, ru.forecast_metrics(y, ls["pred"]))
cs = ru.get_customer_sales(); rfm = ru.rfm_scores(ru.rfm_table(cs)); ks, X = ru.kmeans_search(rfm, range(4, 9))
best = ks.loc[ks["silhouette"].idxmax()]
log("kmeans_rfm", {"k_range": "4-8", "best_k": int(best["k"])}, {"silhouette": float(best["silhouette"]), "davies_bouldin": float(best["davies_bouldin"])})
'''),
M("### Compare the logged runs"),
C('''
tbl = pd.DataFrame(runs).set_index("run"); display(tbl.round(2))
fc = tbl.dropna(subset=["MAPE_weekly_%"])
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
fc["MAPE_weekly_%"].plot(kind="bar", ax=axes[0], color=ru.COLORS[:len(fc)]); axes[0].set_title("MLflow runs: weekly-total MAPE %"); axes[0].tick_params(axis="x", rotation=25)
fc["WAPE_%"].plot(kind="bar", ax=axes[1], color=ru.COLORS[:len(fc)]); axes[1].set_title("MLflow runs: WAPE %"); axes[1].tick_params(axis="x", rotation=25)
ru.save_fig(fig, "07_01_mlflow_forecast_runs", "MLflow: baseline forecasting runs", "All baseline runs are tracked and comparable in one place.", "MLOps - MLflow")

fig, ax = plt.subplots(figsize=(8, 4.2)); ax.bar(ks["k"].astype(str), ks["silhouette"], color=ru.PALETTE["teal"]); ax.set_xlabel("k"); ax.set_ylabel("silhouette"); ax.set_title("Silhouette by k (K-Means run)")
ru.save_fig(fig, "07_02_mlflow_kmeans", "K-Means silhouette scores logged", "Segmentation experiment logged with the same tooling as forecasting.", "MLOps - MLflow")
'''),
C('''
df_runs = ru.mlflow_runs(["RetailPulse-Week1"])
cols = [c for c in df_runs.columns if c in ("tags.mlflow.runName", "run", "status", "params.engine") or c.startswith("metrics.MAPE_weekly")]
print("Runs stored in MLflow:", len(df_runs))
display(df_runs[cols].head(10) if cols else df_runs.head(10))
print("\\nTo open the MLflow UI on your own PC:  mlflow ui --backend-store-uri sqlite:///retailpulse_work/mlflow.db")
ru.log_metrics("mlflow", {"week1_runs": int(len(df_runs))})
'''),
M('''
## Summary of Notebook 07
* 6 baseline runs logged (cleaning, Seasonal Naive, Holt-Winters, Prophet, LSTM, K-Means) with params, metrics and tags.
* **Week-1 checkpoint reached:** EDA, cleaned dataset and baseline models logged in MLflow.
'''),
])

# ---------------------------------------------------------------- 08 HYBRID
STAGES["08_hybrid_ensemble"] = dict(
    title="08 - Hybrid Forecasting Ensemble (Prophet + LSTM)",
    week="Week 2 - Day 8", needs_load=True, cells=[
M('''
## Introduction
Prophet captures trend + calendar effects; LSTM captures non-linear short-term dynamics. A **weighted ensemble** combines both:
`Hybrid(t) = w x Prophet(t) + (1 - w) x LSTM(t)`.
The weight `w` is chosen on the **validation** window (minimising WAPE) and then both models are re-fitted on train + validation to score the **test** window - no leakage.
Finally the hybrid is refit on all data to produce the **30-day-ahead production forecast** used by inventory optimisation and the dashboard.
'''),
C(LOAD),
C('''
fb = ru.forecast_bundle(daily, TARGET, H)
print("Engines:", fb["engines"], "| Prophet weight w =", round(fb["weight_prophet"], 2))
display(fb["metrics"].round(2))
y = fb["test"][TARGET].values
'''),
M("### Step 1 - Blend weight search"),
C('''
bc = fb["blend_curve"]
fig, ax = plt.subplots(figsize=(9, 4.8)); ax.plot(bc["w_prophet"], bc["WAPE_%"], color=ru.PALETTE["purple"], lw=2.5, marker="o")
ax.axvline(fb["weight_prophet"], color="red", ls="--", label=f"best w = {fb['weight_prophet']:.2f}"); ax.legend()
ax.set_xlabel("Weight on Prophet (1 - w on LSTM)"); ax.set_ylabel("Validation WAPE %"); ax.set_title("Ensemble weight optimisation (validation window)")
ru.save_fig(fig, "08_01_blend_weight", "Ensemble weight search", f"Best Prophet weight = {fb['weight_prophet']:.2f}.", "Forecasting - Hybrid")
'''),
M("### Step 2 - Test-window forecasts of all models"),
C('''
te = fb["test"]; P = fb["test_preds"]
fig, ax = plt.subplots(figsize=(13, 5.5)); ax.plot(te["ds"], y, color=ru.PALETTE["navy"], lw=3.2, label="Actual")
for (n, v), c in zip(P.items(), [ "#94A3B8", "#CBD5E1", ru.PALETTE["orange"], ru.PALETTE["teal"], ru.PALETTE["red"]]):
    ax.plot(te["ds"], v, lw=2.8 if n == "Hybrid" else 1.4, ls="-" if n == "Hybrid" else "--", color=c, label=n)
ax.legend(ncol=3); ax.set_title("All models vs actual - 30-day test window")
ru.save_fig(fig, "08_02_all_models_test", "All models vs actual (test window)", "The hybrid line combines Prophet and LSTM.", "Forecasting - Hybrid")

met = fb["metrics"]
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, c in zip(axes, ["MAPE_daily_%", "MAPE_weekly_%", "WAPE_%"]):
    met[c].plot(kind="bar", ax=ax, color=ru.COLORS[:len(met)]); ax.set_title(c); ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(met[c]): ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
ru.save_fig(fig, "08_03_model_comparison", "Model comparison on test window", "Lower is better. The best model by weekly MAPE is highlighted in the table above.", "Forecasting - Hybrid")
'''),
C('''
errs = pd.DataFrame({n: y - v for n, v in P.items()})
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
sns.boxplot(data=errs, ax=axes[0], palette=ru.COLORS); axes[0].axhline(0, color="black", lw=0.8); axes[0].set_title("Residual distribution per model"); axes[0].tick_params(axis="x", rotation=25)
errs.abs().cumsum().set_index(te["ds"].values).plot(ax=axes[1], color=ru.COLORS[:5]); axes[1].set_title("Cumulative absolute error")
ru.save_fig(fig, "08_04_residuals_cumulative", "Residuals and cumulative error per model", "Hybrid should show a smaller, tighter error distribution than its parts.", "Forecasting - Hybrid")

k = (len(y) // 7) * 7
wk = pd.DataFrame({"week": [f"Week {i + 1}" for i in range(k // 7)], "Actual": y[:k].reshape(-1, 7).sum(1), "Hybrid": P["Hybrid"][:k].reshape(-1, 7).sum(1)})
fig, ax = plt.subplots(figsize=(9, 4.8)); x = np.arange(len(wk)); ax.bar(x - 0.2, wk["Actual"], 0.4, label="Actual", color=ru.PALETTE["navy"]); ax.bar(x + 0.2, wk["Hybrid"], 0.4, label="Hybrid", color=ru.PALETTE["teal"])
ax.set_xticks(x); ax.set_xticklabels(wk["week"]); ax.legend(); ax.set_title("Weekly demand totals: actual vs hybrid forecast")
ru.save_fig(fig, "08_05_weekly_totals", "Weekly totals: actual vs hybrid forecast", f"Weekly-total MAPE of hybrid = {met.loc['Hybrid (Prophet+LSTM)', 'MAPE_weekly_%']:.1f}%.", "Forecasting - Hybrid")
'''),
M("### Step 3 - Production forecast: next 30 days beyond the last date"),
C('''
fu = fb["future"]; hist = daily.tail(75)
fig, ax = plt.subplots(figsize=(13, 5.2)); ax.plot(hist["ds"], hist[TARGET], color=ru.PALETTE["navy"], lw=1.8, label="History")
ax.plot(fu["ds"], fu["hybrid"], color=ru.PALETTE["red"], lw=2.8, marker="o", ms=4, label="Hybrid forecast"); ax.fill_between(fu["ds"], fu["lower"], fu["upper"], color=ru.PALETTE["red"], alpha=0.15, label="80% interval")
ax.axvline(fu["ds"].min(), color="grey", ls="--"); ax.legend(); ax.set_title(f"30-day-ahead demand forecast ({fu['ds'].min().date()} -> {fu['ds'].max().date()})")
ru.save_fig(fig, "08_06_future_forecast", "30-day-ahead production forecast", f"Expected total demand next 30 days: {fu['hybrid'].sum():,.0f} units.", "Forecasting - Hybrid")
fu.to_csv(ru.DATA_PROC / "forecast_future.csv", index=False)
pd.DataFrame({"ds": te["ds"].values, "actual": y, **{k_: v for k_, v in P.items()}}).to_csv(ru.DATA_PROC / "forecast_test.csv", index=False)
hy = met.loc["Hybrid (Prophet+LSTM)"]
ok = hy["MAPE_weekly_%"] <= 12
print(f"F-03 target MAPE <= 12% (weekly totals): {'PASS' if ok else 'NOT MET'} -> {hy['MAPE_weekly_%']:.1f}% | daily MAPE {hy['MAPE_daily_%']:.1f}% | WAPE {hy['WAPE_%']:.1f}%")
ru.log_metrics("hybrid", {**hy.to_dict(), "weight_prophet": fb["weight_prophet"], "weekly_target_met": bool(ok), "future_30d_units": float(fu["hybrid"].sum())})
'''),
M('''
## Summary of Notebook 08
* Validation-tuned Prophet + LSTM ensemble, leakage-free test scoring, residual analysis and the **30-day production forecast**.
* Saved `forecast_future.csv` and `forecast_test.csv` (inputs for inventory + dashboard). **6 charts.**
'''),
])
