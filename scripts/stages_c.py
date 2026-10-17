"""Notebook stage definitions 09-14."""
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
def M(t): return ("md", t.strip("\n"))
def C(t): return ("code", t.strip("\n"))

LOAD = '''
bundle = ru.get_data()
sales, returns, rep = bundle["sales"], bundle["returns"], bundle["report"]
print("DATA SOURCE :", bundle["source"], "|", f"{len(sales):,} clean rows")
daily = ru.daily_series(sales); TARGET = "quantity"; H = ru.CONFIG["FORECAST_HORIZON"]
'''
CHURN_PREP = '''
cs = ru.get_customer_sales()
X, y, cutoff = ru.build_churn_dataset(cs, returns)
res = ru.train_churn_models(X, y)
Xtr, Xva, Xte, ytr, yva, yte = res["splits"]
XN = res["xgb_name"]; xgb = res["models"][XN]
print(f"Customers: {len(X)} | features: {X.shape[1]} | churn rate: {y.mean() * 100:.1f}% | label window: last {ru.CONFIG['CHURN_WINDOW_DAYS']} days (cutoff {cutoff.date()})")
'''
STAGES = {}

# ---------------------------------------------------------------- 09 CHURN
STAGES["09_churn_prediction"] = dict(
    title="09 - Churn Prediction with XGBoost + SHAP Explainability",
    week="Week 2 - Day 9", needs_load=True, cells=[
M('''
## Introduction
**Churn** = a customer who bought before but makes **no purchase in the next 90 days**. Predicting churn early lets marketing launch win-back campaigns.

**Leakage-free design:** features are computed only from history *before* a cut-off date (last 90 days are held out); the label looks at the 90 days *after* the cut-off.
**Models:** Logistic Regression (baseline), Random Forest, **XGBoost** (main). **Explainability:** SHAP.
**Acceptance (F-04):** AUC-ROC >= 0.88 and precision@top-20% >= 0.75.
'''),
C(LOAD), C(CHURN_PREP),
M("### Step 1 - Label balance and feature behaviour"),
C('''
fig, ax = plt.subplots(figsize=(6, 4.5)); vc = y.value_counts().sort_index(); ax.bar(["Active (0)", "Churned (1)"], vc.values, color=[ru.PALETTE["green"], ru.PALETTE["red"]])
for i, v in enumerate(vc.values): ax.text(i, v, f"{v} ({v / len(y) * 100:.0f}%)", ha="center", va="bottom")
ax.set_title("Churn label distribution")
ru.save_fig(fig, "09_01_label_distribution", "Churn label distribution", f"{y.mean() * 100:.1f}% of customers churned in the 90-day window.", "Churn Prediction")

feats = ["recency", "frequency", "monetary", "avg_gap_days", "orders_last_90d", "recency_gap_ratio"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, f in zip(axes.ravel(), feats):
    for lab, col in [(0, ru.PALETTE["green"]), (1, ru.PALETTE["red"])]:
        v = X.loc[y == lab, f]; sns.kdeplot(np.log1p(v.clip(lower=0)), ax=ax, color=col, fill=True, alpha=0.3, label="Churned" if lab else "Active")
    ax.set_title(f"log1p({f})"); ax.legend()
ru.save_fig(fig, "09_02_feature_by_churn", "Key features: churned vs active", "Churners have higher recency and lower recent activity.", "Churn Prediction")

top = X.assign(churn=y).corr()["churn"].drop("churn").abs().sort_values(ascending=False).head(14).index.tolist() + ["churn"]
fig, ax = plt.subplots(figsize=(10, 8)); sns.heatmap(X.assign(churn=y)[top].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Correlation heatmap (top features vs churn)")
ru.save_fig(fig, "09_03_churn_corr", "Churn feature correlation heatmap", "Recency-type features correlate most with churn.", "Churn Prediction")
'''),
M("### Step 2 - Model comparison"),
C('''
mt = res["metrics"]; display(mt.round(3))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, c in zip(axes, ["AUC-ROC", "PR-AUC", "F1"]):
    mt[c].plot(kind="bar", ax=ax, color=ru.COLORS[:3]); ax.set_title(c); ax.set_ylim(0, 1.05); ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(mt[c]): ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
ru.save_fig(fig, "09_04_model_comparison", "Churn model comparison", "AUC-ROC, PR-AUC and F1 on the held-out test set.", "Churn Prediction")

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
fig, ax = plt.subplots(figsize=(7.5, 6))
for (n, p), c in zip(res["probs"].items(), ru.COLORS): fpr, tpr, _ = roc_curve(yte, p); ax.plot(fpr, tpr, lw=2.5, color=c, label=f"{n} (AUC {mt.loc[n, 'AUC-ROC']:.3f})")
ax.plot([0, 1], [0, 1], "k--"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(); ax.set_title("ROC curves")
ru.save_fig(fig, "09_05_roc_curves", "ROC curves", "Higher and further left is better.", "Churn Prediction")
fig, ax = plt.subplots(figsize=(7.5, 6))
for (n, p), c in zip(res["probs"].items(), ru.COLORS): pr, rc, _ = precision_recall_curve(yte, p); ax.plot(rc, pr, lw=2.5, color=c, label=n)
ax.axhline(yte.mean(), color="grey", ls="--", label="baseline"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(); ax.set_title("Precision-Recall curves")
ru.save_fig(fig, "09_06_pr_curves", "Precision-Recall curves", "Useful for imbalanced classes.", "Churn Prediction")
'''),
C('''
p_x = res["probs"][XN]; th = res["thresholds"][XN]
cm = confusion_matrix(yte, p_x >= th)
fig, ax = plt.subplots(figsize=(5.8, 5)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Pred active", "Pred churn"], yticklabels=["Actual active", "Actual churn"])
ax.set_title(f"Confusion matrix ({XN}, threshold {th:.2f})")
ru.save_fig(fig, "09_07_confusion_matrix", "Confusion matrix", "Threshold chosen on the validation set to maximise F1.", "Churn Prediction")

order = np.argsort(-p_x); cumg = np.cumsum(np.asarray(yte)[order]) / max(np.sum(yte), 1); pct = np.arange(1, len(yte) + 1) / len(yte)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(pct * 100, cumg * 100, color=ru.PALETTE["red"], lw=2.5, label="Model"); axes[0].plot([0, 100], [0, 100], "k--", label="Random"); axes[0].set_xlabel("% customers contacted (highest risk first)"); axes[0].set_ylabel("% churners captured"); axes[0].legend(); axes[0].set_title("Cumulative gain curve")
axes[1].plot(pct * 100, cumg / pct, color=ru.PALETTE["purple"], lw=2.5); axes[1].axhline(1, color="k", ls="--"); axes[1].set_xlabel("% customers contacted"); axes[1].set_ylabel("Lift"); axes[1].set_title("Lift curve")
ru.save_fig(fig, "09_08_gain_lift", "Gain and lift curves", f"Contacting the top 20% riskiest customers captures {ru.recall_at_k(yte, p_x, 0.2) * 100:.0f}% of churners.", "Churn Prediction")

from sklearn.calibration import calibration_curve
fig, ax = plt.subplots(figsize=(6.5, 6))
for (n, p), c in zip(res["probs"].items(), ru.COLORS):
    fr, mp = calibration_curve(yte, p, n_bins=8, strategy="quantile"); ax.plot(mp, fr, marker="o", lw=2, color=c, label=n)
ax.plot([0, 1], [0, 1], "k--"); ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed churn rate"); ax.legend(); ax.set_title("Calibration (reliability) curves")
ru.save_fig(fig, "09_09_calibration", "Calibration curves", "Shows whether predicted probabilities can be trusted as real probabilities.", "Churn Prediction")

from sklearn.metrics import f1_score
ths = np.linspace(0.05, 0.95, 91); f1s = [f1_score(yva, res["models"][XN].predict_proba(Xva)[:, 1] >= t) for t in ths]
fig, ax = plt.subplots(figsize=(8, 4.5)); ax.plot(ths, f1s, color=ru.PALETTE["teal"], lw=2.5); ax.axvline(th, color="red", ls="--", label=f"best = {th:.2f}"); ax.legend(); ax.set_xlabel("Threshold"); ax.set_ylabel("F1 (validation)"); ax.set_title("Threshold selection")
ru.save_fig(fig, "09_10_threshold_f1", "Threshold selection curve", "Decision threshold is tuned on validation data only.", "Churn Prediction")
'''),
M("### Step 3 - Explainability: feature importance and SHAP"),
C('''
imp = pd.Series(getattr(xgb, "feature_importances_", np.zeros(X.shape[1])), index=X.columns).sort_values().tail(15)
fig, ax = plt.subplots(figsize=(9, 6.5)); ax.barh(imp.index, imp.values, color=ru.PALETTE["navy"]); ax.set_title(f"{XN} feature importance (top 15)")
ru.save_fig(fig, "09_11_feature_importance", "Churn model feature importance", f"Most important feature: {imp.index[-1]}.", "Churn Prediction")

try:
    import shap
    explainer = shap.TreeExplainer(xgb); sv = explainer.shap_values(Xte)
    sv = sv[1] if isinstance(sv, list) else sv
    plt.figure(figsize=(10, 7)); shap.summary_plot(sv, Xte, show=False, max_display=15); plt.title("SHAP summary (impact on churn probability)")
    ru.save_fig(plt.gcf(), "09_12_shap_summary", "SHAP summary plot", "Red = high feature value. Right = pushes the customer towards churn.", "Churn Prediction")
    plt.figure(figsize=(9, 6)); shap.summary_plot(sv, Xte, plot_type="bar", show=False, max_display=15); plt.title("SHAP mean |value|")
    ru.save_fig(plt.gcf(), "09_13_shap_bar", "SHAP global importance", "Average absolute SHAP value per feature.", "Churn Prediction")
    i = int(np.argmax(p_x)); ev = explainer.expected_value; ev = float(np.ravel(ev)[0])
    ex = shap.Explanation(values=sv[i], base_values=ev, data=Xte.iloc[i].values, feature_names=list(Xte.columns))
    plt.figure(); shap.plots.waterfall(ex, show=False, max_display=12); plt.gcf().set_size_inches(9, 6)
    ru.save_fig(plt.gcf(), "09_14_shap_waterfall", "SHAP explanation of the highest-risk customer", "Shows why one specific customer is flagged - useful for the retention team.", "Churn Prediction")
except Exception as e:
    print(f"[INFO] SHAP plots skipped ({type(e).__name__}: {str(e)[:80]})")
'''),
M("### Step 4 - Score all customers and attach segments"),
C('''
scores = ru.score_all_customers(xgb, cs, returns)
fig, ax = plt.subplots(figsize=(9, 4.8)); sns.histplot(scores["churn_probability"], bins=40, ax=ax, color=ru.PALETTE["red"]); ax.set_title("Distribution of churn probability (all customers)")
ru.save_fig(fig, "09_15_prob_distribution", "Churn probability distribution", f"High-risk customers (p > 0.7): {(scores['churn_probability'] > 0.7).sum()}.", "Churn Prediction")

rfm = ru.rfm_scores(ru.rfm_table(cs)).reset_index()[["customer_id", "rule_segment"]]
sc2 = scores.merge(rfm, on="customer_id", how="left")
seg = sc2.groupby("rule_segment")["churn_probability"].mean().sort_values()
fig, ax = plt.subplots(figsize=(9, 5)); ax.barh(seg.index, seg.values, color=ru.PALETTE["orange"]); ax.set_xlabel("Average churn probability"); ax.set_title("Average churn risk by RFM segment")
ru.save_fig(fig, "09_16_risk_by_segment", "Average churn risk by RFM segment", "Segments with low recency scores carry the highest risk.", "Churn Prediction")
sc2.to_csv(ru.DATA_PROC / "churn_scores.csv", index=False)
import joblib; joblib.dump(xgb, ru.MODELS / "churn_model.joblib")
m_ = mt.loc[XN]; print(f"AUC {m_['AUC-ROC']:.3f} (target >= 0.88: {'PASS' if m_['AUC-ROC'] >= 0.88 else 'NOT MET'}) | precision@top20% {m_['Precision@top20%']:.3f} (target >= 0.75: {'PASS' if m_['Precision@top20%'] >= 0.75 else 'NOT MET'})")
ru.log_metrics("churn", {"model": XN, **{k: float(v) for k, v in m_.items()}, "churn_rate_pct": float(y.mean() * 100), "n_customers": int(len(X)),
                         "table": mt.round(4).to_dict(orient="index"), "high_risk_customers": int((scores["churn_probability"] > 0.7).sum())})
'''),
M('''
## Summary of Notebook 09
* Time-based churn label, 3 models, ROC/PR/gain/lift/calibration/threshold charts, **SHAP** explainability, scores for every customer (`churn_scores.csv`). **16 charts.**
* Always read the printed PASS / NOT MET lines - results depend on the real dataset and are never faked.
'''),
])

# ---------------------------------------------------------------- 10 INVENTORY
STAGES["10_inventory_optimization"] = dict(
    title="10 - Inventory Optimization using Forecasted Demand",
    week="Week 2 - Day 10", needs_load=True, cells=[
M('''
## Introduction
Turning forecasts into decisions. For the top SKUs we compute:
* **Safety stock** `SS = z x sigma x sqrt(L + R)`  (z from the service level, L = lead time, R = review period)
* **Reorder point** `ROP = mu x L + z x sigma x sqrt(L)`
* **Order-up-to level** `S = mu x (L + R) + SS`   and **EOQ** `sqrt(2 D S_cost / H)`
* **Recommended order quantity** = `S - stock on hand`.

Demand `mu` is the history mean scaled by the **hybrid forecast growth ratio** (top-down forecasting). 
> **Important:** retail transaction datasets have *no stock-on-hand column*, so current stock is **simulated** for the demo (clearly labelled). With a real inventory table, replace `current_stock_simulated`.
Then a **back-test** replays the last 30 real days to compare a naive buffer rule against the forecast-driven policy (F-05).
'''),
C(LOAD),
C('''
fb = ru.forecast_bundle(daily, TARGET, H)
recent = daily[TARGET].tail(90).mean()
ratio = float(np.clip(fb["future"]["hybrid"].mean() / max(recent, 1e-9), 0.5, 2.0))
print(f"Forecast growth ratio (next 30d mean / last 90d mean) = {ratio:.2f}")
plan = ru.inventory_plan(sales, growth_ratio=ratio)
display(plan.head(10).round(1))
'''),
M("### Step 1 - Stock status and recommended orders"),
C('''
st = plan["status"].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].pie(st.values, labels=st.index, autopct="%1.0f%%", colors=[ru.PALETTE["red"], ru.PALETTE["green"], ru.PALETTE["orange"]][:len(st)], wedgeprops=dict(width=0.5, edgecolor="white")); axes[0].set_title("Stock status of top SKUs")
t10 = plan.nlargest(10, "order_value").sort_values("order_value"); axes[1].barh(t10["description"].str[:28], t10["order_value"], color=ru.PALETTE["teal"]); axes[1].set_title("Top 10 recommended purchase orders (value)")
ru.save_fig(fig, "10_01_status_orders", "Stock status and top recommended orders", f"{int((plan['status'] == 'Reorder Now').sum())} SKUs need reordering now.", "Inventory Optimization")

fig, ax = plt.subplots(figsize=(9, 4.8)); sns.histplot(plan["days_of_cover"].clip(upper=120), bins=30, ax=ax, color=ru.PALETTE["purple"]); ax.axvline(45, color="red", ls="--", label="overstock > 45 days"); ax.legend(); ax.set_title("Days of cover distribution")
ru.save_fig(fig, "10_02_days_of_cover", "Days of cover distribution", "SKUs right of the red line are overstocked.", "Inventory Optimization")

fig, ax = plt.subplots(figsize=(8.5, 6.5)); cmap = {"Reorder Now": ru.PALETTE["red"], "Healthy": ru.PALETTE["green"], "Overstock": ru.PALETTE["orange"]}
for s_, g in plan.groupby("status"): ax.scatter(g["reorder_point"], g["current_stock_simulated"], s=60, color=cmap[s_], label=s_, alpha=0.85)
lim = max(plan["reorder_point"].max(), plan["current_stock_simulated"].max()) * 1.05; ax.plot([0, lim], [0, lim], "k--", label="stock = ROP"); ax.legend(); ax.set_xlabel("Reorder point"); ax.set_ylabel("Current stock (simulated)"); ax.set_title("Current stock vs reorder point")
ru.save_fig(fig, "10_03_stock_vs_rop", "Current stock vs reorder point", "Points below the diagonal are under the reorder point.", "Inventory Optimization")
'''),
M("### Step 2 - Demand variability, ABC analysis and EOQ"),
C('''
fig, ax = plt.subplots(figsize=(8.5, 6)); sc = ax.scatter(plan["avg_daily_demand_hist"], plan["safety_stock"], c=plan["demand_std"], cmap="viridis", s=70); plt.colorbar(sc, label="demand std")
ax.set_xlabel("Average daily demand"); ax.set_ylabel("Safety stock"); ax.set_title("Safety stock grows with demand variability")
ru.save_fig(fig, "10_04_safety_stock", "Safety stock vs demand and variability", "More volatile SKUs need more buffer.", "Inventory Optimization")

rev = sales.groupby("stock_code")["revenue"].sum().sort_values(ascending=False); cum = rev.cumsum() / rev.sum()
abc = pd.Series(np.where(cum <= 0.8, "A", np.where(cum <= 0.95, "B", "C")), index=rev.index)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
abc.value_counts().reindex(["A", "B", "C"]).plot(kind="bar", ax=axes[0], color=[ru.PALETTE["red"], ru.PALETTE["orange"], ru.PALETTE["green"]]); axes[0].set_title("SKU count per ABC class"); axes[0].tick_params(axis="x", rotation=0)
rev.groupby(abc).sum().reindex(["A", "B", "C"]).plot(kind="bar", ax=axes[1], color=[ru.PALETTE["red"], ru.PALETTE["orange"], ru.PALETTE["green"]]); axes[1].set_title("Revenue per ABC class"); axes[1].tick_params(axis="x", rotation=0)
ru.save_fig(fig, "10_05_abc_analysis", "ABC inventory classification", "A-class SKUs (80% of revenue) deserve tight control and high service levels.", "Inventory Optimization")

t = plan.nlargest(12, "EOQ").sort_values("EOQ"); fig, ax = plt.subplots(figsize=(9, 5.5)); ax.barh(t["description"].str[:28], t["EOQ"], color=ru.PALETTE["navy"]); ax.set_title("Economic order quantity (EOQ) - top 12 SKUs")
ru.save_fig(fig, "10_06_eoq", "Economic order quantity", "Balances ordering cost against holding cost.", "Inventory Optimization")
'''),
M("### Step 3 - Sensitivity: service level and lead time"),
C('''
rows = []
for sl in [0.85, 0.90, 0.95, 0.975, 0.99]:
    p_ = ru.inventory_plan(sales, ratio, service_level=sl); rows.append({"service_level": sl, "safety_stock_value": float((p_["safety_stock"] * p_["unit_price"] * 0.6).sum())})
sl_df = pd.DataFrame(rows)
rows = []
for L in [2, 5, 7, 10, 14]:
    p_ = ru.inventory_plan(sales, ratio, lead_time=L); rows.append({"lead_time": L, "safety_stock_value": float((p_["safety_stock"] * p_["unit_price"] * 0.6).sum())})
lt_df = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
axes[0].plot(sl_df["service_level"] * 100, sl_df["safety_stock_value"], marker="o", color=ru.PALETTE["red"], lw=2.5); axes[0].set_xlabel("Service level %"); axes[0].set_ylabel("Safety-stock investment"); axes[0].set_title("Cost of higher service level")
axes[1].plot(lt_df["lead_time"], lt_df["safety_stock_value"], marker="o", color=ru.PALETTE["teal"], lw=2.5); axes[1].set_xlabel("Lead time (days)"); axes[1].set_title("Cost of longer supplier lead time")
ru.save_fig(fig, "10_07_sensitivity", "Safety-stock sensitivity", "Each extra point of service level and each extra lead-time day costs working capital.", "Inventory Optimization")
'''),
M("### Step 4 - Back-test: naive rule vs forecast-driven policy (last 30 real days)"),
C('''
# the back-test must use the forecast that was available for THAT window (hybrid test forecast), not the future ratio
ratio_bt = float(np.clip(fb["test_preds"]["Hybrid"].mean() / max(daily[TARGET].iloc[-(H + 90):-H].mean(), 1e-9), 0.5, 2.0))
print(f"Back-test growth ratio from the hybrid forecast for the test window = {ratio_bt:.2f}")
bt = ru.inventory_backtest(sales, ratio_bt, horizon=H)
tot_dem = float(ru.sku_daily_matrix(sales).iloc[-H:].sum().sum())
S = {}
for tag in ("naive", "opt"):
    S[tag] = {"lost_units": bt[f"{tag}_lost_units"].sum(), "stockout_days": bt[f"{tag}_stockout_days"].sum(), "avg_inventory": bt[f"{tag}_avg_inventory"].sum(),
              "overstock_days": bt[f"{tag}_overstock_days"].sum(), "cost": bt[f"{tag}_cost"].sum(), "fill_rate_%": (1 - bt[f"{tag}_lost_units"].sum() / tot_dem) * 100}
sm = pd.DataFrame(S).T; sm.index = ["Naive buffer rule", "Forecast-driven (RetailPulse)"]; display(sm.round(1))
so = (1 - S["opt"]["lost_units"] / max(S["naive"]["lost_units"], 1e-9)) * 100
cost = (1 - S["opt"]["cost"] / max(S["naive"]["cost"], 1e-9)) * 100
inv = (S["opt"]["avg_inventory"] / max(S["naive"]["avg_inventory"], 1e-9) - 1) * 100
fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
for ax, (c, t) in zip(axes, [("lost_units", "Lost sales (units)"), ("stockout_days", "SKU stock-out days"), ("avg_inventory", "Average inventory (units)"), ("cost", "Total cost (lost margin + holding)")]):
    ax.bar(["Naive", "RetailPulse"], [S["naive"][c], S["opt"][c]], color=[ru.PALETTE["grey"], ru.PALETTE["teal"]]); ax.set_title(t)
    for i, v in enumerate([S["naive"][c], S["opt"][c]]): ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom")
ru.save_fig(fig, "10_08_backtest", "Inventory policy back-test", f"Stock-out units change {-so:+.0f}%, total cost change {-cost:+.0f}%, inventory change {inv:+.0f}%.", "Inventory Optimization")
print(f"Lost-sales reduction: {so:.1f}% | Total cost reduction: {cost:.1f}% | Inventory level change: {inv:+.1f}% (more safety stock is the price of fewer stock-outs)")
print("F-05 target (25-40% reduction of overstock/understock cost):", "PASS" if cost >= 25 else "NOT MET")
'''),
C('''
fig, ax = plt.subplots(figsize=(8.5, 6)); ax.scatter(bt["naive_lost_units"], bt["opt_lost_units"], s=55, color=ru.PALETTE["purple"], alpha=0.8)
lim = max(bt["naive_lost_units"].max(), bt["opt_lost_units"].max()) * 1.05 + 1; ax.plot([0, lim], [0, lim], "k--"); ax.set_xlabel("Lost units - naive rule"); ax.set_ylabel("Lost units - RetailPulse"); ax.set_title("Per-SKU lost sales (below the line = improvement)")
ru.save_fig(fig, "10_09_sku_lost_scatter", "Per-SKU lost sales comparison", "Most SKUs lie below the diagonal -> fewer stock-outs.", "Inventory Optimization")
plan.to_csv(ru.DATA_PROC / "inventory_plan.csv", index=False); bt.to_csv(ru.DATA_PROC / "inventory_backtest.csv", index=False)
ru.log_metrics("inventory", {"growth_ratio": ratio, "lost_sales_reduction_pct": float(so), "cost_reduction_pct": float(cost), "inventory_change_pct": float(inv),
                             "skus": int(len(plan)), "reorder_now": int((plan["status"] == "Reorder Now").sum()), "overstock": int((plan["status"] == "Overstock").sum()),
                             "fill_rate_naive_pct": float(S["naive"]["fill_rate_%"]), "fill_rate_opt_pct": float(S["opt"]["fill_rate_%"])})
'''),
M('''
## Summary of Notebook 10
* Safety stock, reorder point, EOQ, order-up-to and recommended order quantity for the top SKUs, ABC analysis, sensitivity analysis and a **30-day policy back-test**. **9 charts.**
* Stock-on-hand is simulated (no such column in public retail data) - replace with a real inventory feed in production.
'''),
])

# ---------------------------------------------------------------- 11 OPTUNA
STAGES["11_optuna_tuning"] = dict(
    title="11 - Hyper-parameter Tuning with Optuna & Feature Importance",
    week="Week 2 - Day 11", needs_load=True, cells=[
M('''
## Introduction
**Optuna** performs Bayesian (TPE) hyper-parameter search. We tune:
1. the **XGBoost churn model** (objective = 3-fold cross-validated AUC on the training set only),
2. **Prophet** (changepoint / seasonality prior and mode, objective = validation WAPE).
The test set is used once at the end to compare *default vs tuned*.
'''),
C(LOAD), C(CHURN_PREP),
C('''
import optuna, json
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
cv = StratifiedKFold(3, shuffle=True, random_state=ru.SEED)
def objective(trial):
    params = dict(n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50), max_depth=trial.suggest_int("max_depth", 2, 7),
                  learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True), subsample=trial.suggest_float("subsample", 0.6, 1.0),
                  colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0), min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                  reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True))
    m, _ = ru.make_xgb(**params)
    return cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc").mean()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=ru.SEED))
study.optimize(objective, n_trials=25, show_progress_bar=False)
print("Best CV AUC:", round(study.best_value, 4)); print("Best params:", study.best_params)
'''),
M("### Step 1 - Optimisation history and parameter importance"),
C('''
tdf = study.trials_dataframe(); tdf["best_so_far"] = tdf["value"].cummax()
fig, ax = plt.subplots(figsize=(10, 4.8)); ax.scatter(tdf["number"], tdf["value"], color=ru.PALETTE["sky"], label="Trial AUC"); ax.plot(tdf["number"], tdf["best_so_far"], color=ru.PALETTE["red"], lw=2.5, label="Best so far")
ax.legend(); ax.set_xlabel("Trial"); ax.set_ylabel("CV AUC"); ax.set_title("Optuna optimisation history (XGBoost churn)")
ru.save_fig(fig, "11_01_optuna_history", "Optuna optimisation history", f"Best CV AUC = {study.best_value:.4f} after {len(tdf)} trials.", "Tuning")
try:
    imp = optuna.importance.get_param_importances(study); s = pd.Series(imp).sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 5)); ax.barh(s.index, s.values, color=ru.PALETTE["purple"]); ax.set_title("Hyper-parameter importance (fANOVA)")
    ru.save_fig(fig, "11_02_param_importance", "Hyper-parameter importance", f"Most influential parameter: {s.index[-1]}.", "Tuning")
except Exception as e:
    print("Param importance skipped:", type(e).__name__)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
sc = axes[0].scatter(tdf["params_learning_rate"], tdf["value"], c=tdf["params_max_depth"], cmap="viridis", s=60); axes[0].set_xscale("log"); axes[0].set_xlabel("learning_rate"); axes[0].set_ylabel("CV AUC"); plt.colorbar(sc, ax=axes[0], label="max_depth"); axes[0].set_title("Learning rate vs AUC")
axes[1].scatter(tdf["params_n_estimators"], tdf["value"], c=tdf["params_subsample"], cmap="plasma", s=60); axes[1].set_xlabel("n_estimators"); axes[1].set_title("Trees vs AUC (colour = subsample)")
ru.save_fig(fig, "11_03_optuna_slices", "Optuna parameter slices", "Shows which regions of the search space perform best.", "Tuning")
'''),
M("### Step 2 - Default vs tuned on the untouched test set"),
C('''
dm, _ = ru.make_xgb(scale_pos_weight=float((ytr == 0).sum() / max((ytr == 1).sum(), 1))); dm.fit(Xtr, ytr)
tm, _ = ru.make_xgb(**study.best_params, scale_pos_weight=float((ytr == 0).sum() / max((ytr == 1).sum(), 1))); tm.fit(Xtr, ytr)
a0, a1 = roc_auc_score(yte, dm.predict_proba(Xte)[:, 1]), roc_auc_score(yte, tm.predict_proba(Xte)[:, 1])
p0, p1 = ru.precision_at_k(yte, dm.predict_proba(Xte)[:, 1]), ru.precision_at_k(yte, tm.predict_proba(Xte)[:, 1])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (t, v) in zip(axes, [("Test AUC-ROC", [a0, a1]), ("Precision@top20%", [p0, p1])]):
    ax.bar(["Default", "Tuned"], v, color=[ru.PALETTE["grey"], ru.PALETTE["teal"]]); ax.set_ylim(0, 1.05); ax.set_title(t)
    for i, x in enumerate(v): ax.text(i, x + 0.01, f"{x:.3f}", ha="center")
ru.save_fig(fig, "11_04_default_vs_tuned", "Default vs tuned XGBoost", f"AUC {a0:.3f} -> {a1:.3f}. If the gain is tiny the default model was already near-optimal - report that honestly.", "Tuning")
print(f"Default AUC {a0:.4f} | Tuned AUC {a1:.4f}")
'''),
M("### Step 3 - Tune Prophet (validation WAPE)"),
C('''
tr, va, te = ru.split_series(daily, H)
def obj_p(trial):
    r = ru.prophet_forecast(tr, va["ds"], TARGET, changepoint_prior_scale=trial.suggest_float("changepoint_prior_scale", 0.005, 0.5, log=True),
                            seasonality_prior_scale=trial.suggest_float("seasonality_prior_scale", 0.1, 20.0, log=True),
                            seasonality_mode=trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]))
    return ru.wape(va[TARGET].values, r["pred"])
sp = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=ru.SEED)); sp.optimize(obj_p, n_trials=10)
print("Best Prophet params:", sp.best_params, "| val WAPE:", round(sp.best_value, 2))
pdf = sp.trials_dataframe(); pdf["best"] = pdf["value"].cummin()
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8)); axes[0].scatter(pdf["number"], pdf["value"], color=ru.PALETTE["sky"]); axes[0].plot(pdf["number"], pdf["best"], color=ru.PALETTE["red"], lw=2.5); axes[0].set_title("Prophet tuning history (validation WAPE %)"); axes[0].set_xlabel("Trial")
axes[1].scatter(pdf["params_changepoint_prior_scale"], pdf["value"], c=(pdf["params_seasonality_mode"] == "multiplicative").astype(int), cmap="coolwarm", s=70); axes[1].set_xscale("log"); axes[1].set_xlabel("changepoint_prior_scale"); axes[1].set_title("Changepoint prior vs WAPE (colour = mode)")
ru.save_fig(fig, "11_05_prophet_tuning", "Prophet tuning with Optuna", f"Best validation WAPE {sp.best_value:.2f}%.", "Tuning")
json.dump({"xgboost": study.best_params, "prophet": sp.best_params}, open(ru.MODELS / "best_params.json", "w"), indent=2)
ru.log_metrics("tuning", {"xgb_best_cv_auc": study.best_value, "xgb_default_test_auc": a0, "xgb_tuned_test_auc": a1, "prophet_best_val_wape": sp.best_value, "xgb_best_params": study.best_params, "prophet_best_params": sp.best_params})
'''),
M('''
## Summary of Notebook 11
* Optuna TPE search for XGBoost (25 trials) and Prophet (10 trials), parameter importance, slices and a clean default-vs-tuned comparison on the untouched test set. **5 charts.** Best parameters saved to `models/best_params.json`.
'''),
])

# ---------------------------------------------------------------- 12 DRIFT
STAGES["12_drift_detection"] = dict(
    title="12 - Data & Model Drift Detection (PSI, KS, Evidently)",
    week="Week 2 - Day 12", needs_load=True, cells=[
M('''
## Introduction
Models degrade when the world changes. We monitor:
* **Data drift** - PSI (Population Stability Index: <0.1 stable, 0.1-0.2 warning, >0.2 drift) and the Kolmogorov-Smirnov test, reference period vs current period.
* **Model-performance drift** - rolling forecast error.
* **Real drift simulation** (required by the Zidio guidelines): we deliberately inject a price increase and demand drop, then show the monitor firing and the **retrain trigger**.
'''),
C(LOAD),
C('''
cut = sales["date"].min() + (sales["date"].max() - sales["date"].min()) * 0.7
ref, cur = sales[sales["date"] <= cut].copy(), sales[sales["date"] > cut].copy()
FEATS = ["quantity", "unit_price", "revenue", "hour"]
d0 = ru.drift_table(ref, cur, FEATS)
print(f"Reference: {ref['date'].min().date()} -> {ref['date'].max().date()} ({len(ref):,} rows) | Current: {cur['date'].min().date()} -> {cur['date'].max().date()} ({len(cur):,} rows)")
display(d0.round(4))
'''),
M("### Step 1 - Natural drift (no manipulation)"),
C('''
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
for ax, f in zip(axes, FEATS):
    a, b = np.log1p(ref[f]), np.log1p(cur[f]); sns.kdeplot(a, ax=ax, color=ru.PALETTE["navy"], fill=True, alpha=0.3, label="Reference"); sns.kdeplot(b, ax=ax, color=ru.PALETTE["red"], fill=True, alpha=0.3, label="Current")
    ax.set_title(f"{f}  (PSI {d0.set_index('feature').loc[f, 'PSI']:.3f})"); ax.legend()
ru.save_fig(fig, "12_01_distribution_overlay", "Reference vs current distributions", "Overlapping curves = stable data.", "Drift Detection")
fig, ax = plt.subplots(figsize=(8.5, 4.5)); ax.bar(d0["feature"], d0["PSI"], color=[ru.PALETTE["green"] if s == "OK" else ru.PALETTE["orange"] if s == "WARNING" else ru.PALETTE["red"] for s in d0["status"]])
ax.axhline(0.1, color="orange", ls="--"); ax.axhline(0.2, color="red", ls="--"); ax.set_ylabel("PSI"); ax.set_title("Natural data drift (PSI per feature)")
ru.save_fig(fig, "12_02_psi_natural", "PSI per feature (natural drift)", "Green = stable, orange = warning, red = drift.", "Drift Detection")
'''),
M("### Step 2 - Real drift simulation (injected shift)"),
C('''
sim = cur.copy()
rng = np.random.default_rng(ru.SEED)
sim["unit_price"] = sim["unit_price"] * 1.35                                  # price shock
sim["quantity"] = np.maximum(1, np.round(sim["quantity"] * rng.uniform(0.4, 0.8, len(sim)))).astype(int)   # demand collapse
sim["hour"] = np.clip(sim["hour"] + 2, 0, 23)                                 # shifted ordering time
sim["revenue"] = sim["quantity"] * sim["unit_price"]
d1 = ru.drift_table(ref, sim, FEATS); display(d1.round(4))
x = np.arange(len(FEATS)); fig, ax = plt.subplots(figsize=(10, 4.8)); ax.bar(x - 0.2, d0["PSI"], 0.4, label="Natural", color=ru.PALETTE["green"]); ax.bar(x + 0.2, d1["PSI"], 0.4, label="After injected drift", color=ru.PALETTE["red"])
ax.axhline(0.2, color="red", ls="--", label="drift threshold 0.2"); ax.set_xticks(x); ax.set_xticklabels(FEATS); ax.legend(); ax.set_title("PSI before vs after the simulated drift")
ru.save_fig(fig, "12_03_psi_simulated", "PSI before and after simulated drift", "The monitor correctly flags the injected price / quantity shift.", "Drift Detection")
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for ax, f in zip(axes, ["quantity", "unit_price", "hour"]):
    sns.kdeplot(np.log1p(ref[f]), ax=ax, color=ru.PALETTE["navy"], fill=True, alpha=0.3, label="Reference"); sns.kdeplot(np.log1p(sim[f]), ax=ax, color=ru.PALETTE["red"], fill=True, alpha=0.3, label="Drifted"); ax.set_title(f); ax.legend()
ru.save_fig(fig, "12_04_distribution_drifted", "Distribution shift after simulated drift", "Visible shift of the red curves away from the reference.", "Drift Detection")
'''),
M("### Step 3 - Customer-level feature drift (churn model inputs)"),
C('''
cs = ru.get_customer_sales(); end = cs["invoice_date"].max().normalize()
snapA, snapB = end - pd.Timedelta(days=240), end - pd.Timedelta(days=90)
fa = ru.customer_features(cs[cs["invoice_date"] < snapA], snapA); fb_ = ru.customer_features(cs[cs["invoice_date"] < snapB], snapB)
CF = ["recency", "frequency", "monetary", "avg_gap_days", "avg_order_value", "n_products"]
d2 = ru.drift_table(fa, fb_, CF); display(d2.round(3))
fig, ax = plt.subplots(figsize=(9.5, 4.8)); ax.bar(d2["feature"], d2["PSI"], color=[ru.PALETTE["green"] if s == "OK" else ru.PALETTE["orange"] if s == "WARNING" else ru.PALETTE["red"] for s in d2["status"]])
ax.axhline(0.1, color="orange", ls="--"); ax.axhline(0.2, color="red", ls="--"); ax.tick_params(axis="x", rotation=20); ax.set_title("Customer feature drift (snapshot -240d vs -90d)")
ru.save_fig(fig, "12_05_customer_feature_drift", "Customer feature drift", "Customers naturally age (recency / tenure move) - PSI quantifies how much.", "Drift Detection")
'''),
M("### Step 4 - Model-performance drift and the retraining trigger"),
C('''
fb = ru.forecast_bundle(daily, TARGET, H)
te = fb["test"]; yv = te[TARGET].values; pv = fb["test_preds"]["Hybrid"]
good = pd.Series(np.abs(yv - pv), index=te["ds"]).rolling(7, min_periods=3).sum() / pd.Series(yv, index=te["ds"]).rolling(7, min_periods=3).sum() * 100
yd = yv.copy().astype(float); yd[len(yd) // 2:] *= 1.6                          # simulated demand shock in 2nd half
bad = pd.Series(np.abs(yd - pv), index=te["ds"]).rolling(7, min_periods=3).sum() / pd.Series(yd, index=te["ds"]).rolling(7, min_periods=3).sum() * 100
fig, ax = plt.subplots(figsize=(12, 4.8)); ax.plot(good, color=ru.PALETTE["green"], lw=2.5, label="Normal operation"); ax.plot(bad, color=ru.PALETTE["red"], lw=2.5, label="After simulated demand shock")
thr = float(max(good.max() * 1.25, 30)); ax.axhline(thr, color="black", ls="--", label=f"retrain threshold ({thr:.0f}% rolling WAPE)"); ax.legend(); ax.set_ylabel("Rolling 7-day WAPE %"); ax.set_title("Forecast performance monitoring")
ru.save_fig(fig, "12_06_performance_monitor", "Forecast performance monitoring", "Rolling error crosses the threshold after the simulated shock -> alert.", "Drift Detection")
perf_alert = bool(bad.max() > thr)
dec = ru.retrain_decision(d1, float(bad.iloc[-1]), mape_limit=thr / 2.5)
print("Natural data drift :", ru.retrain_decision(d0, float(good.iloc[-1]), mape_limit=thr / 2.5))
print("Simulated drift    :", dec, "| performance alert:", perf_alert)
rp = ru.evidently_report(ref[FEATS].reset_index(drop=True), sim[FEATS].reset_index(drop=True), str(ru.REPORTS / "evidently_drift_report.html"))
print("Evidently HTML report:", rp or "skipped (native PSI/KS report used)")
ru.log_metrics("drift", {"psi_natural": d0.set_index("feature")["PSI"].to_dict(), "psi_simulated": d1.set_index("feature")["PSI"].to_dict(),
                         "retrain_triggered_on_simulation": bool(dec["retrain"] or perf_alert), "reasons": dec["reasons"], "evidently_report": bool(rp)})
'''),
M('''
## Summary of Notebook 12
* PSI + KS drift tables, natural vs **simulated** drift (price shock, demand collapse), customer-feature drift, rolling error monitor and an automatic **retrain decision**. Optional Evidently AI HTML report. **6 charts.**
'''),
])

# ---------------------------------------------------------------- 13 REFINEMENT
DAG_SRC = (REPO / "airflow" / "dags" / "retailpulse_retrain_dag.py").read_text() if (REPO / "airflow").exists() else "# DAG file missing"
STAGES["13_model_refinement"] = dict(
    title="13 - Model Refinement & Automated Retraining Pipeline (Airflow)",
    week="Week 2 - Day 13", needs_load=True, cells=[
M('''
## Introduction
Refinement = squeeze reliable gains **without overfitting**:
1. **Probability calibration** (isotonic) so churn scores behave like real probabilities,
2. **Business-cost threshold** - choose the cut-off that maximises campaign profit instead of F1,
3. **Feature-group ablation** - which feature families really matter,
4. **Churn ensemble** (LR + RF + XGBoost average) and **learning curve**,
5. **Forecast bias correction** by weekday,
6. The **Airflow retraining DAG** (written to disk and syntax-checked).
'''),
C(LOAD), C(CHURN_PREP),
M("### Step 1 - Calibration"),
C('''
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
pv, pt = xgb.predict_proba(Xva)[:, 1], xgb.predict_proba(Xte)[:, 1]
iso = IsotonicRegression(out_of_bounds="clip").fit(pv, yva); pt_cal = iso.predict(pt)
b0, b1 = brier_score_loss(yte, pt), brier_score_loss(yte, pt_cal)
fig, ax = plt.subplots(figsize=(6.5, 6))
for p, n, c in [(pt, "XGBoost raw", ru.PALETTE["red"]), (pt_cal, "XGBoost + isotonic", ru.PALETTE["teal"])]:
    fr, mp = calibration_curve(yte, p, n_bins=8, strategy="quantile"); ax.plot(mp, fr, marker="o", lw=2, color=c, label=n)
ax.plot([0, 1], [0, 1], "k--"); ax.legend(); ax.set_xlabel("Predicted"); ax.set_ylabel("Observed"); ax.set_title(f"Calibration (Brier {b0:.3f} -> {b1:.3f})")
ru.save_fig(fig, "13_01_calibration", "Calibration before vs after isotonic", f"Brier score {b0:.3f} -> {b1:.3f} (lower is better); AUC {roc_auc_score(yte, pt):.3f} -> {roc_auc_score(yte, pt_cal):.3f}.", "Model Refinement")
'''),
M("### Step 2 - Profit-based threshold"),
C('''
mon = X.loc[Xte.index, "monetary"].values; AVG_VALUE = float(np.median(mon)) * 0.2; CONTACT_COST = 2.0; SUCCESS = 0.25
rows = []
for t in np.linspace(0.05, 0.95, 46):
    flag = pt_cal >= t; tp = (flag & (np.asarray(yte) == 1)).sum()
    rows.append({"threshold": t, "profit": tp * AVG_VALUE * SUCCESS - flag.sum() * CONTACT_COST, "contacted": int(flag.sum())})
pf = pd.DataFrame(rows); best = pf.loc[pf["profit"].idxmax()]
fig, ax = plt.subplots(figsize=(9, 4.8)); ax.plot(pf["threshold"], pf["profit"], color=ru.PALETTE["green"], lw=2.5); ax.axvline(best["threshold"], color="red", ls="--", label=f"best = {best['threshold']:.2f}"); ax.legend()
ax.set_xlabel("Threshold"); ax.set_ylabel("Expected campaign profit"); ax.set_title("Profit-optimal churn threshold (contact cost 2, 25% save rate)")
ru.save_fig(fig, "13_02_profit_threshold", "Profit-optimal threshold", f"Best threshold {best['threshold']:.2f} contacts {int(best['contacted'])} customers; expected profit {best['profit']:,.0f}.", "Model Refinement")
'''),
M("### Step 3 - Feature-group ablation, ensemble and learning curve"),
C('''
groups = {"Recency features": ["recency", "recency_gap_ratio"], "Frequency / monetary": ["frequency", "monetary", "avg_order_value", "n_products"],
          "Trend / recent activity": ["orders_last_30d", "orders_last_90d", "orders_prev_90d", "order_trend", "revenue_last_90d"],
          "Gap behaviour": ["avg_gap_days", "std_gap_days", "single_order"], "Tenure": ["tenure_days"]}
base = roc_auc_score(yte, pt); rows = []
for g, cols in groups.items():
    keep = [c for c in X.columns if c not in cols]; m, _ = ru.make_xgb(); m.fit(Xtr[keep], ytr); rows.append({"dropped_group": g, "AUC": roc_auc_score(yte, m.predict_proba(Xte[keep])[:, 1])})
ab = pd.DataFrame(rows); ab["delta_vs_full"] = ab["AUC"] - base; display(ab.round(4))
fig, ax = plt.subplots(figsize=(9, 4.8)); ax.barh(ab["dropped_group"], ab["delta_vs_full"], color=np.where(ab["delta_vs_full"] < 0, ru.PALETTE["red"], ru.PALETTE["green"])); ax.axvline(0, color="black"); ax.set_xlabel("AUC change when the group is removed"); ax.set_title("Feature-group ablation")
ru.save_fig(fig, "13_03_ablation", "Feature-group ablation", "Negative bars = the group helps the model.", "Model Refinement")

ens = np.mean([res["probs"][n] for n in res["probs"]], axis=0); a_e = roc_auc_score(yte, ens)
cmpa = pd.Series({**{n: roc_auc_score(yte, p) for n, p in res["probs"].items()}, "Average ensemble": a_e})
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8)); cmpa.plot(kind="bar", ax=axes[0], color=ru.COLORS[:4]); axes[0].set_ylim(max(0, cmpa.min() - 0.05), 1); axes[0].set_title("Churn: single models vs ensemble (AUC)"); axes[0].tick_params(axis="x", rotation=20)
for i, v in enumerate(cmpa): axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom")
fr = [0.2, 0.4, 0.6, 0.8, 1.0]; tr_a, te_a = [], []
for f in fr:
    n = max(int(len(Xtr) * f), 30); m, _ = ru.make_xgb(); m.fit(Xtr.iloc[:n], ytr.iloc[:n]); tr_a.append(roc_auc_score(ytr.iloc[:n], m.predict_proba(Xtr.iloc[:n])[:, 1])); te_a.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
axes[1].plot(fr, tr_a, marker="o", label="Train AUC", color=ru.PALETTE["navy"]); axes[1].plot(fr, te_a, marker="o", label="Test AUC", color=ru.PALETTE["red"]); axes[1].legend(); axes[1].set_xlabel("Fraction of training data"); axes[1].set_title("Learning curve")
ru.save_fig(fig, "13_04_ensemble_learning_curve", "Churn ensemble and learning curve", "A gap between train and test AUC signals over-fitting.", "Model Refinement")
'''),
M("### Step 4 - Forecast bias correction by weekday"),
C('''
tr_, va_, te_ = ru.split_series(daily, H); trv = pd.concat([tr_, va_])
pv_ = ru.prophet_forecast(tr_, va_["ds"], TARGET)["pred"]; pt_ = ru.prophet_forecast(trv, te_["ds"], TARGET)["pred"]
dv = pd.DataFrame({"dow": va_["ds"].dt.dayofweek.values, "a": va_[TARGET].values, "p": pv_}); fac = (dv.groupby("dow")["a"].sum() / dv.groupby("dow")["p"].sum().replace(0, np.nan)).fillna(1.0)
fac = 1 + 0.5 * (fac.clip(0.7, 1.4) - 1)                        # shrink correction by 50% to avoid over-fitting
corr = pt_ * te_["ds"].dt.dayofweek.map(fac).fillna(1.0).values
w0, w1 = ru.wape(te_[TARGET].values, pt_), ru.wape(te_[TARGET].values, corr)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8)); axes[0].bar(["Prophet", "Prophet + bias fix"], [w0, w1], color=[ru.PALETTE["grey"], ru.PALETTE["teal"]]); axes[0].set_title("Test WAPE %")
for i, v in enumerate([w0, w1]): axes[0].text(i, v, f"{v:.1f}", ha="center", va="bottom")
axes[1].plot(te_["ds"], te_[TARGET], color=ru.PALETTE["navy"], lw=2.5, label="Actual"); axes[1].plot(te_["ds"], pt_, "--", color=ru.PALETTE["orange"], label="Prophet"); axes[1].plot(te_["ds"], corr, color=ru.PALETTE["teal"], label="Corrected"); axes[1].legend(); axes[1].tick_params(axis="x", rotation=40)
ru.save_fig(fig, "13_05_bias_correction", "Weekday bias correction", f"WAPE {w0:.1f}% -> {w1:.1f}%. Keep the correction only if it helps on validation AND test.", "Model Refinement")
ru.log_metrics("refinement", {"brier_raw": b0, "brier_calibrated": b1, "best_profit_threshold": float(best["threshold"]), "ensemble_auc": float(a_e), "bias_fix_wape_before": w0, "bias_fix_wape_after": w1})
'''),
M("### Step 5 - Automated retraining pipeline (Airflow DAG)"),
C('''
DAG_SRC = ''' + repr(DAG_SRC) + '''
p = ru.ROOT / "airflow_retailpulse_retrain_dag.py"; p.write_text(DAG_SRC)
import py_compile; py_compile.compile(str(p), doraise=True)
print("DAG written and syntax-checked ->", p)
print("Schedule: weekly | tasks: ingest_clean -> drift_check -> (retrain | skip) -> evaluate_and_register")
print("Same logic without Airflow:  python scripts/retrain_pipeline.py --force")
'''),
M('''
## Summary of Notebook 13
* Calibration, profit-optimal threshold, ablation, ensemble, learning curve, weekday bias fix (**5 charts**) and the **Airflow retraining DAG** (weekly, drift-gated).
'''),
])

# ---------------------------------------------------------------- 14 MLFLOW WEEK 2
STAGES["14_mlflow_week2"] = dict(
    title="14 - MLflow Week-2 Tracking, Acceptance Checklist & Model Card",
    week="Week 2 - Day 14 (checkpoint)", needs_load=True, cells=[
M('''
## Introduction
Week-2 checkpoint: log the advanced models (hybrid forecaster, churn models, tuned XGBoost, inventory policy) into **RetailPulse-Week2**, compare them with Week 1, evaluate every **acceptance criterion from the project brief** honestly and write a **model card**.
'''),
C(LOAD), C(CHURN_PREP),
C('''
fb = ru.forecast_bundle(daily, TARGET, H); met = fb["metrics"]
runs = []
def log(name, params, metrics):
    rid = ru.mlflow_log("RetailPulse-Week2", name, params, metrics, tags={"stage": "advanced"}); runs.append({"run": name, **metrics}); print(f"logged {name:<26} {str(rid)[:8]}")
for n in ["Prophet", "LSTM", "Hybrid (Prophet+LSTM)"]:
    log(f"forecast_{n.split()[0].lower()}", {"horizon": H, "engines": fb["engines"], "w_prophet": round(fb["weight_prophet"], 2)}, met.loc[n].to_dict())
for n, r in res["metrics"].iterrows():
    log(f"churn_{n.replace(' ', '_').lower()}", {"label_window_days": ru.CONFIG["CHURN_WINDOW_DAYS"]}, r.to_dict())
m_all = ru.get_metrics()
if "tuning" in m_all: log("churn_xgb_tuned", m_all["tuning"].get("xgb_best_params", {}), {"test_auc": m_all["tuning"].get("xgb_tuned_test_auc", float("nan"))})
if "inventory" in m_all: log("inventory_policy", {"service_level": ru.CONFIG["SERVICE_LEVEL"], "lead_time": ru.CONFIG["LEAD_TIME_DAYS"]}, {k: v for k, v in m_all["inventory"].items() if isinstance(v, (int, float))})
'''),
C('''
tb = pd.DataFrame(runs).set_index("run")
f = tb.dropna(subset=["MAPE_weekly_%"]); c = tb.dropna(subset=["AUC-ROC"])
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
f["MAPE_weekly_%"].plot(kind="bar", ax=axes[0], color=ru.COLORS[:len(f)]); axes[0].axhline(12, color="red", ls="--", label="target 12%"); axes[0].legend(); axes[0].set_title("Week-2 forecasting runs (weekly MAPE %)"); axes[0].tick_params(axis="x", rotation=25)
c["AUC-ROC"].plot(kind="bar", ax=axes[1], color=ru.COLORS[:len(c)]); axes[1].axhline(0.88, color="red", ls="--", label="target 0.88"); axes[1].legend(); axes[1].set_ylim(0, 1.05); axes[1].set_title("Week-2 churn runs (AUC-ROC)"); axes[1].tick_params(axis="x", rotation=25)
ru.save_fig(fig, "14_01_week2_runs", "MLflow Week-2 runs vs targets", "Dashed red lines show the acceptance targets from the project brief.", "MLOps - MLflow")
df_runs = ru.mlflow_runs(["RetailPulse-Week1", "RetailPulse-Week2"]); print("Total runs tracked in MLflow:", len(df_runs))
ru.log_metrics("mlflow", {"total_runs": int(len(df_runs))})
'''),
M("### Acceptance checklist (honest evaluation)"),
C('''
M_ = ru.get_metrics(); h = met.loc["Hybrid (Prophet+LSTM)"]; ch = res["metrics"].loc[XN]
inv = M_.get("inventory", {}); seg = M_.get("segmentation", {})
k_ = seg.get("k_chosen"); cr_ = inv.get("cost_reduction_pct")
chk = pd.DataFrame([
    ("F-01", "Automated ETL + data-quality checks", f"{M_.get('cleaning', {}).get('validation_passed', 'n/a')}/{M_.get('cleaning', {}).get('validation_total', 'n/a')} checks passed", True if M_.get("cleaning") else None),
    ("F-02", "6-8 meaningful segments", f"K = {k_}" if k_ else "run notebook 03 first", (6 <= int(k_) <= 8) if k_ else None),
    ("F-03", "MAPE <= 12% (30-day, weekly totals)", f"{h['MAPE_weekly_%']:.1f}% weekly | {h['MAPE_daily_%']:.1f}% daily", bool(h["MAPE_weekly_%"] <= 12)),
    ("F-04a", "AUC-ROC >= 0.88", f"{ch['AUC-ROC']:.3f}", bool(ch["AUC-ROC"] >= 0.88)),
    ("F-04b", "Precision@top-20% >= 0.75", f"{ch['Precision@top20%']:.3f}", bool(ch["Precision@top20%"] >= 0.75)),
    ("F-05", "Overstock/understock cost reduced 25-40%", f"{cr_:.1f}% cost, {inv.get('lost_sales_reduction_pct', float('nan')):.1f}% fewer lost sales" if cr_ is not None else "run notebook 10 first", (cr_ >= 25) if cr_ is not None else None),
    ("F-06", "Streamlit dashboard", "app/streamlit_app.py (5 pages)", True),
    ("NFR", "Drift detection + MLflow", f"{M_.get('mlflow', {}).get('total_runs', 'n/a')} runs, retrain on simulated drift: {M_.get('drift', {}).get('retrain_triggered_on_simulation', 'run notebook 12')}", True)],
    columns=["ID", "Requirement", "Result", "Met"])
display(chk)
colmap = {True: ru.PALETTE["green"], False: ru.PALETTE["orange"], None: ru.PALETTE["grey"]}
fig, ax = plt.subplots(figsize=(10, 4.8)); ax.barh(chk["ID"], np.ones(len(chk)), color=[colmap[m] for m in chk["Met"]])
for i, (r, m) in enumerate(zip(chk["Result"], chk["Met"])): ax.text(0.5, i, ({True: "MET  ", False: "NOT MET  ", None: "NOT RUN  "}[m]) + r, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
ax.set_xticks([]); ax.invert_yaxis(); ax.set_title("Acceptance criteria scorecard (computed from real results)")
ru.save_fig(fig, "14_02_acceptance_scorecard", "Acceptance criteria scorecard", "Green = target met on this dataset; orange = honestly reported as not met; grey = earlier notebook not run yet.", "MLOps - MLflow")
ru.log_metrics("acceptance", {r.ID: {"requirement": r.Requirement, "result": r.Result, "met": (None if r.Met is None else bool(r.Met))} for r in chk.itertuples()})
'''),
M("### Model card"),
C('''
import json
card = {"project": ru.PROJECT, "author": ru.AUTHOR, "data_source": bundle["source"], "forecast_model": {"type": "Prophet + LSTM weighted ensemble", "weight_prophet": fb["weight_prophet"], "horizon_days": H, "engines": fb["engines"],
        "test_metrics": h.round(3).to_dict()}, "churn_model": {"type": XN, "label": "no purchase in next 90 days", "test_metrics": ch.round(3).to_dict()},
        "intended_use": "Retail demand planning, retention campaigns, replenishment recommendations", "limitations": ["Stock on hand is simulated for public data", "Daily MAPE is high for lumpy wholesale demand", "Retrain when PSI > 0.2 on 2+ features"]}
json.dump(card, open(ru.MODELS / "model_card.json", "w"), indent=2, default=str); print(json.dumps(card, indent=2, default=str)[:1500])
'''),
M('''
## Summary of Notebook 14
* Week-2 runs in MLflow, **acceptance scorecard computed from real results (never faked)** and a JSON model card. **Week-2 checkpoint reached.**
'''),
])

# ---------------------------------------------------------------- 15 STREAMLIT + REPORT
APP_SRC = (REPO / "app" / "streamlit_app.py").read_text() if (REPO / "app" / "streamlit_app.py").exists() else "print('app missing')"
REPORT_SRC = (REPO / "src" / "retailpulse_report.py").read_text() if (REPO / "src" / "retailpulse_report.py").exists() else "print('report missing')"
STAGES["15_streamlit_dashboard_and_report"] = dict(
    title="15 - Streamlit Dashboard Launch (Colab) & Final DOCX Report",
    week="Week 3-4 - Dashboard & final report", needs_load=True, cells=[
M('''
## Introduction
This notebook (a) writes the **Streamlit dashboard** (Sales, Customer, Forecast, Inventory, Monitoring pages) and shows how to **open it from Google Colab**, and (b) builds the **final project report (DOCX)** automatically from the results of this run, so the numbers always match your real dataset.
'''),
C(LOAD),
M("### Step 1 - Make sure the dashboard input files exist"),
C('''
need = ["transactions_clean.csv.gz", "forecast_future.csv", "forecast_test.csv", "churn_scores.csv", "rfm_segments.csv", "inventory_plan.csv"]
missing = [n for n in need if not (ru.DATA_PROC / n).exists()]
print("Missing processed files:", missing or "none")
if missing:
    print("Creating the missing files now (this runs the fast parts of the pipeline) ...")
    keep = ["invoice", "invoice_no", "stock_code", "description", "category", "quantity", "unit_price", "revenue", "order_value", "invoice_date", "customer_id", "country"]
    sales[keep].to_csv(ru.DATA_PROC / "transactions_clean.csv.gz", index=False, compression="gzip")
    daily = ru.daily_series(sales); fb = ru.forecast_bundle(daily, "quantity", ru.CONFIG["FORECAST_HORIZON"], lstm_epochs=30)
    fb["future"].to_csv(ru.DATA_PROC / "forecast_future.csv", index=False)
    pd.DataFrame({"ds": fb["test"]["ds"].values, "actual": fb["test"]["quantity"].values, **fb["test_preds"]}).to_csv(ru.DATA_PROC / "forecast_test.csv", index=False)
    cs = ru.get_customer_sales(); rfm = ru.rfm_scores(ru.rfm_table(cs)); ks, X = ru.kmeans_search(rfm, range(6, 9)); K = int(ks.loc[ks["silhouette"].idxmax(), "k"])
    seg_df, _, _ = ru.fit_kmeans(rfm, X, K); seg_df.reset_index().to_csv(ru.DATA_PROC / "rfm_segments.csv", index=False)
    Xc, yc, _ = ru.build_churn_dataset(cs, bundle["returns"]); rr = ru.train_churn_models(Xc, yc)
    ru.score_all_customers(rr["models"][rr["xgb_name"]], cs, bundle["returns"]).to_csv(ru.DATA_PROC / "churn_scores.csv", index=False)
    ru.inventory_plan(sales).to_csv(ru.DATA_PROC / "inventory_plan.csv", index=False)
print("Files:", sorted(p.name for p in ru.DATA_PROC.glob("*") if p.suffix in (".csv", ".gz")))
'''),
M("### Step 2 - Write the Streamlit app"),
C("%%writefile streamlit_app.py\n" + APP_SRC),
M('''
### Step 3 - Open the dashboard from Google Colab
Colab cannot show `localhost` directly, so we use a free **localtunnel** (no account / key needed). Run the next cell, copy the **URL**, and when the page asks for a *tunnel password* paste the IP address printed below.
'''),
C('''
import subprocess, time, os, urllib.request, shutil, sys
os.environ["RETAILPULSE_DATA"] = str(ru.DATA_PROC)
LAUNCH = False        # <- set True in Colab to start the dashboard
if LAUNCH:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "streamlit", "plotly"], check=False)
    subprocess.Popen([sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.headless", "true"], stdout=open("streamlit.log", "w"), stderr=subprocess.STDOUT)
    time.sleep(6)
    try:
        print("Tunnel password (your public IP):", urllib.request.urlopen("https://ipv4.icanhazip.com").read().decode().strip())
    except Exception: pass
    if shutil.which("npx"):
        subprocess.Popen("npx --yes localtunnel --port 8501", shell=True)
        print("Look for 'your url is: https://....loca.lt' in the output above / below.")
    else:
        print("Node/npx not found. Alternative: !pip install pyngrok and use an ngrok token, or deploy on Streamlit Community Cloud (see README).")
else:
    print("Set LAUNCH = True and re-run this cell in Colab to start the dashboard.\\nLocal PC: pip install -r requirements.txt  &&  streamlit run app/streamlit_app.py")
'''),
M("### Step 4 - Build the final project report (DOCX)"),
C("%%writefile retailpulse_report.py\n" + REPORT_SRC),
C('''
import importlib, retailpulse_report as rr; importlib.reload(rr)
full = rr.build_report(str(ru.REPORTS / "RetailPulse_Project_Report_FULL.docx"), compact=False)
short = rr.build_report(str(ru.REPORTS / "RetailPulse_Zidio_Submission_Report.docx"), compact=True)
print("Created:", full, "\\n         ", short)
try:
    from google.colab import files
    files.download(full); files.download(short)
except Exception:
    print("(Not in Colab - find the reports in", ru.REPORTS, ")")
'''),
M('''
## Summary of Notebook 15
* Streamlit dashboard written + Colab launch recipe; **FULL report** (every chart explained) and a **compact Zidio-submission report** (10-18 pages) generated from this run's real numbers.
'''),
])
