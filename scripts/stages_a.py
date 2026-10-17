"""Notebook stage definitions 01-04. Each stage = list of (kind, text)."""

def M(t): return ("md", t.strip("\n"))
def C(t): return ("code", t.strip("\n"))

LOAD = '''
bundle = ru.get_data()
sales, returns, rep, raw = bundle["sales"], bundle["returns"], bundle["report"], bundle["raw"]
print("DATA SOURCE :", bundle["source"])
print("Clean rows  :", f"{len(sales):,}", "| Date range:", sales["date"].min().date(), "->", sales["date"].max().date())
'''

STAGES = {}

# ---------------------------------------------------------------- 01 EDA
STAGES["01_eda_exploration"] = dict(
    title="01 - Data Collection & Exploratory Data Analysis (EDA)",
    week="Week 1 - Day 1", needs_load=False, cells=[
M('''
## Introduction
**RetailPulse** is an end-to-end retail analytics platform. In this notebook (Day 1) we **collect the dataset** and perform a deep **Exploratory Data Analysis**:
distribution analysis, missing values, seasonality, product / country / customer behaviour and correlation heatmap.

**Why EDA first?** Every later model (segmentation, forecasting, churn, inventory) is only as good as our understanding of the data. EDA tells us what cleaning is needed, which patterns (weekly / yearly seasonality) the forecaster must capture, and where the business revenue really comes from.

**How data is loaded (no zip file needed):**
1. `DATASET="auto"` -> uses a file you uploaded, else downloads from Kaggle with `kagglehub`, else falls back to built-in demo data (so the notebook never fails).
2. Set `UPLOAD_MY_FILE=True` to open the Colab file-picker and upload your own CSV / XLSX.
3. Set `DATA_PATH="/content/drive/MyDrive/yourfile.xlsx"` to read from Google Drive.
'''),
C('''
ru.draw_banner()
ru.draw_architecture()
ru.draw_workflow()
'''),
M('''
### Step 1 - Dataset choice & loading
Supported Kaggle datasets: **Online Retail II (UCI)** `cgrymn/online-retail-ii-uci-dataset` and **Synthetic Retail 10M** `mrayyanshehzad/synthetic-retail-dataset-10-million-transactions` (very large -> automatically sampled by customer so Colab RAM is safe).
'''),
C('''
# ---------- USER SETTINGS ----------
DATASET        = "auto"    # "auto" | "uci" | "synthetic" | "upload" | "demo"
UPLOAD_MY_FILE = False     # True -> Colab file picker opens (CSV / XLSX)
DATA_PATH      = None      # e.g. "/content/drive/MyDrive/online_retail_II.xlsx"
MAX_ROWS       = 400_000   # sampling cap for huge files
# -----------------------------------
ru.CONFIG.update(DATASET=DATASET, PATH=DATA_PATH, MAX_ROWS=MAX_ROWS)
if UPLOAD_MY_FILE:
    try:
        from google.colab import files
        up = files.upload()
        for name, content in up.items():
            (ru.DATA_RAW / name).write_bytes(content)
        print("Uploaded:", list(up))
    except ImportError:
        print("Not running in Colab -> copy your file into:", ru.DATA_RAW)

bundle = ru.get_data(force=True)          # load + clean (cached for next notebooks)
sales, returns, rep, raw = bundle["sales"], bundle["returns"], bundle["report"], bundle["raw"]
print("\\nDATA SOURCE :", bundle["source"])
print("Raw rows    :", f"{len(raw):,}", "| Clean sales rows:", f"{len(sales):,}", "| Returns:", f"{len(returns):,}")
'''),
M('''
> **Comment (real dataset):** On the real *Online Retail II (UCI)* file we get ~1.07 million raw rows covering Dec-2009 -> Dec-2011 for a UK online gift-ware wholesaler. Roughly 20-25% rows have no `Customer ID` (guest checkouts) and ~2% are cancellation invoices starting with `C`. The numbers printed above are computed live from *your* loaded data.
'''),
C('''
display(raw.head())
print(raw.dtypes)
display(raw.describe(include="all").T.head(12))
'''),
M("### Step 2 - Missing values (raw data)"),
C('''
miss = (raw.isna().mean() * 100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.barplot(x=miss.values, y=miss.index, ax=ax, color=ru.PALETTE["navy"])
for i, v in enumerate(miss.values):
    ax.text(v + 0.3, i, f"{v:.1f}%", va="center")
ax.set_xlabel("% missing"); ax.set_title("Missing values by column (raw data)")
ru.save_fig(fig, "01_01_missing_values", "Missing values by column (raw data)",
            f"Highest missing share: {miss.index[0]} ({miss.iloc[0]:.1f}%). Missing customer IDs are guest checkouts and are excluded from customer analytics only.", "EDA")
'''),
M("### Step 3 - Data-quality issues found in raw data"),
C('''
issues = {
    "Cancelled invoices (C...)": int(raw["invoice"].str.upper().str.startswith("C").sum()),
    "Negative quantity": int((raw["quantity"] < 0).sum()),
    "Zero / negative price": int((raw["unit_price"] <= 0).sum()),
    "Missing customer id": int(raw["customer_id"].isna().sum()),
    "Missing description": int(raw["description"].isna().sum()),
    "Non-product codes (POST, D, M...)": int(raw["stock_code"].astype(str).str.upper().isin(ru._NON_PRODUCT).sum()),
    "Duplicate rows": int(raw.duplicated().sum()),
}
s = pd.Series(issues).sort_values()
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.barh(s.index, s.values, color=ru.COLORS[:len(s)])
for i, v in enumerate(s.values):
    ax.text(v, i, f" {v:,}", va="center")
ax.set_title("Data-quality issues detected in raw data"); ax.set_xlabel("rows affected")
ru.save_fig(fig, "01_02_data_quality_issues", "Data-quality issues detected in raw data",
            "Counts of cancellations, invalid prices, missing IDs and duplicates that the cleaning stage must handle.", "EDA")
'''),
M("### Step 4 - Business KPIs (cleaned sales)"),
C('''
kpi = {
    "Total revenue": sales["revenue"].sum(), "Orders (invoices)": sales["invoice"].nunique(),
    "Customers": sales["customer_id"].nunique(), "Products (SKUs)": sales["stock_code"].nunique(),
    "Countries": sales["country"].nunique(), "Avg order value": sales.groupby("invoice")["revenue"].sum().mean(),
    "Units sold": sales["quantity"].sum(),
}
for k, v in kpi.items():
    print(f"{k:<20}: {v:>16,.2f}")
ru.log_metrics("eda", {k: float(v) for k, v in kpi.items()})
'''),
M("### Step 5 - Monthly revenue, orders and customers"),
C('''
m = sales.groupby("year_month").agg(revenue=("revenue", "sum"), orders=("invoice", "nunique"), customers=("customer_id", "nunique")).reset_index()
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(m["year_month"], m["revenue"], color=ru.PALETTE["sky"], alpha=0.85, label="Revenue")
ax.set_ylabel("Revenue"); ax.tick_params(axis="x", rotation=60)
ax2 = ax.twinx()
ax2.plot(m["year_month"], m["orders"], color=ru.PALETTE["orange"], marker="o", lw=2.5, label="Orders")
ax2.set_ylabel("Orders"); ax2.grid(False)
ax.set_title("Monthly revenue (bars) and number of orders (line)")
peak = m.loc[m["revenue"].idxmax()]
ru.save_fig(fig, "01_03_monthly_revenue_orders", "Monthly revenue and orders",
            f"Peak month is {peak['year_month']} with revenue {peak['revenue']:,.0f}. Strong Q4 seasonality is typical for gift retail.", "EDA")
'''),
C('''
g = m["revenue"].pct_change() * 100
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.bar(m["year_month"], g, color=np.where(g >= 0, ru.PALETTE["green"], ru.PALETTE["red"]))
ax.axhline(0, color="black", lw=0.8); ax.tick_params(axis="x", rotation=60)
ax.set_ylabel("MoM growth %"); ax.set_title("Month-over-month revenue growth")
ru.save_fig(fig, "01_04_mom_growth", "Month-over-month revenue growth",
            "Green bars show growth, red bars show decline versus the previous month.", "EDA")
'''),
M("### Step 6 - Products and categories"),
C('''
top_rev = sales.groupby("description")["revenue"].sum().nlargest(10).sort_values()
fig, ax = plt.subplots(figsize=(10, 5)); ax.barh(top_rev.index, top_rev.values, color=ru.PALETTE["teal"])
ax.set_title("Top 10 products by revenue"); ax.set_xlabel("Revenue")
ru.save_fig(fig, "01_05_top_products_revenue", "Top 10 products by revenue", f"Best seller by revenue: {top_rev.index[-1]}.", "EDA")

top_qty = sales.groupby("description")["quantity"].sum().nlargest(10).sort_values()
fig, ax = plt.subplots(figsize=(10, 5)); ax.barh(top_qty.index, top_qty.values, color=ru.PALETTE["purple"])
ax.set_title("Top 10 products by units sold"); ax.set_xlabel("Units")
ru.save_fig(fig, "01_06_top_products_quantity", "Top 10 products by units sold", f"Most units sold: {top_qty.index[-1]}.", "EDA")
'''),
C('''
cat = sales.groupby("category")["revenue"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(cat.values, labels=cat.index, autopct="%1.1f%%", startangle=90, colors=ru.COLORS, wedgeprops=dict(width=0.45, edgecolor="white"))
ax.set_title("Revenue share by product category (keyword-inferred)")
ru.save_fig(fig, "01_07_category_share", "Revenue share by category", f"Largest category: {cat.index[0]} ({cat.iloc[0] / cat.sum() * 100:.1f}%).", "EDA")
'''),
M("### Step 7 - Geography"),
C('''
ctry = sales.groupby("country")["revenue"].sum().sort_values(ascending=False)
share = ctry.iloc[0] / ctry.sum() * 100
rest = ctry.iloc[1:11].sort_values()
fig, ax = plt.subplots(figsize=(10, 5)); ax.barh(rest.index, rest.values, color=ru.PALETTE["orange"])
ax.set_title(f"Top 10 countries by revenue (excluding {ctry.index[0]})"); ax.set_xlabel("Revenue")
ru.save_fig(fig, "01_08_country_revenue", "Revenue by country (excluding the dominant market)",
            f"{ctry.index[0]} contributes {share:.1f}% of revenue; export markets are shown separately.", "EDA")
'''),
M("### Step 8 - Time patterns: day of week, hour of day, heatmap"),
C('''
names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow = sales.groupby("dow")["revenue"].sum().reindex(range(7), fill_value=0)
fig, ax = plt.subplots(figsize=(9, 4.5)); ax.bar(names, dow.values, color=ru.COLORS[:7])
ax.set_title("Revenue by day of week"); ax.set_ylabel("Revenue")
ru.save_fig(fig, "01_09_revenue_by_dow", "Revenue by day of week", f"Strongest weekday: {names[int(dow.idxmax())]}. Days with zero revenue indicate the shop is closed.", "EDA")

hr = sales.groupby("hour")["invoice"].nunique()
fig, ax = plt.subplots(figsize=(10, 4.5)); ax.plot(hr.index, hr.values, marker="o", color=ru.PALETTE["navy"], lw=2.5)
ax.fill_between(hr.index, hr.values, alpha=0.15, color=ru.PALETTE["navy"]); ax.set_xlabel("Hour of day"); ax.set_ylabel("Orders")
ax.set_title("Orders by hour of day")
ru.save_fig(fig, "01_10_orders_by_hour", "Orders by hour of day", f"Peak ordering hour: {int(hr.idxmax())}:00.", "EDA")

ct = pd.crosstab(sales["dow"], sales["hour"]).reindex(range(7), fill_value=0)
fig, ax = plt.subplots(figsize=(12, 4.5)); sns.heatmap(ct, cmap="YlGnBu", ax=ax, yticklabels=names)
ax.set_title("Order lines heatmap: day of week x hour"); ax.set_xlabel("Hour"); ax.set_ylabel("")
ru.save_fig(fig, "01_11_dow_hour_heatmap", "Day x hour activity heatmap", "Darker cells mark busiest trading windows.", "EDA")
'''),
M("### Step 9 - Distributions & outliers"),
C('''
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
sns.histplot(np.log10(sales["quantity"]), bins=40, ax=axes[0], color=ru.PALETTE["navy"]); axes[0].set_title("log10(Quantity)")
sns.histplot(np.log10(sales["unit_price"]), bins=40, ax=axes[1], color=ru.PALETTE["teal"]); axes[1].set_title("log10(Unit price)")
sns.histplot(np.log10(sales["revenue"]), bins=40, ax=axes[2], color=ru.PALETTE["orange"]); axes[2].set_title("log10(Line revenue)")
ru.save_fig(fig, "01_12_distributions", "Distributions of quantity, price and revenue (log scale)",
            "All three variables are right-skewed, therefore log transforms are used for clustering and modelling.", "EDA")

fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=sales, x="category", y="revenue", ax=ax, showfliers=False, palette=ru.COLORS[:sales["category"].nunique()])
ax.tick_params(axis="x", rotation=25); ax.set_title("Line revenue by category (outliers hidden)")
ru.save_fig(fig, "01_13_revenue_box_category", "Line revenue by category (boxplot)", "Shows spread of line values per category.", "EDA")

basket = sales.groupby("invoice")["quantity"].sum()
fig, ax = plt.subplots(figsize=(10, 4.5)); sns.histplot(basket.clip(upper=basket.quantile(0.99)), bins=50, ax=ax, color=ru.PALETTE["purple"])
ax.set_title("Basket size (units per order, clipped at 99th pct)"); ax.set_xlabel("Units per order")
ru.save_fig(fig, "01_14_basket_size", "Basket size distribution", f"Median basket = {basket.median():.0f} units, mean = {basket.mean():.0f} units (wholesale-like behaviour).", "EDA")
'''),
M("### Step 10 - Pareto (80/20) analysis"),
C('''
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
pr = sales.groupby("stock_code")["revenue"].sum().sort_values(ascending=False)
cum = pr.cumsum() / pr.sum() * 100
axes[0].plot(np.arange(1, len(cum) + 1) / len(cum) * 100, cum.values, color=ru.PALETTE["navy"], lw=2.5)
axes[0].axhline(80, color="red", ls="--"); axes[0].set_xlabel("% of products"); axes[0].set_ylabel("Cumulative % revenue"); axes[0].set_title("Product Pareto")
n80 = int((cum <= 80).sum() + 1)
cs = ru.get_customer_sales().groupby("customer_id")["revenue"].sum().sort_values(ascending=False)
cumc = cs.cumsum() / cs.sum() * 100
axes[1].plot(np.arange(1, len(cumc) + 1) / len(cumc) * 100, cumc.values, color=ru.PALETTE["teal"], lw=2.5)
axes[1].axhline(80, color="red", ls="--"); axes[1].set_xlabel("% of customers"); axes[1].set_title("Customer Pareto")
n80c = int((cumc <= 80).sum() + 1)
ru.save_fig(fig, "01_15_pareto", "Pareto analysis of products and customers",
            f"{n80 / len(cum) * 100:.0f}% of products and {n80c / len(cumc) * 100:.0f}% of customers generate 80% of revenue.", "EDA")
'''),
M("### Step 11 - Correlation heatmap"),
C('''
num = sales[["quantity", "unit_price", "revenue", "hour", "dow", "month", "week", "is_weekend"]]
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
ax.set_title("Correlation heatmap of numeric features")
ru.save_fig(fig, "01_16_correlation_heatmap", "Correlation heatmap", "Revenue is driven by quantity; price and quantity are weakly (negatively) related.", "EDA")
'''),
M("### Step 12 - Trend, rolling averages and customer acquisition"),
C('''
d = sales.groupby("date")["revenue"].sum().asfreq("D", fill_value=0)
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(d.index, d.values, color="#CBD5E1", lw=1, label="Daily revenue")
ax.plot(d.index, d.rolling(7).mean(), color=ru.PALETTE["sky"], lw=2, label="7-day mean")
ax.plot(d.index, d.rolling(30).mean(), color=ru.PALETTE["red"], lw=2.5, label="30-day mean")
ax.legend(); ax.set_title("Daily revenue with rolling averages")
ru.save_fig(fig, "01_17_rolling_revenue", "Daily revenue with 7 and 30-day rolling averages", "Rolling means expose the trend and the seasonal build-up before the holiday season.", "EDA")

cs = ru.get_customer_sales()
first = cs.groupby("customer_id")["invoice_date"].min().dt.to_period("M").astype(str)
tmp = cs[["customer_id", "year_month"]].drop_duplicates()
tmp["is_new"] = tmp["customer_id"].map(first) == tmp["year_month"]
nr = tmp.groupby(["year_month", "is_new"]).size().unstack(fill_value=0).rename(columns={True: "New", False: "Returning"})
fig, ax = plt.subplots(figsize=(12, 4.8)); nr.plot(kind="bar", stacked=True, ax=ax, color=[ru.PALETTE["orange"], ru.PALETTE["navy"]])
ax.set_title("New vs returning customers per month"); ax.set_xlabel("")
ru.save_fig(fig, "01_18_new_vs_returning", "New vs returning customers per month", "Repeat buyers form the backbone of monthly activity.", "EDA")

hm = sales.pivot_table(index="year", columns="month", values="revenue", aggfunc="sum")
fig, ax = plt.subplots(figsize=(11, 3.8)); sns.heatmap(hm / 1000, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Revenue (thousands)"})
ax.set_title("Revenue heatmap: year x month (thousands)")
ru.save_fig(fig, "01_19_year_month_heatmap", "Year x month revenue heatmap", "Highlights seasonality (autumn / holiday peak).", "EDA")
'''),
M("### Step 13 - Automatic EDA insights"),
C('''
best_m = sales.groupby("month_name")["revenue"].sum().idxmax()
ins = {
    "peak_month_name": best_m, "top_product": top_rev.index[-1], "top_country_share_pct": float(share),
    "median_basket_units": float(basket.median()), "pct_products_for_80pct_revenue": float(n80 / len(cum) * 100),
}
ru.log_metrics("eda", ins)
print("KEY INSIGHTS")
print(f"1) Revenue peaks in {best_m}; plan stock before that month.")
print(f"2) Top product by revenue: {ins['top_product']}")
print(f"3) {ctry.index[0]} is {share:.1f}% of revenue -> export markets are an upside.")
print(f"4) {ins['pct_products_for_80pct_revenue']:.0f}% of SKUs drive 80% of revenue -> focus inventory on A-class items.")
'''),
M('''
## Summary of Notebook 01
* Loaded the dataset (Kaggle / upload / demo) and built a **clean sales table** with revenue and calendar features.
* Produced **19 charts**: missing values, data-quality issues, monthly revenue & orders, growth, top products, category share, country revenue, weekday/hour patterns, heatmaps, distributions, Pareto, correlation, rolling trend, new-vs-returning customers.
* **Next -> Notebook 02:** formal cleaning report, validation checks and feature engineering (RFM, lags, rolling statistics).
'''),
])

# ---------------------------------------------------------------- 02 CLEANING & FE
STAGES["02_data_cleaning_feature_engineering"] = dict(
    title="02 - Data Cleaning, Validation & Feature Engineering",
    week="Week 1 - Day 2", needs_load=True, cells=[
M('''
## Introduction
Raw retail data is messy: cancellations, zero prices, postage fees, duplicates, guest checkouts and extreme outliers. In this notebook we
1. document **what was removed and why** (cleaning waterfall),
2. run **data-validation expectations** (Great-Expectations-style checks implemented natively so they never break on API changes),
3. engineer features: **RFM**, **lag features**, **rolling statistics** and calendar features,
4. save processed files for the dashboard and **Power BI**.
'''),
C(LOAD),
M("### Step 1 - Cleaning waterfall (rows removed at each step)"),
C('''
steps = [("Raw rows", rep["raw_rows"]), ("Duplicates", -rep["duplicates_removed"]), ("Returns / cancellations", -rep["returns_separated"]),
         ("Invalid qty / price", -rep["invalid_qty_or_price_removed"]), ("Non-product codes", -rep["non_product_codes_removed"]),
         ("Extreme outliers", -rep["extreme_outliers_removed"]), ("Clean rows", rep["final_rows"])]
labels = [s[0] for s in steps]; vals = [s[1] for s in steps]
fig, ax = plt.subplots(figsize=(12, 5)); run = 0
for i, (l, v) in enumerate(steps):
    if i == 0 or i == len(steps) - 1:
        ax.bar(l, v, color=ru.PALETTE["navy"]); run = v if i == 0 else run
    else:
        ax.bar(l, v, bottom=run, color=ru.PALETTE["red"]); run += v
    ax.text(i, (run if 0 < i < len(steps) - 1 else v) + rep["raw_rows"] * 0.01, f"{v:,}", ha="center", fontsize=9)
ax.tick_params(axis="x", rotation=25); ax.set_title("Data cleaning waterfall")
ru.save_fig(fig, "02_01_cleaning_waterfall", "Data cleaning waterfall", "Shows exactly how many rows each cleaning rule removed.", "Cleaning & Feature Engineering")
print(f"Retention: {rep['final_rows'] / rep['raw_rows'] * 100:.1f}% of raw rows kept")
ru.log_metrics("cleaning", {"retention_pct": rep["final_rows"] / rep["raw_rows"] * 100, **rep})
'''),
M("### Step 2 - Before vs after: outlier handling"),
C('''
raw_pos = raw[(raw["quantity"] > 0) & (raw["unit_price"] > 0)]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].boxplot([np.log10(raw_pos["quantity"]), np.log10(sales["quantity"])], labels=["Before", "After"], patch_artist=True,
                boxprops=dict(facecolor="#BAE6FD")); axes[0].set_title("log10(quantity)")
axes[1].boxplot([np.log10(raw_pos["unit_price"]), np.log10(sales["unit_price"])], labels=["Before", "After"], patch_artist=True,
                boxprops=dict(facecolor="#BBF7D0")); axes[1].set_title("log10(unit price)")
ru.save_fig(fig, "02_02_outliers_before_after", "Outliers before vs after cleaning", f"Max quantity before {raw_pos['quantity'].max():,.0f} -> after {sales['quantity'].max():,.0f}.", "Cleaning & Feature Engineering")
'''),
M("### Step 3 - Data validation (Great-Expectations-style)"),
C('''
val = ru.validate_data(sales)
display(val)
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.barh(val["expectation"], np.ones(len(val)), color=np.where(val["passed"], ru.PALETTE["green"], ru.PALETTE["red"]))
for i, p in enumerate(val["passed"]): ax.text(0.5, i, "PASS" if p else "FAIL", ha="center", va="center", color="white", fontweight="bold")
ax.set_xticks([]); ax.set_title(f"Data validation: {int(val['passed'].sum())}/{len(val)} expectations passed")
ru.save_fig(fig, "02_03_validation_results", "Data validation results", "All checks should pass before the data is used for modelling.", "Cleaning & Feature Engineering")
ru.log_metrics("cleaning", {"validation_passed": int(val["passed"].sum()), "validation_total": int(len(val))})
'''),
M('''
### Step 4 - Feature engineering: RFM (customer level)
* **Recency** - days since last purchase, **Frequency** - number of orders, **Monetary** - total spend.
* Scores 1-5 via quintiles. These features feed segmentation (03) and churn (09).
'''),
C('''
cs = ru.get_customer_sales()
rfm = ru.rfm_scores(ru.rfm_table(cs))
display(rfm.head())
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, c, col in zip(axes, ["recency", "frequency", "monetary"], [ru.PALETTE["red"], ru.PALETTE["teal"], ru.PALETTE["navy"]]):
    sns.histplot(rfm[c].clip(upper=rfm[c].quantile(0.99)), bins=40, ax=ax, color=col); ax.set_title(c.capitalize())
ru.save_fig(fig, "02_04_rfm_distributions", "RFM feature distributions", "Frequency and monetary are heavily right-skewed - log scaling is applied before clustering.", "Cleaning & Feature Engineering")
'''),
M('''
### Step 5 - Feature engineering: lag features & rolling statistics (demand level)
Lags (1, 2, 3, 7, 14, 21, 28 days), rolling mean / std (7, 14, 28), EWM, expanding mean and calendar flags. All rolling features use `shift(1)` so **no future information leaks** into the features.
'''),
C('''
daily = ru.daily_series(sales)
lagf = ru.make_lag_features(daily, "quantity")
display(lagf.head())
print("Feature matrix:", lagf.shape)
corr = lagf.drop(columns=["ds"]).corr()["y"].drop("y").sort_values()
fig, ax = plt.subplots(figsize=(8, 8)); ax.barh(corr.index, corr.values, color=np.where(corr.values >= 0, ru.PALETTE["teal"], ru.PALETTE["red"]))
ax.set_title("Correlation of engineered features with daily demand")
ru.save_fig(fig, "02_05_feature_target_corr", "Feature correlation with demand", "Rolling means and lag-7 carry the strongest signal for next-day demand.", "Cleaning & Feature Engineering")
'''),
C('''
cols = ["y", "lag_1", "lag_7", "lag_14", "roll_mean_7", "roll_mean_28", "ewm_7", "expanding_mean"]
fig, ax = plt.subplots(figsize=(8.5, 6.5)); sns.heatmap(lagf[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Lag / rolling feature correlation matrix")
ru.save_fig(fig, "02_06_lag_corr_heatmap", "Lag feature correlation matrix", "Highly correlated rolling features - tree models handle this well, linear models need regularisation.", "Cleaning & Feature Engineering")

fig, ax = plt.subplots(figsize=(13, 4.8))
ax.plot(lagf["ds"], lagf["y"], color="#CBD5E1", label="Daily demand")
ax.plot(lagf["ds"], lagf["roll_mean_7"], color=ru.PALETTE["sky"], label="roll_mean_7 (shifted)")
ax.plot(lagf["ds"], lagf["roll_mean_28"], color=ru.PALETTE["red"], label="roll_mean_28 (shifted)")
ax.legend(); ax.set_title("Raw demand vs rolling-mean features")
ru.save_fig(fig, "02_07_rolling_features", "Raw demand vs rolling-mean features", "Shifted rolling means follow the signal with a lag, which is exactly what a forecaster is allowed to see.", "Cleaning & Feature Engineering")
'''),
M("### Step 6 - Save processed files (dashboard + Power BI)"),
C('''
keep = ["invoice", "invoice_no", "stock_code", "description", "category", "quantity", "unit_price", "revenue", "order_value", "invoice_date", "customer_id", "country"]
sales[keep].to_csv(ru.DATA_PROC / "transactions_clean.csv.gz", index=False, compression="gzip")
daily.to_csv(ru.DATA_PROC / "daily_series.csv", index=False)
lagf.to_csv(ru.DATA_PROC / "lag_features.csv", index=False)
rfm.reset_index().to_csv(ru.DATA_PROC / "rfm_table.csv", index=False)
pbi = sales[keep + ["year", "month", "month_name", "year_month", "dow", "hour", "week", "quarter"]].copy()
pbi["day_name"] = pbi["invoice_date"].dt.day_name()
pbi.to_csv(ru.DATA_PROC / "powerbi_ready.csv", index=False)
print("Saved:", sorted(p.name for p in ru.DATA_PROC.glob("*.csv*")))
'''),
M('''
## Summary of Notebook 02
* Cleaning waterfall, before/after outlier handling and **10 validation expectations**.
* Features: **RFM (+scores)**, **7 lags**, **rolling mean/std (7/14/28)**, EWM, expanding mean, calendar flags.
* Saved `transactions_clean.csv.gz`, `daily_series.csv`, `lag_features.csv`, `rfm_table.csv` and `powerbi_ready.csv`.
'''),
])

# ---------------------------------------------------------------- 03 SEGMENTATION
STAGES["03_customer_segmentation"] = dict(
    title="03 - Customer Segmentation (RFM, K-Means, DBSCAN, Cohorts)",
    week="Week 1 - Day 3", needs_load=True, cells=[
M('''
## Introduction
Not all customers are equal. We segment customers using **RFM analysis** (rule-based business segments) and **machine-learning clustering** (K-Means and DBSCAN) and then interpret every segment in business language with an action plan.
Acceptance criterion F-02: **6-8 meaningful segments with business interpretation**.
'''),
C(LOAD),
C('''
cs = ru.get_customer_sales()
rfm = ru.rfm_scores(ru.rfm_table(cs))
print("Customers:", len(rfm))
display(rfm[["recency", "frequency", "monetary", "R", "F", "M", "rule_segment"]].head())
'''),
M("### Step 1 - RFM distributions and scores"),
C('''
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, c, col in zip(axes, ["recency", "frequency", "monetary"], [ru.PALETTE["red"], ru.PALETTE["teal"], ru.PALETTE["navy"]]):
    sns.histplot(np.log1p(rfm[c]), bins=40, ax=ax, color=col, kde=True); ax.set_title(f"log1p({c})")
ru.save_fig(fig, "03_01_rfm_hist", "RFM distributions (log scale)", "Log scaling makes the three RFM variables close to bell-shaped, which suits K-Means.", "Customer Segmentation")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, c in zip(axes, ["R", "F", "M"]):
    rfm[c].value_counts().sort_index().plot(kind="bar", ax=ax, color=ru.PALETTE["sky"]); ax.set_title(f"{c} score distribution"); ax.set_xlabel("")
ru.save_fig(fig, "03_02_rfm_scores", "RFM score distributions", "Quintile scoring gives roughly equal customer counts per score.", "Customer Segmentation")
'''),
M("### Step 2 - Rule-based RFM segments (Champions, Loyal, At Risk ...)"),
C('''
seg = rfm.groupby("rule_segment").agg(customers=("monetary", "size"), revenue=("monetary", "sum"), avg_recency=("recency", "mean"),
                                      avg_frequency=("frequency", "mean")).sort_values("revenue", ascending=False)
seg["revenue_share_%"] = seg["revenue"] / seg["revenue"].sum() * 100
display(seg.round(1))
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
seg["customers"].sort_values().plot(kind="barh", ax=axes[0], color=ru.PALETTE["teal"]); axes[0].set_title("Customers per RFM segment")
seg["revenue_share_%"].sort_values().plot(kind="barh", ax=axes[1], color=ru.PALETTE["orange"]); axes[1].set_title("Revenue share % per RFM segment")
ru.save_fig(fig, "03_03_rule_segments", "Rule-based RFM segments: customers and revenue share", f"{seg.index[0]} generate the highest revenue share ({seg['revenue_share_%'].iloc[0]:.1f}%).", "Customer Segmentation")

hm = rfm.pivot_table(index="R", columns="F", values="monetary", aggfunc="mean").sort_index(ascending=False)
fig, ax = plt.subplots(figsize=(8, 6)); sns.heatmap(hm, annot=True, fmt=".0f", cmap="YlGn", ax=ax)
ax.set_title("Average monetary value by Recency score (R) x Frequency score (F)")
ru.save_fig(fig, "03_04_rf_heatmap", "Average spend by R x F score", "Recent and frequent buyers (top-right) are the most valuable.", "Customer Segmentation")
'''),
M("### Step 3 - Choosing K for K-Means (elbow, silhouette, Davies-Bouldin)"),
C('''
ks, X = ru.kmeans_search(rfm)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
axes[0].plot(ks["k"], ks["inertia"], marker="o", color=ru.PALETTE["navy"]); axes[0].set_title("Elbow (inertia)")
axes[1].plot(ks["k"], ks["silhouette"], marker="o", color=ru.PALETTE["teal"]); axes[1].set_title("Silhouette (higher = better)")
axes[2].plot(ks["k"], ks["davies_bouldin"], marker="o", color=ru.PALETTE["red"]); axes[2].set_title("Davies-Bouldin (lower = better)")
for a in axes: a.set_xlabel("k")
ru.save_fig(fig, "03_05_k_selection", "K selection: elbow, silhouette and Davies-Bouldin", "Metrics used to choose the number of clusters.", "Customer Segmentation")
cand = ks[(ks["k"] >= 6) & (ks["k"] <= 8)]
K = int(cand.loc[cand["silhouette"].idxmax(), "k"])
print(f"Chosen K = {K} (best silhouette inside the business range 6-8 required by F-02)")
display(ks.round(3))
'''),
M("### Step 4 - Final K-Means model and PCA visualisation"),
C('''
from sklearn.decomposition import PCA
seg_df, km, names = ru.fit_kmeans(rfm, X, K)
pc = PCA(n_components=3, random_state=ru.SEED).fit(X); P = pc.transform(X)
fig, ax = plt.subplots(figsize=(10, 7))
for i, (c, n) in enumerate(sorted(names.items())):
    m_ = seg_df["cluster"].values == c; ax.scatter(P[m_, 0], P[m_, 1], s=14, alpha=0.7, label=n, color=ru.COLORS[i % len(ru.COLORS)])
ax.legend(fontsize=8, markerscale=2); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(f"K-Means clusters (K={K}) in PCA space")
ru.save_fig(fig, "03_06_pca_clusters", "K-Means clusters in PCA space", f"PCA explains {pc.explained_variance_ratio_[:2].sum() * 100:.0f}% of RFM variance in 2D.", "Customer Segmentation")

from mpl_toolkits.mplot3d import Axes3D  # noqa
fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection="3d")
ax.scatter(np.log1p(seg_df["recency"]), np.log1p(seg_df["frequency"]), np.log1p(seg_df["monetary"]), c=seg_df["cluster"], cmap="tab10", s=10, alpha=0.7)
ax.set_xlabel("log Recency"); ax.set_ylabel("log Frequency"); ax.set_zlabel("log Monetary"); ax.set_title("3D view of RFM clusters")
ru.save_fig(fig, "03_07_3d_clusters", "3D RFM cluster view", "Each colour is one K-Means cluster in R-F-M space.", "Customer Segmentation")
'''),
M("### Step 5 - Segment profiles and business interpretation"),
C('''
prof = seg_df.groupby("segment").agg(customers=("monetary", "size"), recency=("recency", "mean"), frequency=("frequency", "mean"),
                                     monetary=("monetary", "mean"), total_revenue=("monetary", "sum")).sort_values("total_revenue", ascending=False)
prof["revenue_share_%"] = prof["total_revenue"] / prof["total_revenue"].sum() * 100
display(prof.round(1))

z = (prof[["recency", "frequency", "monetary"]] - prof[["recency", "frequency", "monetary"]].mean()) / prof[["recency", "frequency", "monetary"]].std().replace(0, 1)
fig, ax = plt.subplots(figsize=(8, 0.7 * len(z) + 2)); sns.heatmap(z, annot=prof[["recency", "frequency", "monetary"]].round(0), fmt="", cmap="RdYlGn_r", center=0, ax=ax, cbar_kws={"label": "z-score"})
ax.set_title("Cluster profile heatmap (values = cluster means)"); ax.set_ylabel("")
ru.save_fig(fig, "03_08_cluster_profile_heatmap", "Cluster profile heatmap", "Low recency + high frequency + high monetary = best customers.", "Customer Segmentation")

# Radar chart
cats = ["recency", "frequency", "monetary"]
norm = (prof[cats] - prof[cats].min()) / (prof[cats].max() - prof[cats].min()).replace(0, 1)
norm["recency"] = 1 - norm["recency"]       # recent = good
ang = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist(); ang += ang[:1]
fig = plt.figure(figsize=(8, 8)); ax = plt.subplot(111, polar=True)
for i, (n, r) in enumerate(norm.iterrows()):
    v = r.tolist() + r.tolist()[:1]; ax.plot(ang, v, label=n, color=ru.COLORS[i % len(ru.COLORS)], lw=2); ax.fill(ang, v, alpha=0.08, color=ru.COLORS[i % len(ru.COLORS)])
ax.set_xticks(ang[:-1]); ax.set_xticklabels(["Recency (recent=good)", "Frequency", "Monetary"]); ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.1), fontsize=8)
ax.set_title("Segment radar chart")
ru.save_fig(fig, "03_09_segment_radar", "Segment radar chart", "Normalised R, F and M of each K-Means segment.", "Customer Segmentation")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
prof["customers"].sort_values().plot(kind="barh", ax=axes[0], color=ru.PALETTE["purple"]); axes[0].set_title("Customers per segment")
prof["revenue_share_%"].sort_values().plot(kind="barh", ax=axes[1], color=ru.PALETTE["green"]); axes[1].set_title("Revenue share % per segment")
ru.save_fig(fig, "03_10_segment_size_revenue", "Segment size and revenue share", "A few high-value segments contribute most revenue.", "Customer Segmentation")
'''),
C('''
ACTIONS = {
    "Champions": "Reward, early access to new products, ask for reviews/referrals.",
    "Loyal Regulars": "Upsell bundles, loyalty points, keep communication frequent.",
    "Big Spenders": "Premium / bulk offers, personal account manager.",
    "New / Promising": "Welcome series, second-purchase incentive.",
    "Can't Lose Them": "Urgent win-back with personal outreach and special discount.",
    "At-Risk Regulars": "Reminder emails, 'we miss you' offer.",
    "Dormant Big Spenders": "High-value reactivation campaign.",
    "Hibernating / Lost": "Low-cost reactivation or sunset from marketing spend.",
}
def action_for(name):
    base = name.split(" - ")[0]
    return ACTIONS.get(base, "Review manually.")
interp = prof.reset_index()[["segment", "customers", "recency", "frequency", "monetary", "revenue_share_%"]].round(1)
interp["recommended_action"] = interp["segment"].map(action_for)
display(interp)
interp.to_csv(ru.DATA_PROC / "segment_interpretation.csv", index=False)
'''),
M("### Step 6 - DBSCAN (density-based) comparison"),
C('''
labels, eps, kd = ru.fit_dbscan(X)
from sklearn.metrics import silhouette_score
ncl = len(set(labels)) - (1 if -1 in labels else 0); noise = float((labels == -1).mean() * 100)
sil = silhouette_score(X[labels != -1], labels[labels != -1]) if ncl > 1 else float("nan")
print(f"DBSCAN eps={eps:.3f} -> clusters={ncl}, noise={noise:.1f}%, silhouette(excl. noise)={sil:.3f}")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(kd, color=ru.PALETTE["navy"]); axes[0].axhline(eps, color="red", ls="--", label=f"eps={eps:.2f}"); axes[0].legend(); axes[0].set_title("k-distance plot"); axes[0].set_xlabel("points sorted")
sc = axes[1].scatter(P[:, 0], P[:, 1], c=labels, cmap="tab10", s=12); axes[1].set_title(f"DBSCAN clusters (noise = -1) | {ncl} clusters, {noise:.1f}% noise")
ru.save_fig(fig, "03_11_dbscan", "DBSCAN k-distance and clusters", "DBSCAN isolates outlier customers (noise) that K-Means would force into a cluster.", "Customer Segmentation")
cmp = pd.DataFrame({"K-Means": {"clusters": K, "silhouette": float(ks.loc[ks['k'] == K, 'silhouette'].iloc[0]), "noise_%": 0.0},
                    "DBSCAN": {"clusters": ncl, "silhouette": sil, "noise_%": noise}}).T
display(cmp.round(3))
'''),
M("### Step 7 - Cohort retention analysis"),
C('''
coh = ru.cohort_matrix(cs)
fig, ax = plt.subplots(figsize=(13, 8)); sns.heatmap(coh.iloc[:, :13], annot=True, fmt=".0f", cmap="Blues", ax=ax, vmin=0, vmax=60, cbar_kws={"label": "% customers active"})
ax.set_title("Monthly cohort retention (% of cohort buying again)"); ax.set_xlabel("Months since first purchase"); ax.set_ylabel("Cohort")
ru.save_fig(fig, "03_12_cohort_retention", "Cohort retention heatmap", "Shows how quickly each acquisition cohort decays - input for churn definition.", "Customer Segmentation")
'''),
C('''
out = seg_df.reset_index().merge(rfm[["R", "F", "M", "rule_segment"]].reset_index(), on="customer_id", suffixes=("", "_r"))
out[["customer_id", "recency", "frequency", "monetary", "R", "F", "M", "rule_segment", "cluster", "segment"]].to_csv(ru.DATA_PROC / "rfm_segments.csv", index=False)
ru.log_metrics("segmentation", {"customers": int(len(rfm)), "k_chosen": K, "silhouette_kmeans": float(ks.loc[ks['k'] == K, 'silhouette'].iloc[0]),
                                "dbscan_clusters": int(ncl), "dbscan_noise_pct": noise, "segments": prof["customers"].to_dict(),
                                "revenue_share_pct": prof["revenue_share_%"].round(2).to_dict()})
print("Saved rfm_segments.csv")
'''),
M('''
## Summary of Notebook 03
* RFM scoring + **10 rule-based segments**; K-Means with K chosen by silhouette inside 6-8; DBSCAN for outliers; cohort retention.
* Business interpretation + recommended marketing actions per segment.
* **12 charts.** Output `rfm_segments.csv` is used by churn analysis and the Streamlit Customer dashboard.
'''),
])

# ---------------------------------------------------------------- 04 TIME SERIES PREP
STAGES["04_timeseries_preparation"] = dict(
    title="04 - Time-Series Preparation, Stationarity & Decomposition",
    week="Week 1 - Day 4", needs_load=True, cells=[
M('''
## Introduction
Forecasting needs a clean, regular time series. We build the **daily demand series**, check it for **stationarity** (ADF and KPSS tests), **decompose** trend / seasonality / residual, inspect **autocorrelation (ACF / PACF)** and define a leakage-free **train / validation / test** split (30-day horizon, F-03).
'''),
C(LOAD),
C('''
daily = ru.daily_series(sales)
TARGET = "quantity"
print(daily.shape, "| zero-demand days:", int((daily[TARGET] == 0).sum()))
display(daily.tail())
daily.to_csv(ru.DATA_PROC / "daily_series.csv", index=False)
'''),
M("### Step 1 - Daily demand, revenue and orders"),
C('''
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
for ax, c, col in zip(axes, ["quantity", "revenue", "orders"], [ru.PALETTE["navy"], ru.PALETTE["teal"], ru.PALETTE["orange"]]):
    ax.plot(daily["ds"], daily[c], color=col, lw=1); ax.set_ylabel(c)
axes[0].set_title("Daily units, revenue and orders")
ru.save_fig(fig, "04_01_daily_series", "Daily units, revenue and orders", "All three series share the same seasonal rhythm.", "Time-Series Preparation")

dd = daily.copy(); dd["dow"] = dd["ds"].dt.dayofweek
zero = dd.groupby("dow")[TARGET].apply(lambda s: (s == 0).mean() * 100)
fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], zero.reindex(range(7), fill_value=0).values, color=ru.PALETTE["red"])
ax.set_ylabel("% days with zero demand"); ax.set_title("Zero-demand days by weekday (closed days)")
ru.save_fig(fig, "04_02_zero_days", "Zero-demand days by weekday", f"Closed weekdays detected: {ru.closed_dows(daily, TARGET)}. These are forced to zero in forecasts.", "Time-Series Preparation")
'''),
M("### Step 2 - Rolling statistics and seasonality views"),
C('''
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
axes[0].plot(daily["ds"], daily[TARGET], color="#CBD5E1"); axes[0].plot(daily["ds"], daily[TARGET].rolling(28).mean(), color=ru.PALETTE["red"], lw=2); axes[0].set_title("Rolling mean (28d)")
axes[1].plot(daily["ds"], daily[TARGET].rolling(28).std(), color=ru.PALETTE["purple"], lw=2); axes[1].set_title("Rolling std (28d) - variance changes -> non-stationary")
ru.save_fig(fig, "04_03_rolling_stats", "Rolling mean and rolling std", "A moving mean and moving variance indicate non-stationarity.", "Time-Series Preparation")

dd["month"] = dd["ds"].dt.month; dd["year"] = dd["ds"].dt.year
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
sns.boxplot(data=dd, x="dow", y=TARGET, ax=axes[0], palette=ru.COLORS); axes[0].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]); axes[0].set_title("Weekly seasonality")
mm = dd.groupby(["year", "month"])[TARGET].sum().unstack(0)
mm.plot(ax=axes[1], marker="o"); axes[1].set_title("Yearly seasonality (monthly units)"); axes[1].set_xlabel("month")
ru.save_fig(fig, "04_04_seasonality", "Weekly and yearly seasonality", "Clear weekday pattern plus a strong autumn / holiday peak.", "Time-Series Preparation")
'''),
M("### Step 3 - Decomposition (trend / seasonal / residual)"),
C('''
from statsmodels.tsa.seasonal import seasonal_decompose, STL
s = daily.set_index("ds")[TARGET]
dec = seasonal_decompose(s, model="additive", period=7)
fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
for ax, (n, v, col) in zip(axes, [("Observed", dec.observed, "navy"), ("Trend", dec.trend, "red"), ("Weekly seasonal", dec.seasonal, "teal"), ("Residual", dec.resid, "grey")]):
    ax.plot(v, color=ru.PALETTE[col], lw=1.2); ax.set_ylabel(n)
axes[0].set_title("Classical decomposition (period = 7)")
ru.save_fig(fig, "04_05_decomposition", "Classical decomposition (weekly)", "Separates trend, weekly seasonality and noise.", "Time-Series Preparation")

stl = STL(np.log1p(s), period=7, robust=True).fit()
fig = stl.plot(); fig.set_size_inches(13, 9); fig.suptitle("STL decomposition of log1p(demand)", y=1.01)
ru.save_fig(fig, "04_06_stl", "STL decomposition (robust)", "Robust STL handles the zero (closed) days better than classical decomposition.", "Time-Series Preparation")
'''),
M("### Step 4 - Stationarity tests (ADF & KPSS) and differencing"),
C('''
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
def tests(x, name):
    a = adfuller(x.dropna(), autolag="AIC"); k = kpss(x.dropna(), regression="c", nlags="auto")
    return {"series": name, "ADF stat": a[0], "ADF p": a[1], "KPSS stat": k[0], "KPSS p": k[1],
            "verdict": "Stationary" if (a[1] < 0.05 and k[1] >= 0.05) else ("Trend/seasonal non-stationary" if a[1] >= 0.05 else "Mixed - difference recommended")}
res = pd.DataFrame([tests(s, "original"), tests(s.diff(1), "diff(1)"), tests(s.diff(7), "diff(7) seasonal")])
display(res.round(4))
ru.log_metrics("timeseries", {"adf_p_original": float(res.loc[0, "ADF p"]), "adf_p_diff1": float(res.loc[1, "ADF p"]), "adf_p_diff7": float(res.loc[2, "ADF p"])})
fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
for ax, (n, v) in zip(axes, [("Original", s), ("First difference", s.diff(1)), ("Seasonal difference (7)", s.diff(7))]):
    ax.plot(v, lw=0.9, color=ru.PALETTE["navy"]); ax.set_ylabel(n)
axes[0].set_title("Original vs differenced series")
ru.save_fig(fig, "04_07_differencing", "Original vs differenced series", "Differencing removes trend / weekly seasonality and stabilises the mean.", "Time-Series Preparation")
'''),
M("### Step 5 - ACF / PACF and lag plot"),
C('''
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
plot_acf(s, lags=42, ax=axes[0]); plot_pacf(s, lags=42, ax=axes[1], method="ywm")
axes[0].set_title("ACF - spikes every 7 lags = weekly seasonality"); axes[1].set_title("PACF")
ru.save_fig(fig, "04_08_acf_pacf", "ACF and PACF", "Significant lags at 1, 7, 14 justify the lag features (1, 7, 14, 28).", "Time-Series Preparation")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(s.shift(1), s, s=10, alpha=0.5, color=ru.PALETTE["teal"]); axes[0].set_title("y(t) vs y(t-1)")
axes[1].scatter(s.shift(7), s, s=10, alpha=0.5, color=ru.PALETTE["orange"]); axes[1].set_title("y(t) vs y(t-7)")
ru.save_fig(fig, "04_09_lag_plots", "Lag scatter plots", "The y(t) vs y(t-7) relation is tighter than the 1-day lag.", "Time-Series Preparation")
'''),
M("### Step 6 - Outlier days and train / validation / test split"),
C('''
z = (s - s.rolling(28, min_periods=7).mean()) / s.rolling(28, min_periods=7).std()
out_days = s[z.abs() > 3]
fig, ax = plt.subplots(figsize=(13, 4.5)); ax.plot(s, color="#94A3B8", lw=1); ax.scatter(out_days.index, out_days.values, color=ru.PALETTE["red"], zorder=3, label=f"|z|>3 ({len(out_days)} days)")
ax.legend(); ax.set_title("Anomalous demand days (rolling z-score)")
ru.save_fig(fig, "04_10_outlier_days", "Anomalous demand days", "Spikes usually correspond to big wholesale orders; they are kept but flagged.", "Time-Series Preparation")

tr, va, te = ru.split_series(daily)
fig, ax = plt.subplots(figsize=(13, 4.5))
for d_, c, n in [(tr, ru.PALETTE["navy"], "Train"), (va, ru.PALETTE["orange"], "Validation (30d)"), (te, ru.PALETTE["red"], "Test (30d)")]:
    ax.plot(d_["ds"], d_[TARGET], color=c, label=f"{n}: {len(d_)} days")
ax.legend(); ax.set_title("Chronological split (no shuffling -> no leakage)")
ru.save_fig(fig, "04_11_train_val_test", "Train / validation / test split", "Validation tunes ensemble weights; test is touched only once for the final score.", "Time-Series Preparation")
'''),
M('''
## Summary of Notebook 04
* Built a **continuous daily series**, detected **closed weekdays**, decomposed (classical + STL), ran **ADF/KPSS**, ACF/PACF, differencing and outlier detection.
* **11 charts** and a leakage-free **30-day train/validation/test split**.
* **Next -> Notebook 05:** baseline Prophet model.
'''),
])
