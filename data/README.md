# Data folder
* `sample/online_retail_II_demo_raw.csv` – synthetic raw data (UCI Online Retail II schema) used when no real file is available.
* `processed/transactions_clean.csv.gz` – cleaned sales lines. Columns: invoice, **invoice_no**, stock_code, description, category, quantity, unit_price, revenue, **order_value**, invoice_date, customer_id, country.
* `processed/powerbi_ready.csv` – same + year, month, month_name, year_month, dow, hour, week, quarter, day_name.
* `processed/forecast_future.csv`, `forecast_test.csv`, `rfm_segments.csv`, `churn_scores.csv`, `inventory_plan.csv`, `inventory_backtest.csv`, `daily_series.csv`, `rfm_table.csv`, `segment_interpretation.csv`.
Real data is downloaded by the notebooks (kagglehub) or uploaded by you – never commit large raw files.
