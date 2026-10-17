"""
RetailPulse - AI-Powered Customer Analytics & Demand Forecasting Platform
Shared utility library (single file, works in Google Colab + local + Streamlit).

Author : Santosh Kumar
Project: Zidio Development - Data Science & Analytics (March 2026)

Design rules
------------
* Every public function is defensive: optional libraries (Prophet, PyTorch, SHAP,
  MLflow, Evidently) are wrapped so that the notebook NEVER crashes - it falls back
  to a simpler method and prints a clear message.
* Works with 3 data sources: (1) Kaggle via kagglehub, (2) your own uploaded
  CSV / XLSX, (3) built-in synthetic demo data (automatic fallback).
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# 0. CONFIG
# ----------------------------------------------------------------------------
AUTHOR = "Santosh Kumar"
PROJECT = "RetailPulse - AI-Powered Customer Analytics & Demand Forecasting Platform"
SEED = 42

CONFIG = {
    # "auto" | "uci" | "synthetic" | "upload" | "demo"
    "DATASET": "auto",
    "MAX_ROWS": 400_000,          # sampling cap (customer-level) for huge files
    "FORECAST_HORIZON": 30,       # days ahead (F-03)
    "CHURN_WINDOW_DAYS": 90,      # label window for churn
    "TOP_N_SKUS": 40,             # SKUs used in inventory optimisation
    "LEAD_TIME_DAYS": 5,
    "REVIEW_PERIOD_DAYS": 7,
    "SERVICE_LEVEL": 0.95,
    "ORDER_COST": 40.0,
    "HOLDING_RATE": 0.20,
    "PATH": None,                 # e.g. "/content/drive/MyDrive/online_retail_II.xlsx"
}

KAGGLE_HANDLES = {
    "uci": "cgrymn/online-retail-ii-uci-dataset",
    "synthetic": "mrayyanshehzad/synthetic-retail-dataset-10-million-transactions",
}

ROOT = Path(os.environ.get("RETAILPULSE_HOME", "retailpulse_work")).resolve()
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
CHARTS = ROOT / "outputs" / "charts"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
METRICS_FILE = ROOT / "outputs" / "metrics.json"
CHART_INDEX = ROOT / "outputs" / "charts_index.json"

PALETTE = {
    "navy": "#1F3A5F", "teal": "#14B8A6", "orange": "#F59E0B",
    "red": "#EF4444", "purple": "#6366F1", "green": "#22C55E",
    "grey": "#64748B", "pink": "#EC4899", "sky": "#0EA5E9",
}
COLORS = list(PALETTE.values())

_CACHE: dict = {}


def setup(root: str | None = None, show: bool = True):
    """Create the workspace folders and apply the plotting theme."""
    global ROOT, DATA_RAW, DATA_PROC, CHARTS, MODELS, REPORTS, METRICS_FILE, CHART_INDEX
    if root:
        ROOT = Path(root).resolve()
        DATA_RAW, DATA_PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
        CHARTS, MODELS, REPORTS = ROOT / "outputs" / "charts", ROOT / "models", ROOT / "reports"
        METRICS_FILE, CHART_INDEX = ROOT / "outputs" / "metrics.json", ROOT / "outputs" / "charts_index.json"
    for d in (DATA_RAW, DATA_PROC, CHARTS, MODELS, REPORTS):
        d.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette=COLORS, font_scale=1.0)
    plt.rcParams.update({
        "figure.figsize": (10, 5), "figure.dpi": 100, "savefig.dpi": 130,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=COLORS),
    })
    if show:
        print(f"RetailPulse workspace ready -> {ROOT}")


setup(show=False)


def ensure_packages(pkgs: dict[str, str] | None = None):
    """Install only the packages that are really missing (avoids Colab restarts)."""
    pkgs = pkgs or {
        "openpyxl": "openpyxl", "kagglehub": "kagglehub", "xgboost": "xgboost",
        "shap": "shap", "optuna": "optuna", "mlflow": "mlflow",
        "statsmodels": "statsmodels", "prophet": "prophet",
    }
    missing = [pip for mod, pip in pkgs.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print("Installing missing packages:", missing)
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=False)
    else:
        print("All required packages already available.")


# ----------------------------------------------------------------------------
# 1. LOGGING HELPERS (metrics + chart index feed the final DOCX report)
# ----------------------------------------------------------------------------
def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _json_safe(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


def log_metrics(section: str, values: dict):
    """Merge a dict of results into outputs/metrics.json (used by the report)."""
    data = _read_json(METRICS_FILE, {})
    data.setdefault(section, {}).update(values)
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(json.dumps(data, indent=2, default=_json_safe))


def get_metrics() -> dict:
    return _read_json(METRICS_FILE, {})


def save_fig(fig, fname: str, title: str = "", note: str = "", section: str = "General", show: bool = True):
    """Save a matplotlib figure to outputs/charts and register it for the report."""
    CHARTS.mkdir(parents=True, exist_ok=True)
    path = CHARTS / f"{fname}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    idx = _read_json(CHART_INDEX, {})
    idx[fname] = {"title": title or fname, "note": note, "section": section, "file": str(path)}
    CHART_INDEX.write_text(json.dumps(idx, indent=2, default=_json_safe))
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fmt_money(x):
    return f"{x:,.0f}"


# ----------------------------------------------------------------------------
# 2. SYNTHETIC DEMO DATA (automatic fallback - mimics UCI Online Retail II)
# ----------------------------------------------------------------------------
_ADJ = ["VINTAGE", "RETRO", "CLASSIC", "MODERN", "PINK", "BLUE", "RED", "GREEN", "WHITE", "GOLD",
        "SILVER", "FLORAL", "STRIPED", "SPOTTED", "HANDMADE", "LARGE", "SMALL", "MINI", "JUMBO", "LUXURY"]
_ITEMS = {
    "Kitchen": ["MUG", "TEA POT", "CAKE TIN", "LUNCH BOX", "JAM JAR", "SPOON SET", "BOWL", "APRON", "CUP", "PLATE"],
    "Home Decor": ["CANDLE HOLDER", "PHOTO FRAME", "CUSHION COVER", "LANTERN", "WALL CLOCK", "VASE", "DOORMAT", "MIRROR"],
    "Stationery": ["NOTEBOOK", "PENCIL SET", "GREETING CARD", "WRAPPING PAPER", "STICKER SET", "DIARY"],
    "Gifts & Toys": ["TEDDY BEAR", "GAME", "PUZZLE", "DOLL", "TOY CAR", "GIFT BOX", "MUSIC BOX"],
    "Garden": ["PLANT POT", "BIRD FEEDER", "GARDEN TROWEL", "WATERING CAN", "SEED TIN"],
    "Bags & Accessories": ["SHOPPER BAG", "PURSE", "UMBRELLA", "WASH BAG", "KEY RING", "SCARF"],
}
_COUNTRIES = ["United Kingdom", "Germany", "France", "EIRE", "Netherlands", "Spain", "Belgium",
              "Switzerland", "Portugal", "Australia", "Norway", "Italy"]
_COUNTRY_P = np.array([0.88, 0.025, 0.022, 0.02, 0.008, 0.006, 0.005, 0.004, 0.004, 0.0035, 0.0015, 0.001])
_COUNTRY_P = _COUNTRY_P / _COUNTRY_P.sum()


def generate_demo_data(n_customers: int = 1900, start: str = "2010-01-04", end: str = "2011-12-09",
                       seed: int = SEED) -> pd.DataFrame:
    """Synthetic UCI-style transactions: 8 columns exactly like Online Retail II."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    n_days = len(days)

    # product catalogue
    products = []
    code = 20000
    for cat, items in _ITEMS.items():
        for it in items:
            for adj in rng.choice(_ADJ, size=4, replace=False):
                code += int(rng.integers(1, 9))
                price = float(np.round(rng.lognormal(mean=0.9, sigma=0.55), 2))
                products.append((str(code), f"{adj} {it}", max(price, 0.29)))
    prod = pd.DataFrame(products, columns=["stock_code", "description", "price"])
    n_prod = len(prod)
    pop = 1.0 / np.arange(1, n_prod + 1) ** 0.9
    rng.shuffle(pop)
    pop = pop / pop.sum()

    # seasonality: Nov peak, Jan trough; weekday: no Saturday, Sunday lighter
    month_mult = {1: .7, 2: .75, 3: .9, 4: .85, 5: .95, 6: .95, 7: .95, 8: .95, 9: 1.15, 10: 1.3, 11: 1.8, 12: 1.2}
    dow_mult = {0: 1.0, 1: 1.15, 2: 1.1, 3: 1.2, 4: 0.9, 5: 0.0, 6: 0.55}
    trend = np.linspace(1.0, 1.3, n_days)
    day_w = np.array([month_mult[d.month] * dow_mult[d.dayofweek] for d in days]) * trend
    day_w = day_w / day_w.sum()

    rows = []
    inv_no = 530000

    def make_order(cust_id, day, country):
        nonlocal inv_no
        inv_no += 1
        n_lines = 1 + rng.geometric(1 / 5.0)
        n_lines = min(n_lines, 40)
        pids = rng.choice(n_prod, size=n_lines, replace=False, p=pop)
        hour = int(rng.choice(np.arange(8, 19), p=np.array([.04, .08, .13, .14, .10, .10, .12, .11, .09, .06, .03]) / 1.0))
        ts = pd.Timestamp(day) + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
        for pid in pids:
            sc, desc, pr = prod.iloc[pid]
            qty = int(np.clip(np.round(rng.lognormal(1.3, 0.9)), 1, 120))
            if rng.random() < 0.12:
                qty = int(rng.choice([6, 12, 24, 48]))
            price = float(np.round(pr * rng.choice([1.0, 1.0, 1.0, 0.95, 1.05]), 2))
            rows.append((str(inv_no), sc, desc, qty, ts, price, cust_id, country))
        # cancellation
        if rng.random() < 0.02:
            inv_no += 1
            sc, desc, pr = prod.iloc[rng.choice(n_prod, p=pop)]
            rows.append(("C" + str(inv_no), sc, desc, -int(rng.integers(1, 12)), ts + pd.Timedelta(days=int(rng.integers(1, 9))), pr, cust_id, country))

    for c in range(n_customers):
        cid = 12000 + c
        country = rng.choice(_COUNTRIES, p=_COUNTRY_P)
        first_idx = int(rng.beta(1.2, 2.2) * n_days * 0.92)
        lifetime = int(rng.exponential(520)) + 45
        last_idx = min(n_days - 1, first_idx + lifetime)
        rate = rng.gamma(1.3, 0.017)         # orders / day
        n_orders = max(1, rng.poisson(rate * (last_idx - first_idx + 1)))
        w = day_w[first_idx:last_idx + 1]
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        chosen = rng.choice(np.arange(first_idx, last_idx + 1), size=n_orders, p=w)
        for di in chosen:
            make_order(float(cid), days[di], country)

    # guest orders (no customer id)
    n_guest = int(n_days * 3.0)
    for di in rng.choice(np.arange(n_days), size=n_guest, p=day_w):
        make_order(np.nan, days[di], "United Kingdom")

    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"])

    # realistic dirt
    dup = df.sample(frac=0.003, random_state=seed)
    zero = df.sample(frac=0.001, random_state=seed + 1).copy()
    zero["Price"] = 0.0
    fee = df.sample(n=40, random_state=seed + 2).copy()
    fee["StockCode"], fee["Description"], fee["Quantity"], fee["Price"] = "POST", "POSTAGE", 1, 18.0
    big = df.sample(n=5, random_state=seed + 3).copy()
    big["Quantity"] = rng.integers(3000, 9000, size=5)
    df = pd.concat([df, dup, zero, fee, big], ignore_index=True)
    miss = df.sample(frac=0.004, random_state=seed + 4).index
    df.loc[miss, "Description"] = np.nan
    df = df.sort_values("InvoiceDate").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# 3. DATA LOADING (Kaggle / upload / demo) + column standardisation
# ----------------------------------------------------------------------------
_COLMAP = {
    "invoice": ["invoice", "invoiceno", "invoiceid", "orderid", "transactionid", "orderno", "billno"],
    "stock_code": ["stockcode", "productid", "sku", "itemcode", "productcode", "itemid"],
    "description": ["description", "productname", "itemname", "product", "item"],
    "quantity": ["quantity", "qty", "units", "unitssold", "quantitysold"],
    "invoice_date": ["invoicedate", "orderdate", "transactiondate", "date", "timestamp", "datetime", "purchasedate"],
    "unit_price": ["price", "unitprice", "sellingprice", "priceperunit", "unitcost"],
    "customer_id": ["customerid", "custid", "userid", "clientid", "customer"],
    "country": ["country", "region", "location", "storelocation", "city", "state"],
    "category": ["category", "productcategory", "department"],
    "total": ["totalamount", "totalprice", "total", "revenue", "sales", "amount", "linetotal"],
}


def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Map any retail-like schema to the 8 canonical columns. Raises ValueError if impossible."""
    norm = {_norm(c): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for target, cands in _COLMAP.items():
        for cand in cands:
            if cand in norm:
                out[target] = df[norm[cand]]
                break
    if "invoice_date" not in out:
        raise ValueError("No date column found")
    out["invoice_date"] = pd.to_datetime(out["invoice_date"], errors="coerce")
    if out["invoice_date"].notna().mean() < 0.8:
        raise ValueError("Date column could not be parsed")
    if "quantity" not in out:
        out["quantity"] = 1
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    if "unit_price" not in out:
        if "total" in out:
            out["unit_price"] = pd.to_numeric(out["total"], errors="coerce") / out["quantity"].replace(0, np.nan)
        else:
            raise ValueError("No price / amount column found")
    out["unit_price"] = pd.to_numeric(out["unit_price"], errors="coerce")
    if "invoice" not in out:
        key = out["customer_id"].astype(str) if "customer_id" in out else pd.Series("na", index=out.index)
        out["invoice"] = key + "_" + out["invoice_date"].dt.strftime("%Y%m%d%H")
    if "stock_code" not in out:
        out["stock_code"] = out["description"] if "description" in out else "UNKNOWN"
    if "description" not in out:
        out["description"] = out["stock_code"]
    if "customer_id" not in out:
        out["customer_id"] = np.nan
    if "country" not in out:
        out["country"] = "Unknown"
    cols = ["invoice", "stock_code", "description", "quantity", "invoice_date", "unit_price", "customer_id", "country"]
    if "category" in out:
        cols.append("category")
    out = out[cols].copy()
    out["invoice"] = out["invoice"].astype(str)
    out["stock_code"] = out["stock_code"].astype(str)
    return out


def _read_any(path: Path, max_rows: int) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)
        return pd.concat(sheets.values(), ignore_index=True)
    if suffix in (".csv", ".gz", ".txt"):
        try:
            head = pd.read_csv(path, nrows=2000, encoding="utf-8")
            enc = "utf-8"
        except Exception:
            enc = "ISO-8859-1"
            head = pd.read_csv(path, nrows=2000, encoding=enc)
        size = path.stat().st_size
        est_rows = size / max(len(head.to_csv(index=False)) / len(head), 1)
        if est_rows <= max_rows * 1.2:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        # big file: customer-level hash sampling keeps whole customer histories
        keep_every = int(math.ceil(est_rows / max_rows))
        print(f"Large file (~{est_rows:,.0f} rows). Sampling 1/{keep_every} of customers for Colab memory safety ...")
        chunks = []
        for chunk in pd.read_csv(path, encoding=enc, chunksize=500_000, low_memory=False):
            try:
                std = standardize(chunk)
                if std["customer_id"].notna().mean() > 0.5:
                    cid = pd.to_numeric(std["customer_id"], errors="coerce").fillna(0).astype("int64")
                    chunk = chunk.loc[(cid % keep_every == 0).values]
                else:
                    chunk = chunk.sample(frac=1 / keep_every, random_state=SEED)
            except Exception:
                chunk = chunk.sample(frac=1 / keep_every, random_state=SEED)
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)
    raise ValueError(f"Unsupported file type: {path.name}")


def _candidate_files() -> list[Path]:
    dirs = [DATA_RAW, Path.cwd(), Path("/content"), Path.cwd() / "data", Path.cwd() / "data" / "raw",
            Path.cwd().parent / "data" / "sample", Path.cwd() / "data" / "sample"]
    found, seen = [], set()
    for d in dirs:
        if d.exists() and d.is_dir():
            for p in sorted(d.iterdir()):
                if p.is_file() and p.suffix.lower() in (".csv", ".xlsx", ".xls") and p not in seen:
                    seen.add(p)
                    found.append(p)
    return found


def _kaggle_files(handle: str) -> list[Path]:
    import kagglehub
    path = Path(kagglehub.dataset_download(handle))
    files = [p for p in path.rglob("*") if p.suffix.lower() in (".csv", ".xlsx", ".xls")]
    return sorted(files, key=lambda p: -p.stat().st_size)


def load_raw(dataset: str | None = None, path: str | None = None, max_rows: int | None = None, verbose: bool = True):
    """
    Load the raw dataset. Returns (df_standardised, source_label).
    Order for dataset="auto": uploaded/local file -> Kaggle UCI Online Retail II -> Kaggle synthetic -> demo.
    """
    dataset = dataset or CONFIG["DATASET"]
    max_rows = max_rows or CONFIG["MAX_ROWS"]

    def _try(p: Path, label: str):
        try:
            raw = _read_any(p, max_rows)
            std = standardize(raw)
            if verbose:
                print(f"[OK] Loaded {label}: {p.name} -> {len(std):,} rows")
            return std, f"{label} ({p.name})"
        except Exception as e:  # noqa
            if verbose:
                print(f"[skip] {p.name}: {e}")
            return None

    if path:
        r = _try(Path(path), "User file")
        if r:
            return r

    if dataset in ("auto", "upload"):
        for p in _candidate_files():
            if p.name.startswith("retailpulse_"):
                continue
            r = _try(p, "Uploaded/local file")
            if r:
                return r

    if dataset in ("auto", "uci", "synthetic") and importlib.util.find_spec("kagglehub") is not None:
        order = ["uci", "synthetic"] if dataset in ("auto", "uci") else ["synthetic"]
        for key in order:
            try:
                if verbose:
                    print(f"Downloading from Kaggle: {KAGGLE_HANDLES[key]} ...")
                for p in _kaggle_files(KAGGLE_HANDLES[key]):
                    r = _try(p, f"Kaggle {key}")
                    if r:
                        return r
            except Exception as e:  # noqa
                if verbose:
                    print(f"[skip] Kaggle {key} failed ({type(e).__name__}: {str(e)[:90]})")

    if verbose:
        print("[INFO] Real dataset not available -> using built-in synthetic demo data (same schema as UCI Online Retail II).")
    return standardize(generate_demo_data()), "Synthetic demo data"


# ----------------------------------------------------------------------------
# 4. CLEANING + FEATURE ENGINEERING
# ----------------------------------------------------------------------------
_NON_PRODUCT = {"POST", "D", "M", "DOT", "CRUK", "C2", "BANK CHARGES", "AMAZONFEE", "ADJUST", "ADJUST2", "S", "B",
                "PADS", "TEST001", "TEST002", "GIFT", "SAMPLES", "CARRIAGE", "DISCOUNT"}

_CATEGORY_RULES = [
    ("Kitchen", ["MUG", "TEA", "CAKE", "LUNCH", "JAM", "SPOON", "BOWL", "APRON", "CUP", "PLATE", "KITCHEN", "BAKING", "JUG", "TIN"]),
    ("Home Decor", ["CANDLE", "FRAME", "CUSHION", "LANTERN", "CLOCK", "VASE", "DOORMAT", "MIRROR", "HEART", "DECORATION", "LIGHT", "HOLDER"]),
    ("Stationery", ["NOTEBOOK", "PENCIL", "CARD", "WRAP", "STICKER", "DIARY", "PAPER", "PEN", "BOOK"]),
    ("Gifts & Toys", ["BEAR", "GAME", "PUZZLE", "DOLL", "TOY", "GIFT", "MUSIC", "CHRISTMAS", "BOX"]),
    ("Garden", ["PLANT", "BIRD", "GARDEN", "WATERING", "SEED", "FLOWER"]),
    ("Bags & Accessories", ["BAG", "PURSE", "UMBRELLA", "WASH", "KEY RING", "SCARF", "SHOPPER"]),
]


def infer_category(desc: pd.Series) -> pd.Series:
    d = desc.fillna("").astype(str).str.upper()
    out = pd.Series("Other", index=desc.index, dtype=object)
    for cat, words in reversed(_CATEGORY_RULES):          # first rule wins
        mask = d.str.contains("|".join(re.escape(w) for w in words), regex=True)
        out[mask] = cat
    return out


def clean(df_std: pd.DataFrame, verbose: bool = True):
    """
    Returns (sales, returns, report).
      sales   - valid positive sales lines with revenue + calendar features
      returns - cancellations / negative quantity lines
      report  - dict of rows removed at each step (for the waterfall chart)
    """
    rep = {"raw_rows": len(df_std)}
    df = df_std.copy()
    df = df.dropna(subset=["invoice_date"])
    rep["after_drop_bad_dates"] = len(df)

    before = len(df)
    df = df.drop_duplicates(subset=["invoice", "stock_code", "quantity", "invoice_date", "unit_price", "customer_id"])
    rep["duplicates_removed"] = before - len(df)

    df["is_cancel"] = df["invoice"].astype(str).str.upper().str.startswith("C") | (df["quantity"] < 0)
    returns = df[df["is_cancel"]].copy()
    returns["quantity"] = returns["quantity"].abs()
    rep["returns_separated"] = len(returns)
    df = df[~df["is_cancel"]].copy()

    before = len(df)
    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]
    rep["invalid_qty_or_price_removed"] = before - len(df)

    before = len(df)
    df = df[~df["stock_code"].astype(str).str.upper().isin(_NON_PRODUCT)]
    rep["non_product_codes_removed"] = before - len(df)

    df["description"] = df["description"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    df["revenue"] = df["quantity"] * df["unit_price"]
    # NEW COLUMN 1: invoice_no -> numeric part of the invoice id (clean key for distinct-order counts / joins)
    df["invoice_no"] = pd.to_numeric(df["invoice"].astype(str).str.extract(r"(\d+)")[0], errors="coerce").astype("Int64")
    # NEW COLUMN 2: order_value -> total revenue of the whole invoice (gives AOV / basket value without re-grouping)
    df["order_value"] = df.groupby("invoice")["revenue"].transform("sum")

    # extreme outliers (beyond 99.95th percentile of line revenue / quantity)
    before = len(df)
    q_hi, r_hi = df["quantity"].quantile(0.9995), df["revenue"].quantile(0.9995)
    df = df[(df["quantity"] <= max(q_hi, 1)) & (df["revenue"] <= max(r_hi, 1))]
    rep["extreme_outliers_removed"] = before - len(df)

    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce")
    rep["rows_without_customer_id"] = int(df["customer_id"].isna().sum())

    if "category" not in df.columns:
        df["category"] = infer_category(df["description"])
    df["category"] = df["category"].fillna("Other")

    d = df["invoice_date"]
    df["date"] = d.dt.normalize()
    df["year"], df["month"], df["day"] = d.dt.year, d.dt.month, d.dt.day
    df["dow"], df["hour"] = d.dt.dayofweek, d.dt.hour
    df["week"] = d.dt.isocalendar().week.astype(int)
    df["quarter"] = d.dt.quarter
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["year_month"] = d.dt.to_period("M").astype(str)
    df["month_name"] = d.dt.strftime("%b")
    df = df.sort_values("invoice_date").reset_index(drop=True)
    rep["final_rows"] = len(df)
    if verbose:
        for k, v in rep.items():
            print(f"{k:<32}: {v:>12,}")
    return df, returns.reset_index(drop=True), rep


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Great-Expectations-style checks implemented natively (no API-breaking dependency)."""
    checks = [
        ("No missing invoice_date", df["invoice_date"].notna().all()),
        ("quantity > 0 for all rows", (df["quantity"] > 0).all()),
        ("unit_price > 0 for all rows", (df["unit_price"] > 0).all()),
        ("revenue == quantity * unit_price", np.allclose(df["revenue"], df["quantity"] * df["unit_price"])),
        ("invoice_date is sorted ascending", df["invoice_date"].is_monotonic_increasing),
        ("No duplicate full rows", not df.duplicated(["invoice", "stock_code", "quantity", "invoice_date", "unit_price", "customer_id"]).any()),
        ("customer_id missing < 40%", df["customer_id"].isna().mean() < 0.40),
        ("country has no nulls", df["country"].notna().all()),
        ("unit_price 99.9th pct < 1000", df["unit_price"].quantile(0.999) < 1000),
        ("Date range covers > 180 days", (df["date"].max() - df["date"].min()).days > 180),
    ]
    out = pd.DataFrame(checks, columns=["expectation", "passed"])
    out["passed"] = out["passed"].astype(bool)
    return out


def _cached(name: str, fn):
    if name not in _CACHE:
        _CACHE[name] = fn()
    return _CACHE[name]


def get_data(force: bool = False, verbose: bool = True):
    """Load + clean once per session; also stored on disk (pickle) for fast re-use between notebooks."""
    pk = DATA_PROC / "retailpulse_clean.pkl"
    if not force and "data" in _CACHE:
        return _CACHE["data"]
    if not force and pk.exists():
        try:
            _CACHE["data"] = pickle.load(open(pk, "rb"))
            return _CACHE["data"]
        except Exception:
            pass
    raw, source = load_raw(path=CONFIG.get("PATH"), verbose=verbose)
    sales, returns, rep = clean(raw, verbose=False)
    if verbose:
        print(f"Source: {source} | raw rows {rep['raw_rows']:,} -> clean rows {rep['final_rows']:,}")
    bundle = {"raw": raw, "sales": sales, "returns": returns, "report": rep, "source": source}
    pickle.dump(bundle, open(pk, "wb"))
    _CACHE["data"] = bundle
    log_metrics("data", {"source": source, **{k: int(v) for k, v in rep.items()},
                         "date_min": str(sales["date"].min().date()), "date_max": str(sales["date"].max().date()),
                         "customers": int(sales["customer_id"].nunique()), "products": int(sales["stock_code"].nunique()),
                         "countries": int(sales["country"].nunique()), "total_revenue": float(sales["revenue"].sum()),
                         "orders": int(sales["invoice"].nunique())})
    return bundle


def get_sales():
    return get_data()["sales"]


def get_customer_sales():
    """Sales with a customer id; if the dataset has none, falls back to demo data (with a warning)."""
    s = get_sales()
    sc = s[s["customer_id"].notna()]
    if sc["customer_id"].nunique() >= 100:
        return sc
    print("[WARN] Dataset has too few customer IDs for customer analytics -> using synthetic demo customers for this section.")
    raw = standardize(generate_demo_data())
    demo, _, _ = clean(raw, verbose=False)
    return demo[demo["customer_id"].notna()]


# ----------------------------------------------------------------------------
# 5. CUSTOMER ANALYTICS: RFM, K-Means, DBSCAN, cohorts
# ----------------------------------------------------------------------------
def rfm_table(df: pd.DataFrame, snapshot=None) -> pd.DataFrame:
    snapshot = pd.Timestamp(snapshot) if snapshot is not None else df["invoice_date"].max().normalize() + pd.Timedelta(days=1)
    g = df.groupby("customer_id")
    t = pd.DataFrame({
        "recency": (snapshot - g["invoice_date"].max()).dt.days,
        "frequency": g["invoice"].nunique(),
        "monetary": g["revenue"].sum(),
        "n_products": g["stock_code"].nunique(),
        "first_purchase": g["invoice_date"].min(),
        "last_purchase": g["invoice_date"].max(),
    })
    t["avg_order_value"] = t["monetary"] / t["frequency"]
    t["tenure_days"] = (snapshot - t["first_purchase"]).dt.days
    return t


def rfm_scores(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["R"] = pd.qcut(t["recency"].rank(method="first", ascending=False), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    t["F"] = pd.qcut(t["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    t["M"] = pd.qcut(t["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    t["RFM_score"] = t["R"] * 100 + t["F"] * 10 + t["M"]
    t["RFM_total"] = t["R"] + t["F"] + t["M"]
    r, fm = t["R"], (t["F"] + t["M"]) / 2
    conds = [
        (r >= 4) & (fm >= 4.5), (r == 1) & (fm >= 4), (r <= 2) & (fm >= 3), (r >= 3) & (fm >= 3.5),
        (r >= 4) & (fm >= 2.5), (r >= 4), (r == 3) & (fm >= 2.5), (r == 3), (r == 2), (r == 1) & (fm <= 2),
    ]
    names = ["Champions", "Can't Lose Them", "At Risk", "Loyal Customers", "Potential Loyalists",
             "New Customers", "Need Attention", "Promising", "About To Sleep", "Lost"]
    t["rule_segment"] = np.select(conds, names, default="Hibernating")
    return t


def kmeans_search(rfm: pd.DataFrame, ks=range(2, 11)):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(np.log1p(rfm[["recency", "frequency", "monetary"]]))
    rows = []
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(len(X), 4000), replace=False)
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X[idx], km.labels_[idx]),
                     "davies_bouldin": davies_bouldin_score(X, km.labels_)})
    return pd.DataFrame(rows), X


def fit_kmeans(rfm: pd.DataFrame, X: np.ndarray, k: int):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
    out = rfm.copy()
    out["cluster"] = km.labels_
    prof = out.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
    med = out[["recency", "frequency", "monetary"]].median()
    names = {}
    table = {
        (1, 1, 1): "Champions", (1, 1, 0): "Loyal Regulars", (1, 0, 1): "Big Spenders", (1, 0, 0): "New / Promising",
        (0, 1, 1): "Can't Lose Them", (0, 1, 0): "At-Risk Regulars", (0, 0, 1): "Dormant Big Spenders", (0, 0, 0): "Hibernating / Lost",
    }
    base_of = {}
    for c, r in prof.iterrows():
        key = (int(r["recency"] < med["recency"]), int(r["frequency"] >= med["frequency"]), int(r["monetary"] >= med["monetary"]))
        base_of[c] = table[key]
    descr = {2: ["Top Value", "Entry Value"], 3: ["Top Value", "Mid Value", "Entry Value"],
             4: ["Top Value", "Upper-Mid Value", "Lower-Mid Value", "Entry Value"]}
    for base in set(base_of.values()):
        members = [c for c, b in base_of.items() if b == base]
        if len(members) == 1:
            names[members[0]] = base
        else:
            members = sorted(members, key=lambda c: -prof.loc[c, "monetary"])
            labels = descr.get(len(members), [f"Group {i + 1}" for i in range(len(members))])
            for c, lab in zip(members, labels):
                names[c] = f"{base} - {lab}"
    out["segment"] = out["cluster"].map(names)
    return out, km, names


def fit_dbscan(X: np.ndarray, min_samples: int = 6, pct: float = 92):
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    kdist = np.sort(nn.kneighbors(X)[0][:, -1])
    eps = float(np.percentile(kdist, pct))
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    return labels, eps, kdist


def cohort_matrix(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["customer_id", "invoice_date"]].copy()
    d["order_month"] = d["invoice_date"].dt.to_period("M")
    d["cohort"] = d.groupby("customer_id")["invoice_date"].transform("min").dt.to_period("M")
    d["offset"] = (d["order_month"].dt.year - d["cohort"].dt.year) * 12 + (d["order_month"].dt.month - d["cohort"].dt.month)
    m = d.groupby(["cohort", "offset"])["customer_id"].nunique().unstack(fill_value=0)
    ret = m.div(m[0], axis=0) * 100
    last = d["order_month"].max()
    for coh in ret.index:                     # months that have not happened yet -> NaN (not 0)
        max_off = (last.year - coh.year) * 12 + (last.month - coh.month)
        ret.loc[coh, [c for c in ret.columns if c > max_off]] = np.nan
    ret.index = ret.index.astype(str)
    return ret.round(1)


# ----------------------------------------------------------------------------
# 6. TIME SERIES + FORECASTING (Prophet, LSTM, Hybrid)
# ----------------------------------------------------------------------------
def daily_series(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("date").agg(quantity=("quantity", "sum"), revenue=("revenue", "sum"),
                               orders=("invoice", "nunique"), customers=("customer_id", "nunique"))
    idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
    g = g.reindex(idx).fillna(0.0)
    g.index.name = "ds"
    return g.reset_index()


def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y > 0
    return float(np.mean(np.abs(y[m] - p[m]) / y[m]) * 100) if m.any() else float("nan")


def wape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / max(np.abs(y).sum(), 1e-9) * 100)


def smape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    den = (np.abs(y) + np.abs(p)) / 2
    m = den > 0
    return float(np.mean(np.abs(y[m] - p[m]) / den[m]) * 100)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def block_mape(y, p, block=7):
    """MAPE of non-overlapping 7-day totals (weekly demand planning accuracy)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    n = (len(y) // block) * block
    if n == 0:
        return float("nan")
    ys, ps = y[:n].reshape(-1, block).sum(1), p[:n].reshape(-1, block).sum(1)
    return mape(ys, ps)


def forecast_metrics(y, p) -> dict:
    return {"MAPE_daily_%": mape(y, p), "MAPE_weekly_%": block_mape(y, p), "WAPE_%": wape(y, p),
            "sMAPE_%": smape(y, p), "RMSE": rmse(y, p), "MAE": mae(y, p), "Bias": float(np.mean(np.asarray(p) - np.asarray(y)))}


def split_series(ds: pd.DataFrame, horizon: int | None = None):
    h = horizon or CONFIG["FORECAST_HORIZON"]
    n = len(ds)
    return ds.iloc[: n - 2 * h].copy(), ds.iloc[n - 2 * h: n - h].copy(), ds.iloc[n - h:].copy()


def closed_dows(train: pd.DataFrame, target: str) -> list[int]:
    t = train.copy()
    t["dow"] = pd.to_datetime(t["ds"]).dt.dayofweek
    share_zero = t.groupby("dow")[target].apply(lambda s: (s <= 0).mean())
    return [int(d) for d, v in share_zero.items() if v >= 0.9]


def _apply_closed(pred, dates, closed):
    pred = np.clip(np.asarray(pred, float), 0, None)
    dow = pd.to_datetime(pd.Series(dates)).dt.dayofweek.values
    pred[np.isin(dow, closed)] = 0.0
    return pred


def naive_forecast(train: pd.DataFrame, future_dates, target: str = "quantity", weeks: int = 4):
    t = train.copy()
    t["dow"] = pd.to_datetime(t["ds"]).dt.dayofweek
    recent = t.tail(7 * weeks)
    means = recent.groupby("dow")[target].mean()
    dow = pd.to_datetime(pd.Series(future_dates)).dt.dayofweek
    return dow.map(means).fillna(recent[target].mean()).values.astype(float)


def holt_winters_forecast(train: pd.DataFrame, future_dates, target: str = "quantity"):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y = train[target].astype(float).values
    try:
        fit = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7).fit()
        pred = fit.forecast(len(future_dates))
    except Exception:
        pred = naive_forecast(train, future_dates, target)
    return _apply_closed(pred, future_dates, closed_dows(train, target))


def prophet_forecast(train: pd.DataFrame, future_dates, target: str = "quantity",
                     changepoint_prior_scale: float = 0.05, seasonality_prior_scale: float = 10.0,
                     seasonality_mode: str = "multiplicative", return_model: bool = False):
    """Prophet forecast. Falls back to Holt-Winters if Prophet is unavailable. Returns dict."""
    closed = closed_dows(train, target)
    try:
        import logging
        for name in ("cmdstanpy", "prophet", "stan"):
            logging.getLogger(name).setLevel(logging.ERROR)
        from prophet import Prophet
        m = Prophet(yearly_seasonality=len(train) >= 540, weekly_seasonality=True, daily_seasonality=False,
                    seasonality_mode=seasonality_mode, changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale, interval_width=0.8)
        try:
            m.add_country_holidays(country_name="GB")
        except Exception:
            pass
        m.fit(train[["ds", target]].rename(columns={target: "y"}))
        fut = pd.DataFrame({"ds": pd.to_datetime(list(future_dates))})
        fc = m.predict(fut)
        out = {"pred": _apply_closed(fc["yhat"].values, fut["ds"], closed),
               "lower": _apply_closed(fc["yhat_lower"].values, fut["ds"], closed),
               "upper": _apply_closed(fc["yhat_upper"].values, fut["ds"], closed),
               "engine": "Prophet"}
        if return_model:
            out["model"], out["forecast_df"] = m, fc
        return out
    except Exception as e:  # noqa
        print(f"[INFO] Prophet unavailable ({type(e).__name__}); using Holt-Winters fallback.")
        p = holt_winters_forecast(train, future_dates, target)
        sd = float(np.std(train[target].tail(60)))
        return {"pred": p, "lower": np.clip(p - 1.28 * sd, 0, None), "upper": p + 1.28 * sd, "engine": "Holt-Winters (fallback)"}


def _calendar_feats(dates, closed):
    d = pd.to_datetime(pd.Series(list(dates)))
    dow, mon = d.dt.dayofweek.values, d.dt.month.values
    return np.column_stack([np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                            np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12),
                            np.isin(dow, closed).astype(float)])


def lstm_forecast(train: pd.DataFrame, future_dates, target: str = "quantity", window: int = 28,
                  epochs: int = 60, hidden: int = 32, lr: float = 0.01, seed: int = SEED, return_history: bool = False):
    """PyTorch LSTM with known-future calendar covariates, recursive multi-step forecast.
    Falls back to a gradient-boosting lag model when PyTorch is not installed."""
    closed = closed_dows(train, target)
    y = np.log1p(train[target].astype(float).values)
    mu, sd = y.mean(), y.std() + 1e-9
    ys = (y - mu) / sd
    cal = _calendar_feats(train["ds"], closed)
    try:
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        np.random.seed(seed)
        feats = np.column_stack([ys, cal])
        Xw, yw, cw = [], [], []
        for t in range(window, len(ys)):
            Xw.append(feats[t - window:t])
            yw.append(ys[t])
            cw.append(cal[t])
        Xw, yw, cw = (torch.tensor(np.array(a), dtype=torch.float32) for a in (Xw, yw, cw))
        n_val = max(int(len(Xw) * 0.12), 20)
        tr_idx, va_idx = slice(0, len(Xw) - n_val), slice(len(Xw) - n_val, len(Xw))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(6, hidden, batch_first=True)
                self.head = nn.Sequential(nn.Linear(hidden + 5, 16), nn.ReLU(), nn.Linear(16, 1))

            def forward(self, x, c):
                o, _ = self.lstm(x)
                return self.head(torch.cat([o[:, -1, :], c], dim=1)).squeeze(-1)

        net = Net()
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        hist, best, best_state, wait = [], 1e9, None, 0
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(Xw[tr_idx]))
            tl = 0.0
            for i in range(0, len(perm), 32):
                b = perm[i:i + 32]
                opt.zero_grad()
                loss = loss_fn(net(Xw[tr_idx][b], cw[tr_idx][b]), yw[tr_idx][b])
                loss.backward()
                opt.step()
                tl += loss.item() * len(b)
            net.eval()
            with torch.no_grad():
                vl = loss_fn(net(Xw[va_idx], cw[va_idx]), yw[va_idx]).item()
            hist.append((tl / len(perm), vl))
            if vl < best - 1e-5:
                best, wait = vl, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= 12:
                    break
        if best_state:
            net.load_state_dict(best_state)
        net.eval()
        fut_cal = _calendar_feats(future_dates, closed)
        window_feats = feats[-window:].copy()
        preds = []
        with torch.no_grad():
            for i in range(len(fut_cal)):
                x = torch.tensor(window_feats[None, :, :], dtype=torch.float32)
                c = torch.tensor(fut_cal[i][None, :], dtype=torch.float32)
                p = net(x, c).item()
                preds.append(p)
                window_feats = np.vstack([window_feats[1:], np.concatenate([[p], fut_cal[i]])])
        pred = np.expm1(np.array(preds) * sd + mu)
        out = {"pred": _apply_closed(pred, future_dates, closed), "engine": "PyTorch LSTM", "history": hist}
        return out
    except ImportError:
        print("[INFO] PyTorch not installed -> using gradient-boosting lag model instead of LSTM.")
    except Exception as e:  # noqa
        print(f"[INFO] LSTM failed ({type(e).__name__}: {e}); using gradient-boosting lag model.")
    from sklearn.ensemble import HistGradientBoostingRegressor
    lags = [1, 2, 3, 7, 14, 21, 28]
    full = list(ys)
    Xl, yl = [], []
    for t in range(max(lags), len(ys)):
        Xl.append([ys[t - l] for l in lags] + list(cal[t]))
        yl.append(ys[t])
    gb = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, random_state=seed).fit(np.array(Xl), np.array(yl))
    fut_cal = _calendar_feats(future_dates, closed)
    preds = []
    for i in range(len(fut_cal)):
        row = [full[-l] for l in lags] + list(fut_cal[i])
        p = float(gb.predict(np.array([row]))[0])
        full.append(p)
        preds.append(p)
    pred = np.expm1(np.array(preds) * sd + mu)
    return {"pred": _apply_closed(pred, future_dates, closed), "engine": "GBM lag model (LSTM fallback)", "history": []}


def make_lag_features(daily: pd.DataFrame, target: str = "quantity") -> pd.DataFrame:
    """Lag, rolling and calendar features for supervised demand models."""
    d = daily[["ds", target]].copy().rename(columns={target: "y"})
    for l in (1, 2, 3, 7, 14, 21, 28):
        d[f"lag_{l}"] = d["y"].shift(l)
    for w in (7, 14, 28):
        d[f"roll_mean_{w}"] = d["y"].shift(1).rolling(w).mean()
        d[f"roll_std_{w}"] = d["y"].shift(1).rolling(w).std()
    d["roll_max_7"] = d["y"].shift(1).rolling(7).max()
    d["ewm_7"] = d["y"].shift(1).ewm(span=7).mean()
    d["expanding_mean"] = d["y"].shift(1).expanding().mean()
    dt = pd.to_datetime(d["ds"])
    d["dow"], d["month"], d["week"] = dt.dt.dayofweek, dt.dt.month, dt.dt.isocalendar().week.astype(int)
    d["day_of_month"], d["is_weekend"] = dt.dt.day, (dt.dt.dayofweek >= 5).astype(int)
    d["is_month_end"] = dt.dt.is_month_end.astype(int)
    return d.dropna().reset_index(drop=True)


def optimize_blend(y_val, p_prophet, p_lstm):
    best_w, best = 0.5, 1e18
    rows = []
    for w in np.linspace(0, 1, 21):
        s = wape(y_val, w * np.asarray(p_prophet) + (1 - w) * np.asarray(p_lstm))
        rows.append((w, s))
        if s < best:
            best, best_w = s, w
    return float(best_w), pd.DataFrame(rows, columns=["w_prophet", "WAPE_%"])


def forecast_bundle(daily: pd.DataFrame, target: str = "quantity", horizon: int | None = None,
                    force: bool = False, lstm_epochs: int = 60) -> dict:
    """Full pipeline: val fit -> blend weight -> test fit -> final future forecast. Cached."""
    h = horizon or CONFIG["FORECAST_HORIZON"]
    key = f"bundle_{target}_{h}_{len(daily)}"
    f = MODELS / f"{key}.pkl"
    if not force and key in _CACHE:
        return _CACHE[key]
    if not force and f.exists():
        try:
            _CACHE[key] = pickle.load(open(f, "rb"))
            return _CACHE[key]
        except Exception:
            pass
    tr, va, te = split_series(daily, h)
    pv = prophet_forecast(tr, va["ds"], target)
    lv = lstm_forecast(tr, va["ds"], target, epochs=lstm_epochs)
    w, blend_curve = optimize_blend(va[target].values, pv["pred"], lv["pred"])
    trv = pd.concat([tr, va])
    pt = prophet_forecast(trv, te["ds"], target, return_model=True)
    lt = lstm_forecast(trv, te["ds"], target, epochs=lstm_epochs)
    hyb = w * pt["pred"] + (1 - w) * lt["pred"]
    nv = naive_forecast(trv, te["ds"], target)
    hw = holt_winters_forecast(trv, te["ds"], target)
    y = te[target].values
    table = pd.DataFrame({
        "Seasonal Naive": forecast_metrics(y, nv), "Holt-Winters": forecast_metrics(y, hw),
        f"Prophet": forecast_metrics(y, pt["pred"]), "LSTM": forecast_metrics(y, lt["pred"]),
        "Hybrid (Prophet+LSTM)": forecast_metrics(y, hyb)}).T
    # final future forecast (fit on everything)
    fut_dates = pd.date_range(daily["ds"].max() + pd.Timedelta(days=1), periods=h, freq="D")
    pf = prophet_forecast(daily, fut_dates, target)
    lf = lstm_forecast(daily, fut_dates, target, epochs=lstm_epochs)
    hf = w * pf["pred"] + (1 - w) * lf["pred"]
    err = rmse(y, hyb)
    future = pd.DataFrame({"ds": fut_dates, "prophet": pf["pred"], "lstm": lf["pred"], "hybrid": hf,
                           "lower": np.clip(hf - 1.28 * err, 0, None), "upper": hf + 1.28 * err})
    bundle = {"target": target, "horizon": h, "weight_prophet": w, "blend_curve": blend_curve,
              "train": tr, "val": va, "test": te, "test_preds": {"Seasonal Naive": nv, "Holt-Winters": hw, "Prophet": pt["pred"],
                                                               "LSTM": lt["pred"], "Hybrid": hyb},
              "prophet_lower": pt["lower"], "prophet_upper": pt["upper"], "prophet_model": None,
              "prophet_forecast_df": pt.get("forecast_df"), "lstm_history": lt.get("history", []),
              "metrics": table, "future": future, "engines": {"prophet": pt["engine"], "lstm": lt["engine"]}}
    try:
        pickle.dump({**bundle, "prophet_model": None}, open(f, "wb"))
    except Exception:
        pass
    _CACHE[key] = bundle
    log_metrics("forecast", {"target": target, "horizon_days": h, "blend_weight_prophet": w,
                             "engines": bundle["engines"], "table": table.round(3).to_dict(orient="index")})
    return bundle


# ----------------------------------------------------------------------------
# 7. CHURN PREDICTION
# ----------------------------------------------------------------------------
def customer_features(hist: pd.DataFrame, snapshot, returns: pd.DataFrame | None = None) -> pd.DataFrame:
    snapshot = pd.Timestamp(snapshot)
    f = rfm_table(hist, snapshot)[["recency", "frequency", "monetary", "n_products", "avg_order_value", "tenure_days"]].copy()
    orders = (hist.groupby(["customer_id", "invoice"])
              .agg(date=("invoice_date", "min"), value=("revenue", "sum"), items=("quantity", "sum"), lines=("stock_code", "count"))
              .reset_index().sort_values(["customer_id", "date"]))
    orders["gap"] = orders.groupby("customer_id")["date"].diff().dt.days
    gap = orders.groupby("customer_id")["gap"].agg(avg_gap_days="mean", std_gap_days="std")
    f = f.join(gap)
    f["single_order"] = (f["frequency"] == 1).astype(int)
    f["avg_gap_days"] = f["avg_gap_days"].fillna(f["avg_gap_days"].median())
    f["std_gap_days"] = f["std_gap_days"].fillna(0)
    f["avg_items_per_order"] = orders.groupby("customer_id")["items"].mean()
    f["avg_lines_per_order"] = orders.groupby("customer_id")["lines"].mean()
    for w in (30, 90):
        o = orders[orders["date"] > snapshot - pd.Timedelta(days=w)]
        f[f"orders_last_{w}d"] = o.groupby("customer_id").size().reindex(f.index).fillna(0)
    prev = orders[(orders["date"] <= snapshot - pd.Timedelta(days=90)) & (orders["date"] > snapshot - pd.Timedelta(days=180))]
    f["orders_prev_90d"] = prev.groupby("customer_id").size().reindex(f.index).fillna(0)
    f["order_trend"] = (f["orders_last_90d"] - f["orders_prev_90d"]) / (f["orders_prev_90d"] + 1)
    r90 = orders[orders["date"] > snapshot - pd.Timedelta(days=90)].groupby("customer_id")["value"].sum()
    f["revenue_last_90d"] = r90.reindex(f.index).fillna(0)
    f["recency_gap_ratio"] = f["recency"] / f["avg_gap_days"].clip(lower=1)
    f["weekend_share"] = hist.groupby("customer_id")["is_weekend"].mean()
    f["avg_hour"] = hist.groupby("customer_id")["hour"].mean()
    f["is_uk"] = (hist.groupby("customer_id")["country"].agg(lambda s: s.iloc[0]) == "United Kingdom").astype(int)
    f["n_categories"] = hist.groupby("customer_id")["category"].nunique()
    if returns is not None and len(returns):
        r = returns[(returns["invoice_date"] < snapshot) & returns["customer_id"].notna()]
        f["n_returns"] = r.groupby("customer_id").size().reindex(f.index).fillna(0)
    else:
        f["n_returns"] = 0
    return f.fillna(0)


def build_churn_dataset(sales: pd.DataFrame, returns: pd.DataFrame | None = None, window_days: int | None = None):
    """Time-based label (no leakage): features from history <= cutoff; churn = no purchase in the next `window_days`."""
    window_days = window_days or CONFIG["CHURN_WINDOW_DAYS"]
    end = sales["invoice_date"].max()
    cutoff = (end - pd.Timedelta(days=window_days)).normalize()
    hist = sales[sales["invoice_date"] < cutoff + pd.Timedelta(days=1)]
    fut = sales[sales["invoice_date"] >= cutoff + pd.Timedelta(days=1)]
    X = customer_features(hist, cutoff + pd.Timedelta(days=1), returns)
    y = (~X.index.isin(fut["customer_id"].unique())).astype(int)
    return X, pd.Series(y, index=X.index, name="churn"), cutoff


def precision_at_k(y, p, frac=0.2):
    y, p = np.asarray(y), np.asarray(p)
    k = max(int(len(y) * frac), 1)
    top = np.argsort(-p)[:k]
    return float(y[top].mean())


def recall_at_k(y, p, frac=0.2):
    y, p = np.asarray(y), np.asarray(p)
    k = max(int(len(y) * frac), 1)
    top = np.argsort(-p)[:k]
    return float(y[top].sum() / max(y.sum(), 1))


def make_xgb(**kw):
    params = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
                  min_child_weight=3, reg_lambda=2.0, eval_metric="auc", random_state=SEED, n_jobs=-1)
    params.update(kw)
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(**params), "XGBoost"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED), "HistGradientBoosting (XGBoost fallback)"


def split_churn(X, y, seed: int = SEED):
    from sklearn.model_selection import train_test_split
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=seed)
    return Xtr, Xva, Xte, ytr, yva, yte


def train_churn_models(X, y):
    """Logistic Regression, Random Forest, XGBoost. Returns dict(models, splits, metrics DataFrame, probs)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
    Xtr, Xva, Xte, ytr, yva, yte = split_churn(X, y)
    pos_w = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    xgb, xname = make_xgb(scale_pos_weight=pos_w)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight="balanced_subsample",
                                                random_state=SEED, n_jobs=-1),
        xname: xgb,
    }
    rows, probs, thresholds = {}, {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
        ths = np.linspace(0.05, 0.95, 91)
        th = float(ths[np.argmax([f1_score(yva, pv >= t) for t in ths])])
        thresholds[name] = th
        probs[name] = pt
        rows[name] = {"AUC-ROC": roc_auc_score(yte, pt), "PR-AUC": average_precision_score(yte, pt),
                      "F1": f1_score(yte, pt >= th), "Precision@top20%": precision_at_k(yte, pt, 0.2),
                      "Recall@top20%": recall_at_k(yte, pt, 0.2), "Brier": brier_score_loss(yte, pt), "threshold": th}
    return {"models": models, "xgb_name": xname, "splits": (Xtr, Xva, Xte, ytr, yva, yte),
            "metrics": pd.DataFrame(rows).T, "probs": probs, "thresholds": thresholds}


def score_all_customers(model, sales: pd.DataFrame, returns: pd.DataFrame | None = None) -> pd.DataFrame:
    snap = sales["invoice_date"].max().normalize() + pd.Timedelta(days=1)
    X = customer_features(sales, snap, returns)
    out = X[["recency", "frequency", "monetary"]].copy()
    out["churn_probability"] = model.predict_proba(X)[:, 1]
    out["risk_band"] = pd.cut(out["churn_probability"], [-0.01, 0.4, 0.7, 1.01], labels=["Low", "Medium", "High"])
    return out.reset_index()


# ----------------------------------------------------------------------------
# 8. INVENTORY OPTIMISATION
# ----------------------------------------------------------------------------
def z_value(service_level: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(service_level))


def sku_daily_matrix(sales: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    top_n = top_n or CONFIG["TOP_N_SKUS"]
    top = sales.groupby("stock_code")["revenue"].sum().nlargest(top_n).index
    d = sales[sales["stock_code"].isin(top)].groupby(["date", "stock_code"])["quantity"].sum().unstack(fill_value=0)
    idx = pd.date_range(sales["date"].min(), sales["date"].max(), freq="D")
    return d.reindex(idx, fill_value=0)


def inventory_plan(sales: pd.DataFrame, growth_ratio: float = 1.0, lead_time: int | None = None,
                   review: int | None = None, service_level: float | None = None, top_n: int | None = None,
                   order_cost: float | None = None, holding_rate: float | None = None, seed: int = SEED) -> pd.DataFrame:
    """Reorder point / safety stock / EOQ / order-up-to per SKU.
    NOTE: real datasets contain no stock-on-hand column, so current stock is SIMULATED for the demo."""
    L = lead_time or CONFIG["LEAD_TIME_DAYS"]
    R = review or CONFIG["REVIEW_PERIOD_DAYS"]
    sl = service_level or CONFIG["SERVICE_LEVEL"]
    S_cost = order_cost or CONFIG["ORDER_COST"]
    H_rate = holding_rate or CONFIG["HOLDING_RATE"]
    z = z_value(sl)
    mat = sku_daily_matrix(sales, top_n)
    recent = mat.tail(90)
    desc = sales.groupby("stock_code")["description"].agg(lambda s: s.mode().iloc[0])
    price = sales.groupby("stock_code")["unit_price"].median()
    rng = np.random.default_rng(seed)
    rows = []
    for sku in mat.columns:
        mu_h, sd_h = float(recent[sku].mean()), float(recent[sku].std())
        mu_f = mu_h * growth_ratio
        ss = z * sd_h * math.sqrt(L + R)
        rop = mu_f * L + z * sd_h * math.sqrt(L)
        s_up = mu_f * (L + R) + ss
        annual = mu_f * 365
        unit_cost = 0.6 * float(price[sku])
        eoq = math.sqrt(2 * annual * S_cost / max(H_rate * unit_cost, 1e-6))
        stock = float(np.round(rng.uniform(0.15, 4.0) * mu_h * (L + R)))
        cover = stock / max(mu_f, 1e-6)
        order_qty = max(0.0, math.ceil(s_up - stock))
        status = "Reorder Now" if stock < rop else ("Overstock" if cover > 45 else "Healthy")
        rows.append({"stock_code": sku, "description": desc[sku], "unit_price": float(price[sku]),
                     "avg_daily_demand_hist": mu_h, "forecast_daily_demand": mu_f, "demand_std": sd_h,
                     "safety_stock": ss, "reorder_point": rop, "order_up_to": s_up, "EOQ": eoq,
                     "current_stock_simulated": stock, "days_of_cover": cover, "recommended_order_qty": order_qty,
                     "status": status, "order_value": order_qty * unit_cost})
    return pd.DataFrame(rows).sort_values("order_value", ascending=False).reset_index(drop=True)


def _simulate_policy(demand, target_level, L, R, init):
    stock, pipe = float(init), {}
    lost, lost_days, inv_sum, over_days = 0.0, 0, 0.0, 0
    cap = max(float(np.mean(demand)) * 30, 1.0)
    for t, d in enumerate(demand):
        stock += pipe.pop(t, 0.0)
        sold = min(stock, d)
        stock -= sold
        if d > sold:
            lost += d - sold
            lost_days += 1
        inv_sum += stock
        over_days += stock > cap
        if t % R == 0:
            q = target_level - (stock + sum(pipe.values()))
            if q > 0:
                pipe[t + L] = pipe.get(t + L, 0.0) + q
    n = len(demand)
    return {"lost_units": lost, "stockout_days": lost_days, "avg_inventory": inv_sum / n, "overstock_days": over_days,
            "fill_rate": 1 - lost / max(float(np.sum(demand)), 1e-9)}


def inventory_backtest(sales: pd.DataFrame, growth_ratio: float = 1.0, horizon: int | None = None, **kw) -> pd.DataFrame:
    """Back-test on the last `horizon` days: naive buffer rule vs forecast-driven policy (real demand replayed)."""
    h = horizon or CONFIG["FORECAST_HORIZON"]
    L, R = CONFIG["LEAD_TIME_DAYS"], CONFIG["REVIEW_PERIOD_DAYS"]
    z = z_value(CONFIG["SERVICE_LEVEL"])
    mat = sku_daily_matrix(sales, kw.get("top_n"))
    hist, test = mat.iloc[:-h], mat.iloc[-h:]
    rows = []
    for sku in mat.columns:
        mu, sd = float(hist[sku].tail(90).mean()), float(hist[sku].tail(90).std())
        s_naive = mu * (L + R) * 1.25
        s_opt = mu * growth_ratio * (L + R) + z * sd * math.sqrt(L + R)
        dem = test[sku].values.astype(float)
        a = _simulate_policy(dem, s_naive, L, R, s_naive)
        b = _simulate_policy(dem, s_opt, L, R, s_opt)
        price = float(sales.loc[sales["stock_code"] == sku, "unit_price"].median())
        margin, hold = 0.4 * price, 0.6 * price * CONFIG["HOLDING_RATE"] / 365
        row = {"stock_code": sku, "unit_price": price}
        for tag, r_ in (("naive", a), ("opt", b)):
            row.update({f"{tag}_{k}": v for k, v in r_.items()})
            row[f"{tag}_cost"] = r_["lost_units"] * margin + r_["avg_inventory"] * h * hold
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 9. DRIFT DETECTION & MONITORING
# ----------------------------------------------------------------------------
def psi(ref, cur, bins: int = 10) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_table(ref: pd.DataFrame, cur: pd.DataFrame, cols) -> pd.DataFrame:
    from scipy.stats import ks_2samp
    rows = []
    for c in cols:
        p = psi(ref[c], cur[c])
        ks = ks_2samp(ref[c], cur[c])
        shift = (cur[c].mean() - ref[c].mean()) / (abs(ref[c].mean()) + 1e-9) * 100
        rows.append({"feature": c, "PSI": p, "KS_stat": ks.statistic, "KS_p_value": ks.pvalue, "mean_shift_%": shift,
                     "status": "DRIFT" if p > 0.2 else ("WARNING" if p > 0.1 else "OK")})
    return pd.DataFrame(rows)


def retrain_decision(drift_df: pd.DataFrame, current_mape: float, mape_limit: float = 12.0, psi_limit: float = 0.2) -> dict:
    n_drift = int((drift_df["PSI"] > psi_limit).sum())
    reasons = []
    if n_drift >= 2:
        reasons.append(f"{n_drift} features with PSI > {psi_limit}")
    if current_mape > mape_limit * 2.5:
        reasons.append(f"rolling MAPE {current_mape:.1f}% is far above target {mape_limit}%")
    return {"retrain": bool(reasons), "reasons": reasons, "n_drifted_features": n_drift}


def evidently_report(ref: pd.DataFrame, cur: pd.DataFrame, path: str):
    """Optional Evidently AI HTML report - supports both old and new Evidently APIs; never raises."""
    try:
        try:  # evidently >= 0.7
            from evidently import Report
            from evidently.presets import DataDriftPreset
            snap = Report([DataDriftPreset()]).run(cur, ref)
            snap.save_html(path)
        except ImportError:  # evidently 0.4.x
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset
            rep = Report(metrics=[DataDriftPreset()])
            rep.run(reference_data=ref, current_data=cur)
            rep.save_html(path)
        return path
    except Exception as e:  # noqa
        print(f"[INFO] Evidently report skipped ({type(e).__name__}). Native PSI/KS drift report above is the main output.")
        return None


# ----------------------------------------------------------------------------
# 10. MLFLOW (safe wrapper - never crashes the notebook)
# ----------------------------------------------------------------------------
def _mlflow_setup(experiment: str):
    import mlflow
    ROOT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        mlflow.create_experiment(experiment, artifact_location=(ROOT / "mlartifacts" / experiment.replace(" ", "_")).as_uri())
    mlflow.set_experiment(experiment)
    return mlflow


def mlflow_log(experiment: str, run_name: str, params: dict | None = None, metrics: dict | None = None,
               artifacts: list | None = None, tags: dict | None = None):
    params, metrics = params or {}, metrics or {}
    clean_m = {str(k).replace("%", "pct").replace("@", "_at_").replace(" ", "_"): float(v) for k, v in metrics.items()
               if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v)}
    try:
        mlflow = _mlflow_setup(experiment)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({str(k)[:250]: str(v)[:250] for k, v in params.items()})
            mlflow.log_metrics(clean_m)
            if tags:
                mlflow.set_tags({str(k): str(v) for k, v in tags.items()})
            mlflow.set_tag("author", AUTHOR)
            for a in artifacts or []:
                if a and Path(a).exists():
                    mlflow.log_artifact(str(a))
            return run.info.run_id
    except Exception as e:  # noqa
        fb = ROOT / "outputs" / "mlflow_fallback.jsonl"
        fb.parent.mkdir(parents=True, exist_ok=True)
        with open(fb, "a") as fh:
            fh.write(json.dumps({"experiment": experiment, "run": run_name, "params": params, "metrics": clean_m}, default=_json_safe) + "\n")
        print(f"[INFO] MLflow unavailable ({type(e).__name__}); run saved to {fb.name} instead.")
        return None


def mlflow_runs(experiments: list[str] | None = None) -> pd.DataFrame:
    try:
        mlflow = _mlflow_setup("RetailPulse-Week1")
        exps = experiments or [e.name for e in mlflow.search_experiments() if e.name != "Default"]
        df = mlflow.search_runs(experiment_names=exps)
        return df
    except Exception as e:  # noqa
        print(f"[INFO] Could not read MLflow runs ({type(e).__name__}).")
        fb = ROOT / "outputs" / "mlflow_fallback.jsonl"
        if fb.exists():
            return pd.DataFrame([json.loads(l) for l in fb.read_text().splitlines()])
        return pd.DataFrame()


# ----------------------------------------------------------------------------
# 11. PROJECT VISUALS (banner, architecture, workflow) - drawn in code, no external images
# ----------------------------------------------------------------------------
def _gradient(ax, c1, c2, extent):
    g = np.linspace(0, 1, 256).reshape(1, -1)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("g", [c1, c2])
    ax.imshow(g, aspect="auto", cmap=cmap, extent=extent, zorder=0)


def draw_banner(show=True):
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    _gradient(ax, "#0F2744", "#14B8A6", [0, 13, 0, 3.6])
    ax.text(0.5, 2.45, "RetailPulse", fontsize=38, fontweight="bold", color="white", va="center")
    ax.text(0.5, 1.6, "AI-Powered Customer Analytics & Demand Forecasting Platform", fontsize=16, color="#E6FFFA", va="center")
    ax.text(0.5, 0.95, "Demand Forecasting  |  Customer Segmentation  |  Churn Prediction  |  Inventory Optimization",
            fontsize=11.5, color="#CCFBF1", va="center")
    ax.text(0.5, 0.4, f"Made by {AUTHOR}   |   Zidio Development - Data Science & Analytics   |   March 2026",
            fontsize=10.5, color="white", va="center", style="italic")
    xs = np.linspace(9.3, 12.4, 8)
    hs = np.array([0.6, 1.0, 0.8, 1.4, 1.2, 1.8, 1.6, 2.2])
    for x, hgt, c in zip(xs, hs, ["#F59E0B"] * 8):
        ax.add_patch(plt.Rectangle((x, 0.5), 0.28, hgt, color=c, alpha=0.9, zorder=3))
    ax.plot(xs + 0.14, hs + 0.75, color="white", lw=2.5, marker="o", zorder=4)
    return save_fig(fig, "00_project_banner", "RetailPulse project banner", "Project hero image.", "Project Visuals", show)


def _box(ax, x, y, w, h, text, fc="#E0F2FE", ec="#0369A1", fs=9, bold=False):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.4, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", zorder=4)


def draw_architecture(show=True):
    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.text(7, 6.9, "RetailPulse - End-to-End Architecture", ha="center", fontsize=16, fontweight="bold", color="#1F3A5F")
    layers = [
        ("1. DATA LAYER", "#DBEAFE", "#1D4ED8", ["Kaggle / UCI Online Retail II", "CSV / XLSX upload", "ETL + Cleaning", "Data validation (GE-style)"]),
        ("2. FEATURE ENGINEERING", "#DCFCE7", "#15803D", ["RFM scores", "Lag features (1,7,14,28)", "Rolling averages", "Customer behaviour features"]),
        ("3. MODEL LAYER", "#FEF3C7", "#B45309", ["K-Means / DBSCAN", "Prophet + LSTM hybrid", "XGBoost churn + SHAP", "Inventory optimiser", "MLflow tracking"]),
        ("4. SERVING LAYER", "#FCE7F3", "#BE185D", ["Streamlit dashboard", "FastAPI endpoints", "Power BI report", "Docker + Kubernetes"]),
        ("5. MONITORING LAYER", "#EDE9FE", "#6D28D9", ["PSI / KS drift detection", "Evidently AI report", "Rolling MAPE tracking", "Auto-retrain trigger"]),
    ]
    w, gap = 2.4, 0.3
    for i, (title, fc, ec, items) in enumerate(layers):
        x = 0.3 + i * (w + gap)
        _box(ax, x, 5.7, w, 0.65, title, fc=ec, ec=ec, fs=10, bold=True)
        ax.texts[-1].set_color("white")
        for j, it in enumerate(items):
            _box(ax, x, 4.6 - j * 0.95, w, 0.75, it, fc=fc, ec=ec, fs=9)
        if i < len(layers) - 1:
            ax.annotate("", xy=(x + w + gap - 0.03, 6.02), xytext=(x + w + 0.03, 6.02),
                        arrowprops=dict(arrowstyle="-|>", color="#334155", lw=2))
    ax.annotate("", xy=(0.8, 0.45), xytext=(13.2, 0.45), arrowprops=dict(arrowstyle="-|>", color="#6D28D9", lw=2, ls="--"))
    ax.text(7, 0.15, "Feedback loop: drift detected -> retrain -> new model version in MLflow -> redeploy", ha="center", fontsize=10, color="#6D28D9", style="italic")
    return save_fig(fig, "00_architecture_diagram", "RetailPulse architecture diagram",
                    "Five layers: data, feature engineering, models, serving and monitoring with a retraining feedback loop.", "Project Visuals", show)


def draw_workflow(show=True):
    steps = ["Collect dataset", "Import libraries", "Load dataset", "Data cleaning", "Feature engineering", "EDA",
             "RFM segmentation", "Demand forecasting", "Churn prediction", "Inventory optimisation", "Dashboard (Streamlit / Power BI)", "Deploy (FastAPI / Streamlit)"]
    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.text(7, 4.25, "RetailPulse - 12 Step Workflow", ha="center", fontsize=15, fontweight="bold", color="#1F3A5F")
    for i, s in enumerate(steps):
        row, col = divmod(i, 6)
        x, y = 0.35 + col * 2.28, 2.65 - row * 1.75
        _box(ax, x, y, 1.95, 1.0, f"{i + 1}\n{s}", fc=["#DBEAFE", "#DCFCE7", "#FEF3C7"][min(i // 4, 2)], ec=COLORS[i % len(COLORS)], fs=9, bold=True)
        if col < 5:
            ax.annotate("", xy=(x + 2.25, y + 0.5), xytext=(x + 1.97, y + 0.5), arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.8))
    ax.annotate("", xy=(0.9, 2.55), xytext=(13.0, 3.6), arrowprops=dict(arrowstyle="-|>", color="#94A3B8", lw=1.2, connectionstyle="arc3,rad=-0.15", ls=":"))
    return save_fig(fig, "00_workflow", "RetailPulse 12-step workflow", "Workflow from data collection to deployment.", "Project Visuals", show)
