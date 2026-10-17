"""Assemble the 14 stage notebooks + MAIN notebook (+ 15 launcher) and optionally execute them."""
import sys, json, os, re
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from stages_a import STAGES as SA
try:
    from stages_b import STAGES as SB
except Exception:
    SB = {}
try:
    from stages_c import STAGES as SC
except Exception:
    SC = {}
STAGES = {**SA, **SB, **SC}

AUTHOR = "Santosh Kumar"
UTILS = (REPO / "src" / "retailpulse_utils.py").read_text()
REPORT = (REPO / "src" / "retailpulse_report.py").read_text() if (REPO / "src" / "retailpulse_report.py").exists() else ""

PKG = '''
# Installs ONLY what is missing (so Colab does not need a restart in most cases)
import importlib.util, subprocess, sys
need = {"openpyxl": "openpyxl", "kagglehub": "kagglehub", "xgboost": "xgboost", "shap": "shap", "optuna": "optuna",
        "mlflow": "mlflow", "statsmodels": "statsmodels", "prophet": "prophet", "docx": "python-docx"}
missing = [pip for mod, pip in need.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("Installing:", missing)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=False)
    print("If Colab shows 'Restart runtime', click it once and run all cells again.")
else:
    print("All packages already installed.")
'''
SETUP = '''
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from IPython.display import display
import retailpulse_utils as ru
ru.setup()
print("Author:", ru.AUTHOR, "|", ru.PROJECT)
'''


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


def header(title, week):
    return md(f"""# RetailPulse - {title}
**AI-Powered Customer Analytics & Demand Forecasting Platform**

| | |
|---|---|
| **Project made by** | **{AUTHOR}** |
| **Organisation** | Zidio Development - Data Science & Analytics Domain |
| **Phase** | {week} |
| **Version** | 2.0 - Industry Edition (March 2026) |

> Run order: `Runtime -> Run all`. The first 3 cells install missing packages, write the helper library `retailpulse_utils.py` and import it.""")


def boot_cells():
    return [md("### Setup (run once)\n1) install missing packages  2) write helper library  3) import & create workspace folders"),
            code(PKG), code("%%writefile retailpulse_utils.py\n" + UTILS), code(SETUP)]


def stage_cells(st):
    out = []
    for kind, text in st["cells"]:
        out.append(md(text) if kind == "md" else code(text))
    return out


def new_nb(cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "colab": {"provenance": []}}
    return nb


def build_all(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for name, st in STAGES.items():
        nb = new_nb([header(st["title"], st["week"])] + boot_cells() + stage_cells(st))
        p = out_dir / f"{name}.ipynb"
        nbf.write(nb, p)
        names.append(p)
    # MAIN
    cells = [header("MAIN NOTEBOOK (all stages end-to-end)", "All 28 days - complete pipeline")] + [md(
        "## Table of contents\n" + "\n".join(f"{i + 1}. **{st['title']}**" for i, st in enumerate(STAGES.values())) +
        "\n\n> This single notebook runs the complete RetailPulse project: EDA -> cleaning -> segmentation -> forecasting -> churn -> inventory -> tuning -> drift -> MLflow -> report.")]
    cells += boot_cells()
    for i, (name, st) in enumerate(STAGES.items()):
        cells.append(md(f"---\n# PART {i + 1} - {st['title']}\n*{st['week']}*"))
        cells += [c for c in stage_cells(st)]
    nb = new_nb(cells)
    p = out_dir / "00_RetailPulse_MAIN.ipynb"
    nbf.write(nb, p)
    names.insert(0, p)
    return names


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "notebooks"
    for p in build_all(out):
        print("built", p.name)
