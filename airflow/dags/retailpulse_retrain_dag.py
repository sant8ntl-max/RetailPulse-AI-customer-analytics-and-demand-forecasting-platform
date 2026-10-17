"""
RetailPulse - automated retraining DAG (Apache Airflow 2.x)
Flow: ingest_clean -> drift_check -> (retrain | skip) -> evaluate_and_register
Run `python scripts/retrain_pipeline.py` for the same logic without Airflow.
Author: Santosh Kumar
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

try:  # Airflow is optional - the file stays importable for tests / linting
    from airflow import DAG
    from airflow.operators.python import BranchPythonOperator, PythonOperator
    from airflow.operators.empty import EmptyOperator
except Exception:  # pragma: no cover
    DAG = None

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def ingest_clean(**_):
    import retailpulse_utils as ru
    ru.get_data(force=True)


def drift_check(**_):
    from retrain_pipeline import drift_needs_retrain
    return "retrain" if drift_needs_retrain() else "skip_retrain"


def retrain(**_):
    from retrain_pipeline import run_retrain
    run_retrain()


def evaluate_and_register(**_):
    import retailpulse_utils as ru
    print("Latest metrics:", ru.get_metrics().get("retrain", {}))


if DAG is not None:
    default_args = {"owner": "santosh.kumar", "retries": 1, "retry_delay": timedelta(minutes=5)}
    with DAG("retailpulse_retraining", default_args=default_args, start_date=datetime(2026, 3, 1),
             schedule="@weekly", catchup=False, tags=["retailpulse", "mlops"]) as dag:
        t1 = PythonOperator(task_id="ingest_clean", python_callable=ingest_clean)
        t2 = BranchPythonOperator(task_id="drift_check", python_callable=drift_check)
        t3 = PythonOperator(task_id="retrain", python_callable=retrain)
        t4 = EmptyOperator(task_id="skip_retrain")
        t5 = PythonOperator(task_id="evaluate_and_register", python_callable=evaluate_and_register, trigger_rule="none_failed_min_one_success")
        t1 >> t2 >> [t3, t4] >> t5
