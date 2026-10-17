"""Execute a notebook headlessly (used to verify there are no errors). Usage: run_nb.py in.ipynb out.ipynb [workspace]"""
import sys, os, time
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
ws = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/tmp/rp_ws_" + src.stem)
ws.mkdir(parents=True, exist_ok=True)
os.environ["RETAILPULSE_HOME"] = str(ws / "retailpulse_work")
nb = nbf.read(src, as_version=4)
t = time.time()
ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
try:
    ep.preprocess(nb, {"metadata": {"path": str(ws)}})
    status = "OK"
except Exception as e:
    status = "FAIL: " + str(e)[-1500:]
nbf.write(nb, dst)
print(f"{src.name}: {status} ({time.time() - t:.0f}s)")
sys.exit(0 if status == "OK" else 1)
