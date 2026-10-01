"""Case study 2 on the real fed-batch runs: each run alone, and jointly (shared kinetics).

    python run_realdata.py --workers 24
The analyses run one after the other; within each, the candidates are refined in parallel.
Unknown measurement noise: per-channel SDs are profiled out (profile-likelihood BIC).
Results go to results/real/<analysis>.pkl.
"""
import os
os.environ.setdefault("GFORTRAN_UNBUFFERED_ALL", "1")
import sys
import time
import pickle
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kinid import case2 as c2  # noqa: E402

WORKBOOK = os.path.join(HERE, "..", "..", "data", "FedBatch_xylitol_in_YP.xlsx")
OUT = os.path.join(HERE, "results", "real")
ANALYSES = {"BC12": ["BC12"], "BC15": ["BC15"], "BC18": ["BC18"], "BC19": ["BC19"],
            "joint3": ["BC12", "BC15", "BC18"], "joint4": ["BC12", "BC15", "BC18", "BC19"],
            "joint2": ["BC15", "BC18"], "joint3b": ["BC15", "BC18", "BC19"]}


def _job(key, workers):
    path = os.path.join(OUT, f"{key}.pkl")
    if os.path.exists(path):
        return key, 0.0
    t0 = time.time()
    runs = c2.load_real(WORKBOOK)
    res = c2.analyse([runs[k] for k in ANALYSES[key]], sigma=None, workers=workers)
    res["runs"] = ANALYSES[key]
    pickle.dump(res, open(path, "wb"))
    return key, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--only", nargs="*", default=list(ANALYSES))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for key in a.only:
        key, dt = _job(key, a.workers)
        print(f"{key}: {dt:.0f} s", flush=True)
