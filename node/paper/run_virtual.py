"""Case study 2, virtual runs: real feed, volume, initial state and sampling schedule; known kinetics.

V18: run BC18 with M31 calibrated on BC18.  V15: run BC15 with M28 calibrated on BC15.
5 % Gaussian noise (of each channel's range), 10 noise realisations, known noise level (chi2-BIC).

    python run_virtual.py --seeds 10 --workers 20
"""
import os
os.environ.setdefault("GFORTRAN_UNBUFFERED_ALL", "1")
import sys
import time
import pickle
import argparse
import multiprocessing as mp

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kinid import case2 as c2  # noqa: E402

WORKBOOK = os.path.join(HERE, "..", "..", "data", "FedBatch_xylitol_in_YP.xlsx")
OUT = os.path.join(HERE, "results", "virtual")
VIRTUAL = {"V18": ("BC18", "M31"), "V15": ("BC15", "M28")}
NOISE = 0.05


def calibrated(base, mech):
    res = pickle.load(open(os.path.join(HERE, "results", "real", f"{base}.pkl"), "rb"))
    return res["library"][mech]["theta"]


def make(vname, seed):
    base, mech = VIRTUAL[vname]
    runs = c2.load_real(WORKBOOK)
    return c2.make_virtual(runs[base], c2.LIBRARY[mech], calibrated(base, mech), noise=NOISE,
                           seed=1000 * seed + list(VIRTUAL).index(vname), name=vname)


def _job(job):
    vname, seed = job
    path = os.path.join(OUT, f"{vname}_{seed}.pkl")
    if os.path.exists(path):
        return path, 0.0
    t0 = time.time()
    run = make(vname, seed)
    res = c2.analyse([run], sigma="known")
    res.update(name=vname, seed=seed, mech=VIRTUAL[vname][1], theta_true=calibrated(*VIRTUAL[vname]),
               Y=run["Y"], truth=run["truth"], sigma=run["sigma"], t=run["t"])
    pickle.dump(res, open(path, "wb"))
    return path, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = [(v, s) for s in range(a.seeds) for v in VIRTUAL]
    t0 = time.time()
    with mp.get_context("fork").Pool(a.workers, maxtasksperchild=1) as pool:
        for i, (path, dt) in enumerate(pool.imap_unordered(_job, jobs), 1):
            print(f"[{i}/{len(jobs)}] {os.path.basename(path)}  {dt:.0f} s  (elapsed {time.time() - t0:.0f} s)", flush=True)
