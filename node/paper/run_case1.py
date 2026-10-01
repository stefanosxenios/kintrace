"""Case study 1 runs: every experiment x noise realisation, the mu-form ablation, and the closing-the-loop test.

    python run_case1.py main      --seeds 10 --workers 30
    python run_case1.py ablation  --seeds 10 --workers 30
    python run_case1.py loop      --seeds 10 --workers 10
Results go to results/case1*/ as pickles; existing files are skipped.
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
from kinid import case1 as c1  # noqa: E402

OUT = os.path.join(HERE, "results")


def _main(job):
    name, seed = job
    path = os.path.join(OUT, "case1", f"{name}_{seed}.pkl")
    if os.path.exists(path):
        return path, 0.0
    t0 = time.time()
    res = c1.run_job(name, seed)
    pickle.dump(res, open(path, "wb"))
    return path, time.time() - t0


def _ablation(job):
    name, seed, factor = job
    path = os.path.join(OUT, "case1_ablation", f"{name}_{seed}_{'factor' if factor else 'softplus'}.pkl")
    if os.path.exists(path):
        return path, 0.0
    t0 = time.time()
    runs = c1.make_runs(name, seed, keys=("a",))
    dense = c1.fit_node(runs, factor=factor)
    mu_fn, p = c1.truth_of(name)
    true_mu = mu_fn(dense["X"], np.clip(dense["S"], 0, None), p)
    res = dict(name=name, seed=seed, factor=factor, loss=dense["loss"],
               mu_rel_err=float(np.sqrt(np.mean((dense["mu"] - true_mu) ** 2)) / true_mu.max()),
               S_min=float(dense["Sr"].min()), Yxs=dense["Yxs"])
    pickle.dump(res, open(path, "wb"))
    return path, time.time() - t0


def _loop(seed):
    path = os.path.join(OUT, "case1_loop", f"{seed}.pkl")
    if os.path.exists(path):
        return path, 0.0
    t0 = time.time()
    load = lambda n: pickle.load(open(os.path.join(OUT, "case1", f"{n}_{seed}.pkl"), "rb"))
    hidden = load("hidden")["joint"]
    others = {}
    for name in c1.TRUE:
        r = load(name)
        others[name] = (c1.make_runs(name, seed), r["joint"]["node"], r["joint"]["library"])
    res = c1.loop_job(seed, hidden, others)
    pickle.dump(res, open(path, "wb"))
    return path, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["main", "ablation", "loop"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()
    for sub in ("case1", "case1_ablation", "case1_loop"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    if a.stage == "main":
        # slowest experiments first, so the pool finishes evenly
        order = ["hidden", "aiba", "moser", "tessier", "monod+kd", "haldane+kd"] + \
                [n for n in c1.TRUE if n not in ("aiba", "moser", "tessier", "monod+kd", "haldane+kd")]
        jobs, fn = [(n, s) for s in range(a.seeds) for n in order], _main
    elif a.stage == "ablation":
        jobs, fn = [(n, s, f) for s in range(a.seeds) for n in c1.LIB4 for f in (True, False)], _ablation
    else:
        jobs, fn = list(range(a.seeds)), _loop
    t0 = time.time()
    with mp.get_context("fork").Pool(a.workers, maxtasksperchild=1) as pool:
        for i, (path, dt) in enumerate(pool.imap_unordered(fn, jobs), 1):
            print(f"[{i}/{len(jobs)}] {os.path.basename(path)}  {dt:.0f} s  (elapsed {time.time() - t0:.0f} s)", flush=True)
