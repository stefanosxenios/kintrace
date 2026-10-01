# Code for the manuscript

*Hybrid neural ODEs and physically constrained symbolic regression for kinetic mechanism identification and
discovery in fed-batch bioprocesses* (manuscript in `../latex/`).

Every number, table and figure of the manuscript is produced by the scripts below; the report scripts write
the figures to `../latex/figs/` and the tables and number macros to `../latex/generated/`.

## Layout

| Path | Content |
|---|---|
| `kinid/core.py` | hybrid neural ODE training, bounded refinement, BIC ranking (known or profiled noise) |
| `kinid/sr.py` | physically constrained symbolic regression (implicit rational form, exponential terms) |
| `kinid/case1.py` | case study 1: single-substrate fed-batch, 11-mechanism library, hidden law, escalation, closing the loop |
| `kinid/case2.py` | case study 2: xylitol bioconversion, library M21–M31, real-data loader, virtual runs, nested restarts |
| `run_case1.py` | case study 1 runs (`main`, `ablation`, `loop`) |
| `run_realdata.py` | real runs BC12, BC15, BC18, BC19 alone and jointly (profile-likelihood BIC) |
| `run_virtual.py` | virtual runs V18 and V15 (needs `results/real/BC15.pkl` and `BC18.pkl`) |
| `report_case1.py`, `report_case2.py` | statistics, figures, tables and LaTeX macros |
| `results/` | pickled results (`*_old*` folders: superseded runs, not used in the manuscript) |

## Reproduce

Python environment: `/home/kostasme/miniconda3/envs/.symbiolabs` (PyTorch, torchdiffeq, SciPy, SymPy, pandas,
matplotlib). The real data are `FedBatch_xylitol_in_YP.xlsx` (sheet `S. cerevisiae (plasmid)`) from the Zenodo
dataset 10.5281/zenodo.10671450 (Toivari and Wiebe, 2025), expected at `../../data/`.

```bash
python run_case1.py main     --seeds 10 --workers 30
python run_case1.py ablation --seeds 10 --workers 30
python run_case1.py loop     --seeds 10 --workers 10
python run_realdata.py --workers 24   # each run alone and the joint groups (joint2, joint3, joint3b, joint4)
python run_virtual.py --seeds 10 --workers 20
python report_case1.py
python report_case2.py
cd ../latex && tectonic manuscript.tex
```

Existing result files are skipped, so a run can be resumed. Approximate times on the 64-core server: case
study 1 about 5 min of one core per experiment (120 experiments in `main`); real data about 50 min in total;
virtual runs about 20 min with 6 workers.
