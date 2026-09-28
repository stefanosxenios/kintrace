# kintrace

Code for *Attention-Based LSTM for Kinetic Mechanism Identification and Parameter
Estimation: The Case for Complex Bioprocesses* (manuscript under review).

A two-stage pipeline that turns a bioreactor time series into an interpretable
ODE kinetic model:

1. **Mechanism identification** — a bidirectional LSTM with additive attention
   classifies which candidate kinetic mechanism generated the trajectory. Each
   trajectory is min–max scaled to its own range, so identification depends on
   the *shape* of the curve, not its magnitude.
2. **Parameter estimation** — a bank of dedicated LSTM regressors, one per
   mechanism, estimates that mechanism's kinetic parameters, which bounded
   least-squares fitting then refines.

Both networks are trained entirely on simulated fermentations.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run every command from the repository root.

## Case Study 1 — single-substrate benchmark (4 mechanisms)

Monod, Contois, Haldane substrate inhibition and logistic biomass inhibition,
sharing one fed-batch mass balance `[X, S, V]`.

```bash
# 1. simulate the training set (~19k trajectories after screening)
python datagen/generate_simple.py --n 5000 --timepoints 100

# 2. train (use a NEW tag: training resumes from any checkpoint already in the tag dir)
python benchmark_simple/run.py --stage classifier --rep hybrid --scaling per_traj --tag mine --epochs 120
python benchmark_simple/run.py --stage regressor  --rep hybrid --scaling global   --tag mine --epochs 120

# 3. refinement benchmark: LSTM estimate vs naive / random / Latin-hypercube multistart
python benchmark_simple/run.py --stage refine   --rep hybrid --scaling global --tag mine --refine-n 200
python benchmark_simple/run.py --stage assemble --rep hybrid --scaling global --tag mine
```

Outputs go to `results/simple_benchmark_<tag>/`. The `--rep hybrid` flag is the
mixed representation of the paper (states concatenated with specific rates).

Figures, using the shipped weights in `results/simple_benchmark_rec/`
(the data from step 1 is still required, for the input scaler):

```bash
python benchmark_simple/dataset_figure.py
python benchmark_simple/training_matrices.py
python benchmark_simple/attention_on_curves.py
python benchmark_simple/ood_paper_figures.py --restarts 40
```

## Case Study 2 — dual-substrate xylitol bioconversion (11 mechanisms)

Glucose supports growth, xylose is converted to xylitol; state
`[X, S1, S2, P, V]`. The library (models 21–31) combines product inhibition,
substrate inhibition and cell death on growth and on conversion.

```bash
# 1. simulate (large: 20k trajectories per mechanism; for a quick run lower --n
#    AND pass a smaller --batch below, since each training split must hold one full batch)
python datagen/generate_v2.py --n 20000 --workers 8

# 2. train
python xylitol_case_study/run.py --stage classifier --rep hybrid --scaling per_traj --tag mine --epochs 120
python xylitol_case_study/run.py --stage regressor  --rep hybrid --regressor per_mechanism --tag mine --epochs 120

# 3. in-silico figures (confusion matrix, per-parameter R²)
python xylitol_case_study/figures.py --tag main
```

### Real fermentations

`xylitol_case_study/evaluate_realdata.py` classifies, estimates and refines
measured fed-batch runs. The experimental data are **not included**; they are
reported in Toivari *et al.* (2025), *Bioresource Technology Reports* 32, 102410,
doi:[10.1016/j.biteb.2025.102410](https://doi.org/10.1016/j.biteb.2025.102410).
With the workbook placed at `data/FedBatch_xylitol_in_YP.xlsx`:

```bash
python xylitol_case_study/evaluate_realdata.py --tag main \
  --xlsx data/FedBatch_xylitol_in_YP.xlsx --sheet "S. cerevisiae (plasmid)" \
  --experiments BC12 BC15 BC18 BC19 --method gpr --plots --baselines
```

## Shipped weights

| path | contents |
|---|---|
| `results/simple_benchmark_rec/` | Case 1 classifier + 4 per-mechanism regressors |
| `results/xylitol_use_case/main/` | Case 2 classifier + 11 per-mechanism regressors, input scaler, parameter bounds |

## Layout

```
LSTMs/                classifier / regressor networks and training loops
kinetic_models/       mechanism ODEs and feed-profile handling
datagen/              in-silico data generation, parameter ranges, noise
dataloader/           in-silico loading and real-data preprocessing
benchmark_simple/     Case Study 1 pipeline, fitting and figures
xylitol_case_study/   Case Study 2 pipeline and real-data evaluation
```
# kintrace
