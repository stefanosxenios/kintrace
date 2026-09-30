# Neural ODEs for kinetic-mechanism identification

This folder applies a **Neural ODE** approach to the two case studies of the kintrace paper. It is a
starting point, contained in one notebook: [`notebooks/node_case_studies.ipynb`](notebooks/node_case_studies.ipynb).

The paper answers two questions for a fed-batch run: *which kinetic mechanism* governs it, and *what
are its parameters*. It does this with LSTMs trained offline on a large in-silico library. Here, each run
is fitted directly instead. A hybrid Neural ODE learns the kinetics from the data, and the mechanism
library is only used afterwards, to put a name and parameters on what was learned.

| step | paper (kintrace) | here |
|---|---|---|
| learn the dynamics | – | hybrid Neural ODE (§2) |
| initial parameters | per-mechanism LSTM regressor | fit rate laws to the learned rates (§3) |
| refinement | least squares, normalised parameters | same, from two starting points (§4) |
| selection | LSTM classifier, softmax confidence | BIC weights (§5) |

The kintrace code is reused as is: the mechanism ODEs, parameter ranges, feed profiles and the noise model.

---

## 1. The problem

A run gives measurements $y_{ij}$ of channel $j$ (biomass, substrates, product) at times $t_i$. The feed
$F(t)$ is known. Each mechanism $m$ in a library $\mathcal{M}$ is an ODE model

$$\dot{x} = f_m\big(x, F(t);\ \theta_m\big),$$

and the goal is to find the mechanism $m$ and its parameters $\theta_m$ that explain the data.

All mechanisms share the same **mass balances** and differ only in the **specific rates**. For Case 1
(state $[X, S, V]$, dilution $D = F/V$):

$$\dot X = (\mu - D)\,X, \qquad \dot S = -\frac{\mu X}{Y_{XS}} + D\,(S_{in} - S), \qquad \dot V = F.$$

The mechanisms differ in $\mu$: Monod $\frac{\mu_{max}S}{K_S+S}$, Contois $\frac{\mu_{max}S}{K_cX+S}$,
Haldane $\frac{\mu_{max}S}{K_S+S+S^2/K_I}$, and biomass inhibition $\frac{\mu_{max}S}{K_S+S}\left(1-\frac{X}{X_{max}}\right)$.

For Case 2 (state $[X, S_1, S_2, P, V]$, paper Eq. 8):

$$\dot X = (\mu_1 - k_d - D)X,\quad \dot S_1 = -\frac{\mu_1 X}{Y_{XS_1}} + D(S_1^f - S_1),\quad
\dot S_2 = -\frac{\mu_2 X}{Y_{PS_2}} + D(S_2^f - S_2),\quad \dot P = \mu_2 X - DP.$$

The 11 mechanisms M21–M31 switch terms on or off in

$$\mu_1 = \frac{\mu_{max,1} S_1}{K_{S_1} + S_1 + S_2^2/K_{SI_1}} \cdot \frac{1}{1 + P/K_{PI_1}}, \qquad
\mu_2 = \frac{\mu_{max,2} S_2}{K_P + S_2 + S_2^2/K_{SI_2}} \cdot \frac{1}{1 + P/K_{PI_2}},$$

and also include or omit death, $k_d$.

## 2. Step 1 — hybrid Neural ODE

The mass balances are kept, and the unknown rate is replaced by a small neural network $\mathrm{NN}_\phi$
(2 hidden layers of 16 tanh units). For Case 1:

$$\mu(X, S) = \mathrm{softplus}\Big(\mathrm{NN}_\phi\big(X/s_X,\ S/s_S\big)\Big)\cdot\frac{S}{S + K},
\qquad K = 0.1\, s_S.$$

Here $s_j$ is the largest measured value of channel $j$. The factor $S/(S+K)$ only says "no growth without
substrate", which is true for every candidate. **No mechanism is assumed.** The yield $Y_{XS}$ is a
trainable scalar.

For Case 2, one network gives both rates, $[\mu_1, \mu_2] = \mathrm{NN}_\phi(S_1, S_2, P)$, with the same
kind of substrate factors. $Y_{XS_1}$, $Y_{PS_2}$ and $k_d$ are trainable scalars, and $k_d$ can shrink
to about 0.

**Training.** The model is integrated with fixed-step RK4 (`torchdiffeq`) and compared with the data:

$$\mathcal{L}(\phi, Y, \delta) = \frac{1}{|\Omega|}\sum_{(i,j)\in\Omega}
\left(\frac{\hat{x}_j(t_i) - y_{ij}}{s_j}\right)^2,
\qquad \hat{x}(0) = y_0 + 0.05\, s \odot \delta.$$

Here $\Omega$ is the set of measured values, so missing samples are skipped. The trainable offset $\delta$
corrects the noisy first sample, which is used as the initial state. Two details make the training reliable:

* **Horizon curriculum.** At iteration $n$ of $N$, only the first
  $k = \min\!\big(1,\ 0.25 + 2n/N\big)\cdot T$ samples enter the loss. Fitting the whole fed-batch run
  from the start often gets stuck in a false minimum, such as a spurious lag phase.
* **Stiffness.** In the substrate-limited phase, $\partial \dot S/\partial S \approx -\mu_{max} X / (Y K)$.
  Explicit RK4 is only stable when $h\,|\lambda| \lesssim 2.8$. The softening constant $K$ therefore
  cannot be tiny. The network can still represent sharper kinetics through its own dependence on $S$.

Case 1 uses 300 Adam steps with $h = 0.5$ h. Case 2 uses 200 steps with $h = 1$ h. The best
full-horizon state is kept.

## 3. Step 2 — match each rate law to the learned rates

The trained Neural ODE gives a trajectory $\hat x(t)$ and a learned rate $\hat\mu(t)$. For every
candidate $m$, its rate law is fitted to $\hat\mu$ on a dense time grid $\mathcal{T}$:

$$\theta_m^{(0)} = \arg\min_{\theta}\ \sum_{t\in\mathcal{T}} \Big(\mu_m\big(\hat x(t);\theta\big) - \hat\mu(t)\Big)^2 .$$

This is an algebraic fit with no ODE solves, so it is cheap. It plays the role of the paper's
per-mechanism LSTM regressor, but it needs no training library and is not confined to a sampling box.
The yields are taken from the Neural ODE. For Case 2, the fit matches $\mu_1$, the net growth
$\mu_1 - k_d$, and $\mu_2$.

## 4. Step 3 — refine each candidate against the data

Each candidate's full ODE, using kintrace's own right-hand side, is fitted by bounded least squares:

$$\chi^2_m(\theta, x_0) = \sum_{(i,j)\in\Omega}\left(\frac{x_j(t_i;\theta, x_0) - y_{ij}}{\sigma_j}\right)^2,
\qquad \sigma_j = \text{noise} \times \text{range of channel } j.$$

* **Noise level.** $\sigma_j$ is assumed known: 3 % in Case 1 and 5 % in Case 2.
* **Bounds.** $\theta \in [0.1\,\ell,\ 5\,u]$, where $[\ell, u]$ are the paper's sampling ranges, so the
  estimate can leave the sampled region.
* **Initial state.** $x_0$ is refined together with $\theta$. In exponential growth, even a small
  error in $X_0$ cannot be absorbed by the kinetics.
* **Two starts.** The refinement starts once from $\theta_m^{(0)}$ and once from the centre
  $(\ell + u)/2$ (the paper's "naive" start), and the lower $\chi^2$ is kept.
* **Normalised coordinates and step size.** The optimisation runs in
  $z = 1 + (\theta - \ell)/(u - \ell) \in [1, 2]$ with finite-difference step $h = 10^{-3}\, z$. A
  forward difference has error $\approx \varepsilon/h + O(h)$, where $\varepsilon \approx 10^{-6}$ is
  the ODE solver's error. With scipy's default $h \approx 10^{-8}$, that gives
  $\varepsilon/h \approx 10^{2}$: the Jacobian is mostly noise and the fit stops early, far from the
  optimum. With $h \approx 10^{-3}$ the error is about $10^{-3}$. The offset of 1 in $z$ keeps the
  relative step from vanishing for parameters at 0 (for example $P_0 = 0$).

## 5. Step 4 — select the mechanism

With $n$ measured values and $k_m$ kinetic parameters:

$$\mathrm{BIC}_m = \chi^2_m + k_m \ln n, \qquad
w_m = \frac{e^{-\Delta_m/2}}{\sum_{m'} e^{-\Delta_{m'}/2}}, \qquad
\Delta_m = \mathrm{BIC}_m - \min_{m'}\mathrm{BIC}_{m'}.$$

The weight $w_m$ plays the role of the classifier's confidence, and candidates with $w_m > 0.1$ form the
shortlist, as with the paper's $p > 0.1$.

---

## 6. Results

**Case 1.** Four mechanisms were simulated with kintrace, one run each, sharing a 6 h batch phase
followed by three feed rates. Each run has 49 samples with 3 % noise, so $n = 98$.

| true mechanism | selected | weight | $\chi^2$ of true mechanism |
|---|---|---|---|
| Monod | Monod | 0.54 (Contois 0.38) | 47 |
| Contois | Contois | 0.89 | 86 |
| Haldane | Haldane | 1.00 | 427 (not at optimum, see §7) |
| biomass inhibition | biomass inhibition | 1.00 | 84 |

The Neural ODE recovered $Y_{XS}$ within 2 % in every run, and its learned $\mu(t)$ reproduces the
signature of each mechanism. For example, Haldane growth speeds up as $S$ falls, and biomass-inhibited
growth stops at $X_{max}$ while substrate accumulates.

**Case 2.** The real BC runs are not in the repository, so two virtual runs stand in for them. Each has
13 irregular offline samples with 5 % noise, so $n = 52$.

| run | generated with | selected | weight | shortlist ($w > 0.1$) |
|---|---|---|---|---|
| BC18-like | M31, paper Table 7 parameters | M31 | 1.00 | M31 |
| BC15-like | M28 | M28 | 0.85 | M28, M30 |

The BC15-like shortlist, M28 and M30, matches the paper's result for the real BC15 run.

**Robustness.** Re-running over 6 noise realisations selected the true mechanism in 22 of 24 Case 1
runs; both misses were Monod read as Contois at $w \approx 0.56$, the near-tie the paper also reports.
It selected the true mechanism in 12 of 12 Case 2 runs.

## 7. What was learned along the way

* **The learned rates alone do not identify the mechanism.** Along one trajectory, $X$ and $S$ (or
  $S_2$ and $P$) change together. A falling $\hat\mu(t)$ can then be explained by
  $X/X_{max}$ or by $S^2/K_I$ equally well, so selection must use the refined fits against the data (§4).
* **Individual constants are often not identifiable, even when the fit is good.** In the Haldane law,
  $\mu_{max}$, $K_S$ and $K_I$ trade off, and its refinement still stops in a local optimum. In Case 2,
  $\mu_{max,2}$ and $K_P$ trade off.
* **BIC needs the noise level.** The Neural ODE's residuals underestimate it, because the network
  overfits sparse data, so the noise level is assumed known instead.
* **The finite-difference step matters for kintrace too.** `benchmark_simple/fitting.py` uses scipy's
  default step. In a quick check with 20 simulated runs per mechanism, its naive start reached the optimum
  on 24–55 % of runs with the default step, and on 100 % with a step of $10^{-3}$. The paper's
  "LSTM start 86 % vs naive 39 %" result (Sec. 4.5) should be re-run with the larger step.

## 8. Running it

```bash
pip install torchdiffeq                 # on top of kintrace's requirements.txt
cd kintrace/node/notebooks
jupyter notebook node_case_studies.ipynb   # about 5 min on a CPU
```

**Next steps:** fit several runs jointly with shared kinetics; plug in the real BC runs through
`xylitol_case_study/realdata.py::load_experiments`; warm-start from the LSTM estimates; run symbolic
regression on the learned rates to propose laws outside the library.
