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

$$\mu(X,S) \;=\; \underbrace{\mathrm{softplus}\big(\mathrm{NN}_\phi(X/s_X,\ S/s_S)\big)}_{g(X,S)\ \ge\ 0:\ \text{learned shape}} \;\cdot\; \underbrace{\frac{S}{S+K}}_{\text{fixed: }0 \text{ at } S=0,\ \to 1 \text{ for } S \gg K}, \qquad K = 0.1\, s_S.$$

Here $s_j$ is the largest measured value of channel $j$. The yield $Y_{XS}$ is a trainable scalar.
**No mechanism is assumed.**

For Case 2, one network gives both rates, $[\mu_1, \mu_2] = \mathrm{NN}_\phi(S_1, S_2, P)$, each multiplied
by its own substrate factor, with $K = 0.25\, s_{S_1}$ for glucose and $0.1\, s_{S_2}$ for xylose.
$Y_{XS_1}$, $Y_{PS_2}$ and $k_d$ are trainable scalars, and $k_d$ can shrink to about 0.

### Why this form for μ

Each of the three parts has one job.

1. **The network learns the shape.** It sees $X$ and $S$ scaled to about $[0, 1]$ and can learn any
   dependence: whether μ depends on $X$ (Contois, biomass inhibition), whether it falls at high $S$
   (Haldane), and so on.
2. **Softplus keeps the rate non-negative.** $\mathrm{softplus}(z) = \ln(1+e^z) > 0$. A negative growth
   rate would make substrate appear and biomass shrink; death is modelled separately by $k_d$. Softplus is
   smooth, so gradients flow everywhere, unlike clipping at 0.
3. **$S/(S+K)$ stops growth when there is no substrate.** The factor is exactly 0 at $S = 0$, whatever the
   network outputs, and it approaches 1 when $S \gg K$. This matters because a fed-batch run spends most of
   its time substrate-limited. In that phase

   $$\frac{dS}{dt} = D\,(S_{in} - S) - \frac{\mu(X,S)\,X}{Y}$$

   has a low, stable level $S^*$ at which uptake equals supply:

   $$\mu(X, S^*) = \frac{Y D (S_{in} - S^*)}{X}, \qquad \text{stable when } \frac{\partial \mu}{\partial S} > 0.$$

   Because $\mu(X,0) = 0$ and μ rises with $S$, a small shift in $S^*$ is enough for μ to follow every feed
   change, as in a real culture. With softplus alone, $\mu(X,0) > 0$. When that exceeds the supply, no
   $S^* \ge 0$ exists: the model consumes substrate that isn't there and $S$ goes negative.

**Choosing K** is a compromise:

* **Too small.** The ramp near $S = 0$ becomes steep and the ODE becomes stiff. The stiffness is roughly
  $\lambda \approx g X/(Y K)$, and fixed-step RK4 is only stable when $h|\lambda| \lesssim 2.8$. In an early
  test with $K = 0.01\,s_S$, $S$ oscillated below 0 and training stalled.
* **Too large.** The network must undo a strong, wrong prior near $S = 0$, which makes it harder to learn.

$K = 0.1\,s_S$ (about 3 g/L in Case 1) is close to the substrate noise ($\sigma_S \approx 0.9$ g/L), so it is
about as fine as the data can resolve anyway. Glucose in Case 2 uses $0.25\,s_{S_1}$ because of the longer
1 h step.

**Evidence** (a separate test, not in the notebook). On the four Case 1 runs, with three initialisations each,
dropping the factor (softplus only)
made the fit 1.6–10× worse. The loss's noise floor is about $9\times10^{-4}$.

| mechanism | fit loss, with $S/(S+K)$ | fit loss, softplus only | lowest model $S$ [g/L], with / without |
|---|---|---|---|
| Monod | $4.7\times10^{-4}$ | $4.7\times10^{-3}$ | −0.4 / −2.1 |
| Contois | $9.4\times10^{-4}$ | $2.4\times10^{-3}$ | +1.3 / +1.2 |
| Haldane | $2.6$–$4.9\times10^{-3}$ | $0.95$–$1.6\times10^{-2}$ | −0.3 / −1.8 |
| biomass inhibition | $7.8\times10^{-4}$ | $1.3\times10^{-3}$ | +3.5 / +3.6 |

The Monod run, which is substrate-limited for most of its duration, suffers most. The runs whose $S$ stays
above zero suffer least.

**No bias in the answer.** Every mechanism in the library already has $\mu \ge 0$ and $\mu(S{=}0) = 0$, so
this form rules out only rates no candidate could produce. $K$ exists only inside the Neural ODE. The final
parameters, including $K_S$, come from refining the mechanistic ODEs against the data (§4), with no $K$ involved.

**Training.** The model is integrated with fixed-step RK4 (`torchdiffeq`) and compared with the data:

$$\mathcal{L}(\phi, Y, \delta) = \frac{1}{|\Omega|}\sum_{(i,j)\in\Omega}
\left(\frac{\hat{x}_j(t_i) - y_{ij}}{s_j}\right)^2,
\qquad \hat{x}(0) = y_0 + 0.05\, s \odot \delta.$$

Here $\Omega$ is the set of measured values, so missing samples are skipped. The trainable offset $\delta$
corrects the noisy first sample, which is used as the initial state. Two details make the training reliable:

* **Horizon curriculum.** At iteration $n$ of $N$, only the first
  $k = \min\!\big(1,\ 0.25 + 2n/N\big)\cdot T$ samples enter the loss. Fitting the whole fed-batch run
  from the start often gets stuck in a false minimum, such as a spurious lag phase.
* **A non-stiff rate.** The substrate factor keeps $K$ large enough for RK4 to stay stable; see "Choosing K" above.

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

All numbers below come from the executed notebook, except the robustness study in §6.3.

**How to read $\chi^2$.** If a model is correct and $\sigma_j$ is right, $\chi^2$ should be close to
$n - p$ (data points minus fitted parameters). Values well above that mean the fit is poor, or stuck in a
local optimum. Values somewhat below are expected here: the noise model clips negative samples at 0, which
removes part of the noise wherever a species sits near zero (for example, substrate in the fed phase).

### 6.1 Case 1 — single substrate, 4 mechanisms

**Setup.** All four runs share $X_0 = 1$ g/L, $S_0 = 30$ g/L, $V_0 = 1.5$ L, $S_{in} = 120$ g/L and 24 h,
with a feed of 0, 0.02, 0.05 and 0.08 L/h in four 6 h segments. That is a batch phase followed by rising
feed. The true parameters are $\mu_{max} = 0.5$ (0.6 for Haldane), $K_S = K_c = 1$, $K_I = 10$,
$X_{max} = 20$ and $Y_{XS} = 0.5$. Each run has 49 samples of $X$ and $S$ with 3 % noise, so $n = 98$.

**Step 1: the Neural ODE fit.**

| run | fit loss | learned $Y_{XS}$ (true 0.5) |
|---|---|---|
| Monod | $4.8\times10^{-4}$ | 0.495 |
| Contois | $9.7\times10^{-4}$ | 0.496 |
| Haldane | $2.6\times10^{-3}$ | 0.491 |
| biomass inhibition | $8.1\times10^{-4}$ | 0.503 |

With 3 % noise, the loss of a perfect model is about $0.03^2 \approx 9\times10^{-4}$. Three runs reach this
noise floor, so the network describes the data as well as the true model does. The Haldane run is about 3×
above it. Its early biomass samples are very noisy compared with their size, and the network settled on a
compromise between the initial biomass and the early growth rate. The yield is within 2 % in every run. The learned
$\mu(t)$ curves (notebook figure) show each mechanism's signature without being told the mechanism:
Monod and Contois growth drops when the batch substrate runs out, Haldane growth *speeds up* as $S$ falls,
and biomass-inhibited growth stops at $X \approx 20$ while substrate accumulates.

**Steps 2–4: mechanism selection.**

| true mechanism | selected | weight | runner-up | $\chi^2$ of true mechanism |
|---|---|---|---|---|
| Monod | Monod | 0.54 | Contois ≈ 0.38 | 47 |
| Contois | Contois | 0.89 | biomass inhibition ≈ 0.1 | 86 |
| Haldane | Haldane | 1.00 | – | 427 |
| biomass inhibition | biomass inhibition | 1.00 | – | 84 |

* **All four are correct.**
* **Monod vs Contois is a near-tie.** Contois reduces to Monod when $K_c X \ll S$, so a Monod run can be
  fitted almost as well by Contois. The weight of 0.54 correctly signals the ambiguity; the paper's
  classifier makes the same confusion.
* **Haldane and biomass inhibition are unambiguous**, because their signatures (growth speeding up as $S$
  falls; a biomass plateau with substrate piling up) cannot be reproduced by the others.
* **The Haldane fit is not at its optimum.** A $\chi^2$ of 427 against about 93 expected means the
  refinement stopped in a local optimum, from both starts. Haldane still wins because every other
  mechanism fits much worse.

**Parameters of the true mechanism.**

| mechanism | parameter | true | NODE-matched start | refined |
|---|---|---|---|---|
| Monod | $\mu_{max}$, $K_S$, $Y_{XS}$ | 0.5, 1.0, 0.5 | 0.46, 2.31, 0.50 | 0.54, 1.98, 0.50 |
| Contois | $\mu_{max}$, $K_c$, $Y_{XS}$ | 0.5, 1.0, 0.5 | 0.52, 1.28, 0.50 | 0.47, 0.91, 0.50 |
| Haldane | $\mu_{max}$, $K_S$, $K_I$, $Y_{XS}$ | 0.6, 1.0, 10, 0.5 | 3.51, 25.0, 0.99, 0.49 | 0.69, 3.58, 5.58, 0.44 |
| biomass inh. | $\mu_{max}$, $K_S$, $X_{max}$, $Y_{XS}$ | 0.5, 1.0, 20, 0.5 | 0.68, 4.20, 19.70, 0.50 | 0.58, 1.93, 19.75, 0.50 |

* **Well recovered:** $\mu_{max}$, $Y_{XS}$, $K_c$ and $X_{max}$. The NODE-matched start is already close for
  most of them; for example, $X_{max}$ = 19.70 before refinement.
* **Weakly recovered: $K_S$** (about 2 instead of 1). $K_S$ only shapes growth while $S$ passes through
  a few g/L, which is brief and close to the substrate noise ($\sigma_S \approx 0.9$ g/L).
* **Not recovered: Haldane.** The run stays at high $S$ (25–30 g/L) until the last hours. There,
  $\mu_{max}$, $K_S$ and $K_I$ largely trade off: as $S$ grows, $\mu$ tends towards $\mu_{max} K_I / S$, so
  the network's rates can't separate the constants. The start therefore lands in a poor region ($K_S$ at its bound of 25), and
  refinement does not fully recover. The paper reports the same weakness for Haldane ($R^2$ = 0.32, 0.08,
  0.72 for $\mu_{max}$, $K_S$ and $K_I$).
* **The two starts give the same $\chi^2$** in three runs (47, 86, 84). In the Haldane run the centre start
  did slightly better (427 vs 503).

### 6.2 Case 2 — xylitol, 11 mechanisms

**Setup.** Both runs start from $X = 1$, $S_1 = 20$, $S_2 = P = 0$ g/L and $V_0 = 0.28$ L. Each is sampled
13 times at irregular, offline-style times with 5 % noise, so $n = 52$.

| run | generated with | feed | samples |
|---|---|---|---|
| BC18-like | M31, paper Table 7 (BC18) parameters | from 20 h at 2.0 mL/h, 2.5 mL/h from 80 h; 85 g/L glucose, 300 g/L xylose; 140 h | 0, 4, 8, 12, 16, 22, 30, 45, 60, 80, 100, 120, 140 h |
| BC15-like | M28 ($\mu_{max,2} = 4$, $K_P = 60$, $K_{PI_2} = 10$) | from 25 h at 1.5 mL/h, 3.0 mL/h from 60 h; 108.8 / 334 g/L; 100 h | 0, 3, 6, 9, 12, 16, 21, 28, 40, 55, 70, 85, 100 h |

**Step 1: the Neural ODE fit.**

| run | fit loss | $Y_{XS_1}$ (true) | $Y_{PS_2}$ (true) | $k_d$ (true) |
|---|---|---|---|---|
| BC18-like | $1.2\times10^{-3}$ | 0.56 (0.62) | 1.28 (1.24) | 0.015 (0.018) |
| BC15-like | $1.9\times10^{-3}$ | 0.78 (0.70) | 1.25 (1.30) | 0.005 (0) |

With 5 % noise, the noise floor is about $2.5\times10^{-3}$. Both losses are *below* it: with only 13
samples per species, the network fits part of the noise. That is why its residuals are not used as the noise
estimate for BIC. The yields are within about 10 %, and the learned death rate points the right way: clearly
non-zero for M31 (0.015) and small for M28 (0.005). In BC18-like, the learned $\mu_1(t)$ and $\mu_2(t)$ follow the true
rates closely. In BC15-like, the learned $\mu_2(t)$ has the right level and trend but smooths out its dips.

**Steps 2–4: mechanism selection.**

| run | selected | weight | shortlist ($w > 0.1$) | $\chi^2$ |
|---|---|---|---|---|
| BC18-like (true M31) | M31 | 1.00 | M31 | 27.9 |
| BC15-like (true M28) | M28 | 0.85 | M28, M30 | 21.1 |

The full ranking explains both decisions. $\chi^2$ is the better of the two starts; $\ln 52 = 3.95$ per
parameter in BIC.

| mechanism | $k$ | BC18-like $\chi^2$ | BC15-like $\chi^2$ | BC15-like weight |
|---|---|---|---|---|
| M21–M24 (growth-side inhibition only) | 6–8 | 522–548 | 99–128 | 0 |
| M25–M27 (death + growth-side inhibition) | 8–9 | 183–190 | 104–105 | 0 |
| M28 ($K_{PI_2}$) | 7 | 164 | **21.1** | **0.85** |
| M29 ($K_{SI_2}$) | 7 | 548 | 61.5 | 0 |
| M30 ($K_{SI_2}$ + $K_{PI_2}$) | 8 | 163 | 20.8 | 0.14 |
| M31 ($k_d$ + $K_{SI_2}$ + $K_{PI_2}$) | 9 | **27.9** | 21.4 | 0.01 |

* **BC18-like needs both death and product inhibition of conversion.** Only M31 has both. Mechanisms with
  product inhibition but no death (M28, M30) can't explain the slow biomass decline, and those with death
  but no conversion inhibition (M25–M27) can't explain the slowing xylitol production. Every alternative is
  at least 135 $\chi^2$ units worse, so the weight is 1.00.
* **BC15-like is a nested tie.** M28, M30 and M31 fit equally well ($\chi^2 \approx 21$), because M30 and
  M31 contain M28: switching their extra terms off reproduces it. BIC charges 3.95 per extra parameter, so
  the simplest, M28, wins (0.85), and M30 stays on the shortlist (0.14, one parameter more, 3.7 BIC units
  behind). This is the same M28/M30 shortlist the paper found for the real BC15 run.
* **Product inhibition of conversion is required in BC15-like.** M29 (xylose inhibition only) is 40 $\chi^2$
  units behind, and the growth-side mechanisms are about 80 behind.

**Parameters of the true mechanism.**

| parameter | BC18-like: true / refined | BC15-like: true / refined |
|---|---|---|
| $\mu_{max,1}$ | 0.32 / 0.36 | 0.30 / 0.34 |
| $Y_{XS_1}$, $Y_{PS_2}$ | 0.62, 1.24 / 0.57, 1.29 | 0.70, 1.30 / 0.68, 1.23 |
| $k_d$ | 0.018 / 0.015 | – |
| $K_{PI_2}$ | 8.52 / 8.47 | 10.0 / 33.1 |
| $\mu_{max,2}$, $K_P$ | 5.54, 40 / 2.17, 8.3 | 4.0, 60 / 0.64, 18.2 |
| $K_{SI_2}$ | 23.4 / 250 (upper bound) | – |
| $K_{S_1}$ | 0.17 / 1.0 (upper bound) | 0.10 / 1.0 (upper bound) |

* **Well recovered:** yields, $\mu_{max,1}$, $k_d$, and $K_{PI_2}$ in BC18-like.
* **Not recovered individually: the conversion constants.** $\mu_{max,2}$, $K_P$, $K_{SI_2}$ and $K_{PI_2}$
  all enter one expression, $\mu_2$, and different combinations give the same $\mu_2(t)$ along the run. In
  BC18-like the fit even switches the Haldane term off ($K_{SI_2}$ at its bound) and compensates with smaller
  $\mu_{max,2}$ and $K_P$. In BC15-like, weaker product inhibition ($K_{PI_2}$ = 33) is offset by a lower
  maximum rate.
* **$K_{S_1}$ is invisible.** Glucose sits near 0 after the batch phase, below the 1 g/L noise.
* The fits are still good ($\chi^2$ = 28 and 21). The data constrain the *rates*, not every individual
  constant, which matches the paper's identifiability findings.
* **The starts agree for the plausible candidates** (M28, M30, M31 identical from both). For several wrong
  candidates in BC15-like, the NODE start reached a much better fit than the centre start (for example M21:
  105 vs 187; M29: 62 vs 121). Keeping the better of the two gives every candidate a fair chance.

### 6.3 Robustness (outside the notebook)

The notebook shows one noise realisation per run. The same pipeline was re-run over 6 noise realisations.
It selected the true mechanism in **22 of 24** Case 1 runs; both misses were Monod read as Contois at
$w \approx 0.56$, the near-tie described above. It selected the true mechanism in **12 of 12** Case 2 runs:
BC18-like → M31 at weight 1.00, and BC15-like → M28 at weight 0.79–0.87 with M30 shortlisted.

**Run time** on one CPU core: about 20 s per Neural ODE in Case 1 and 40–60 s in Case 2. Refining all 11
candidates from two starts takes 45–65 s per run.

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
