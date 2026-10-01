# Neural ODEs for kinetic-mechanism identification

This folder applies a **Neural ODE** approach to the case studies of the kintrace paper. It is a starting
point, in two notebooks:

* [`notebooks/node_case_studies.ipynb`](notebooks/node_case_studies.ipynb) (**part 1**, §1–7): both case
  studies of the paper, with the paper's mechanism libraries.
* [`notebooks/node_case1_library_sr.ipynb`](notebooks/node_case1_library_sr.ipynb) (**part 2**, §8): Case 1
  with a larger, 11-mechanism library, a run from a law that is *not* in the library, and **symbolic
  regression** that proposes new rate laws from what the Neural ODE learned, for Case 1 and for the two
  Case 2 rates.

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

## 8. Part 2 — a larger Case 1 library and symbolic regression

Part 2 asks three questions. How does selection behave when the library grows? What happens when the
true law is not in the library at all? And can symbolic regression recover the two Case 2 rates?

### 8.1 The 11-mechanism library

The mass balance gains an optional death term:

$$\dot X = (\mu - k_d - D)\,X, \qquad \dot S = -\frac{\mu X}{Y_{XS}} + D\,(S_{in} - S), \qquad \dot V = F.$$

| rate law | $\mu(X,S)$ | in the library |
|---|---|---|
| Monod | $\mu_{max}\,\frac{S}{K_S+S}$ | with and without $k_d$ |
| Contois | $\mu_{max}\,\frac{S}{K_cX+S}$ | with and without $k_d$ |
| Haldane | $\mu_{max}\,\frac{S}{K_S+S+S^2/K_I}$ | with and without $k_d$ |
| biomass inhibition | $\mu_{max}\,\frac{S}{K_S+S}\left(1-\frac{X}{X_{max}}\right)$ | with and without $k_d$ |
| Tessier | $\mu_{max}\left(1-e^{-S/K_S}\right)$ | without $k_d$ |
| Moser | $\mu_{max}\,\frac{S^n}{K_S+S^n}$ | without $k_d$ |
| Aiba | $\mu_{max}\,\frac{S}{K_S+S}\,e^{-S/K_I}$ | without $k_d$ |

The parameter ranges are kintrace's, extended with $k_d \in [0.005, 0.1]$ h⁻¹, $n \in [1, 3]$, Moser
$K_S \in [0.1, 25]$ and Aiba $K_I \in [5, 50]$ g/L.

**Data.** Every run uses the part-1 design (a 6 h batch phase, then three rising feed rates; 49 samples;
3 % noise). The true parameters lie inside the ranges: $\mu_{max} = 0.5$ (0.6 for Haldane and Aiba),
$Y_{XS} = 0.5$ and $k_d = 0.04$ h⁻¹ for the death variants. There are 11 runs, one per mechanism, plus
the hidden run of §8.3.

**Changes to the Neural ODE.** The Neural ODE also learns $k_d \ge 0$, and three training details changed:

* **Robust initial state.** It starts from the median of the first three samples, because with noise clipped
  at zero the first biomass sample can read 0 g/L, and a culture started at zero never grows.
* **Range-weighted loss.** The loss weights each channel by its **range** ($\max - \min$) instead of its
  maximum, matching the noise model (3 % of range). For a run whose substrate never falls near zero, the
  maximum is about twice the range, and the old weighting under-fitted the substrate.
* **Learning rate.** It is $10^{-2}$ instead of $5\times10^{-3}$.

### 8.2 Symbolic regression on the learned rates

The library can only choose among laws someone wrote down. Symbolic regression instead *proposes* a law
from the learned rate $\hat\mu(t)$ and states $\hat X(t), \hat S(t)$.

**A linear trick for rational laws.** Multiplying a rate law by its denominator makes it linear in its
coefficients. For example, Haldane becomes

$$\mu S \;=\; \mu_{max}\,S \;-\; K_S\,\mu \;-\; \tfrac{1}{K_I}\,\mu S^2 .$$

So we regress $\mu S$ on a small dictionary of terms,

$$\mu S = \sum_k c_k\,\theta_k(X, S, \mu), \qquad
\theta \in \{\,S,\ SX,\ S^2,\ \mu,\ \mu X,\ \mu S^2,\ \mu SX\,\},$$

and read the rate law back as a ratio. The terms without μ form the numerator $N$, and those with μ form
the denominator $D$:

$$\mu = \frac{N}{D}, \qquad N = \sum_{\text{plain}} c_k\,\theta_k, \qquad
D = S - \sum_{\mu\text{-terms}} c_k\,\theta_k/\mu.$$

This dictionary expresses Monod ($S$, $\mu$), Contois ($S$, $\mu X$), Haldane ($S$, $\mu$, $\mu S^2$) and
biomass inhibition ($S$, $SX$, $\mu$) exactly, and the hidden law with $S$, $SX$, $\mu$, $\mu S^2$. Tessier,
Moser and Aiba (exponentials, powers) can only be approximated.

**Sparsity by best-subset search.** With 7 terms there are only 127 subsets, so every subset of up to 4
terms is tried. SINDy's thresholded least squares, tried first, approximates this search for large
dictionaries, but here it returned meaningless models for the Haldane and hidden runs.

**Fitting each law.** Every subset defines a law $\mu = N/D$ whose coefficients are fitted to $\hat\mu$
directly by least squares, like the library laws in §3:

$$\min_{a,\,b}\ \sum_t \left(\frac{\sum_{\text{plain}} a_k\,\theta_k}{S + \sum_{\mu\text{-terms}} b_k\,\theta_k/\mu} - \hat\mu\right)^2,
\qquad b_k \ge 0 .$$

The linear form is only used to *write* the laws. Fitting it directly by linear least squares weights each
point by $D^2$, so the late part of a run, where $D$ is large, dominates; on the Case 2 conversion rate
that chose physically impossible laws.

**Validity rules.** A candidate law is kept only if it is physically sensible:

* **It has a numerator.** A model made only of μ-terms (for example $\mu S = a\mu + b\mu S^2$) reduces to
  a relation among $S$ values alone and "fits" without describing any rate.
* **Its denominator terms are positive** ($b_k \ge 0$), like $K_S$, $S^2/K_I$ or $K_c X$. Unconstrained
  fits produce denominators such as $S - 1.2 - 0.06\,P$, that is product *activation* and a rate that
  blows up.
* **The rate vanishes smoothly when the substrate runs out.** $D > 0$ at $S = 0$, the same prior as the
  Neural ODE's factor $S/(S+K)$ (§2). Otherwise laws such as $\mu = a + bX$, where $S$ cancels, switch on
  abruptly at any $S > 0$.
* **The rate is non-negative** along the run.

For each size, the **three** best valid laws are kept (up to 9 per run). Several laws fit $\hat\mu$ almost
equally well, and the ranking at the rate level is too uncertain to trust a single one.

**The data decide, not the regression.** The regression sees only one trajectory, so several term sets fit
$\hat\mu$ about equally well (the confounding of §7). Each candidate, with and without $k_d$, therefore
becomes an ODE model. Its parameters are the coefficients $c_k$ (sign-preserving bounds
$[0.1\,c_k,\ 10\,c_k]$), $Y_{XS}$ and optionally $k_d$. It is refined against the data (§4) and ranked by
BIC (§5) together with the 11 library mechanisms.

Some candidates are *the same law* as a library mechanism written as terms: $\{S,\ \mu X\}$ is exactly
Contois and $\{S,\ \mu\}$ is exactly Monod. Such a candidate ties with its library twin (ΔBIC ≈ 0) and the
two share the BIC weight.

### 8.3 The hidden run

One run is generated from a law that combines two library ingredients but is not itself in the library:

$$\mu = \mu_{max}\,\frac{S}{K_S + S + S^2/K_I}\left(1 - \frac{X}{X_{max}}\right),
\qquad \mu_{max} = 0.6,\ K_S = 1,\ K_I = 20,\ X_{max} = 30.$$

The setting was chosen so that both ingredients show in the data: Haldane-type acceleration while the batch
substrate is used up, then a biomass plateau with substrate building up. With $X_{max} = 15$, the first
setting tried, the substrate never ran low, plain biomass inhibition fitted as well as the true law, and
there was nothing to discover.

### 8.4 Results

**Library selection** (BIC weights; shortlist = $w > 0.1$):

| run | selected (weight) | shortlist |
|---|---|---|
| Monod | Contois (0.40) | Contois, Tessier 0.28, Moser 0.12 (Monod 0.09) |
| Monod+$k_d$ | Contois+$k_d$ (0.47) | Contois+$k_d$, **Monod+$k_d$ 0.46** |
| Contois | **Contois** (0.81) | Contois |
| Contois+$k_d$ | **Contois+$k_d$** (0.80) | Contois+$k_d$, biomass inh.+$k_d$ 0.20 |
| Haldane | Contois (0.87) | Contois |
| Haldane+$k_d$ | Aiba (0.82) | Aiba, **Haldane+$k_d$ 0.18** |
| biomass inh. | **biomass inh.** (0.90) | biomass inh., biomass inh.+$k_d$ 0.10 |
| biomass inh.+$k_d$ | **biomass inh.+$k_d$** (1.00) | biomass inh.+$k_d$ |
| Tessier | **Tessier** (0.35) | Tessier, Monod 0.35, Contois 0.10 |
| Moser | **Moser** (0.70) | Moser, Contois 0.26 |
| Aiba | **Aiba** (0.48) | Aiba, Haldane 0.44 |
| hidden | biomass inh. (0.91) | biomass inh. |

* **The true mechanism is ranked first in 7 of 11 runs and shortlisted in 9 of 11.** With 4 mechanisms
  (part 1) all four were ranked first. A larger library means more near-ties and lower weights.
* **The misses are genuine ambiguities.**
  * Monod, Contois, Tessier and Moser differ mainly at low $S$, which the fed phase visits below the noise.
  * On the Haldane run, $X$ and $S$ rise together, so a falling μ can be blamed on either, and Contois
    fits as well.
  * Haldane+$k_d$ and Aiba are two forms of substrate inhibition.
* **Death is detected.** Every death run has its +$k_d$ mechanism selected or shortlisted, and no
  death-free run selects one. The learned $k_d$ is 0.031–0.032 h⁻¹ for three of the four death runs (true
  0.04; 0.012 for Haldane+$k_d$), and 0.006–0.016 for the death-free runs.

**Symbolic regression** (ΔBIC = best symbolic-regression law − best library mechanism; negative means the
new law wins):

| run | ΔBIC | best law found by symbolic regression |
|---|---|---|
| Monod, Monod+$k_d$, Contois, Contois+$k_d$, Haldane | −0.0 to 0.0 | Contois $\{S, \mu X\}$, e.g. $0.47\,S/(S + 0.91\,X)$ on the Contois run (true $K_c = 1$): the library law rediscovered |
| biomass inh.+$k_d$ | 0.0 | biomass inhibition $\{S, SX, \mu\}$, rediscovered |
| Tessier | 0.0 | Monod $\{S, \mu\}$, the same tie as in the library |
| Aiba | +0.1 | a new rational law that ties; the data barely constrain the shape |
| biomass inh., Moser, Haldane+$k_d$ | +4.5, +2.0, +39 | – |
| **hidden** | **−28.1** | $0.666\,\frac{S\,(1 - X/30.5)}{1.96 + S + S^2/16.3}$ (weight 0.59) |

* **No false discoveries.** On all 11 library runs, the best symbolic-regression law ties with or loses to
  the best library mechanism. Most ties are the library law itself, rediscovered in term form.
* **The missing law is found.** On the hidden run the new law wins by ΔBIC ≈ 28 ($\chi^2$ = 73 vs 106).
  Its terms $\{S, SX, \mu, \mu S^2\}$ are exactly those of the true law $0.6\,S(1 - X/30)/(1 + S + S^2/20)$,
  and its constants are close: $X_{max} \approx 30.5$ (true 30), $K_I \approx 16$ (20) and
  $K_S \approx 2.0$ (1.0, the least identifiable one, as in part 1).
* **A goodness-of-fit test alone would not have flagged the hidden run.** The best library mechanism reaches
  $\chi^2 = 106$ for $n = 98$, which is within the noise band ($\chi^2 \approx n \pm 3\sqrt{2n}$). The
  symbolic-regression proposal is what exposes the missing law.
* **Symbolic regression also gives readable laws on library runs**, such as a logistic law with
  $X_{max} \approx 19.7$ (true 20) on the biomass-inhibition run.

### 8.5 Case 2: symbolic regression on μ1 and μ2

The two virtual fermentations of §6.2 (BC18-like from M31, BC15-like from M28) are refitted with the
part-1 Neural ODE, and the 11 library mechanisms are refined as before. Each rate gets its own dictionary:

$$\mu_1 S_1 = \sum_k c_k\,\theta_k,\qquad \theta \in \{S_1,\ \mu_1,\ \mu_1 S_2^2,\ \mu_1 P,\ \mu_1 S_1 P\},$$

$$\mu_2 S_2 = \sum_k c_k\,\theta_k,\qquad \theta \in \{S_2,\ \mu_2,\ \mu_2 S_2^2,\ \mu_2 P,\ \mu_2 S_2 P,\ \mu_2 S_2^2 P\}.$$

Together they cover every term of M21–M31: Monod growth, cross-inhibition by xylose and product inhibition
of growth for μ1; Haldane and product inhibition of conversion for μ2. For example, M28's conversion law
becomes

$$\mu_2 S_2 = \mu_{max,2}\,S_2 - K_P\,\mu_2 - \tfrac{K_P}{K_{PI_2}}\,\mu_2 P - \tfrac{1}{K_{PI_2}}\,\mu_2 S_2 P .$$

The best valid law of each size for μ1 (up to 3 terms) and μ2 (up to 5 terms) are combined into full ODE
models (Eq. 8), with and without $k_d$: 12 models for BC18-like and 16 for BC15-like. They are refined
against the data and ranked by BIC with the library.

| run | best library | best found by symbolic regression | ΔBIC (SR − library) |
|---|---|---|---|
| BC18-like (true M31) | M31, $\chi^2$ = 27.9 (weight 0.13) | Monod μ1, death, $\mu_2 = \frac{0.057\,S_2}{1 + 0.0009\,S_2 + 0.0045\,P S_2}$, $\chi^2$ = 29.6 (weight 0.41) | −2.2 |
| BC15-like (true M28) | M28, $\chi^2$ = 21.1 (weight 0.51) | Monod μ1, $\mu_2 = \frac{0.0237\,S_2}{1 + 0.058\,S_2 + 0.0018\,P S_2}$, $\chi^2$ = 22.6 (weight 0.24) | +1.5 |

* **The key ingredients are recovered in both runs.** μ1 is found to be Monod, as in every library model.
  Death is included where the run has it (BC18-like) and left out where it doesn't (BC15-like). Product
  inhibition of conversion appears as a $P\,S_2$ term in the denominator of μ2. In BC15-like that term's
  coefficient is 0.0018, against 0.0017 in the true law
  $\mu_2 = 0.067\,S_2/(1 + 0.017\,S_2 + 0.1\,P + 0.0017\,P S_2)$.
* **The found laws tie with the true mechanism.** On BC18-like, the found law has one parameter fewer than
  M31 and ranks first by ΔBIC = −2.2; on BC15-like, M28 stays first by 1.5. Differences this small are not
  evidence either way. With 13 samples per species, compact forms such as these and the textbook laws
  describe the data equally well, and the found rates follow the true μ1 and μ2 closely (notebook figure).

### 8.6 What was learned in part 2

* **Symbolic regression inherits the confounding.** It reads the learned rates along one trajectory, so on
  the Haldane run it proposes a Contois-type law, just as the library does. Only runs in which $X$ and $S$
  vary independently can fix this.
* **The dictionary limits what can be found.** Only rational laws built from the 7 terms can be expressed.
* **The Neural ODE's $\chi^2$ is only a rough reference.** A few hundred Adam steps leave it above a
  refined mechanism in several runs (for example 165 vs 106 on the hidden run), so it cannot serve as a
  strict lack-of-fit benchmark.
* **Symbolic regression needs physical rules to be useful.** Fitting the laws in rate space, positive
  denominator terms, and a rate that vanishes when the substrate runs out (§8.2) were all needed; without
  them, the best-fitting candidates were meaningless or physically impossible.
* **Each run has one noise realisation.** Unlike part 1 (§6.3), these results were not repeated over
  several noise draws.
* **Implementation note.** The ODE solver (LSODA) prints its warnings from Fortran, which bypasses Python,
  so the notebook silences them by redirecting the process's stdout while the solver runs.

## 9. Running it

```bash
pip install torchdiffeq sympy           # on top of kintrace's requirements.txt
cd kintrace/node/notebooks
jupyter notebook node_case_studies.ipynb       # part 1, about 5 min on a CPU
jupyter notebook node_case1_library_sr.ipynb   # part 2, about 15 min on a CPU
```

**Next steps:**
1. Fit several runs jointly with shared kinetics and different feeds or initial substrate, so that $X$
   and $S$ vary independently. This breaks the confounding that limits both library selection and symbolic
   regression.
2. Plug in the real BC runs through `xylitol_case_study/realdata.py::load_experiments`.
3. Warm-start from the LSTM estimates.
4. Reach non-rational laws: add exponential terms to the dictionary, or use genetic-programming symbolic
   regression (for example PySR).
5. Add accepted symbolic-regression laws to the library for later runs.
