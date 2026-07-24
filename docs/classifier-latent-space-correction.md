# Classifier-Based Latent Space Correction for Normalizing Flows

> Write-up of discussions on correcting normalizing flow samples via density-ratio estimation and resampling in latent space. Target application: calorimeter simulation (LEMuRS, $\ll 10^4$ dimensions, conditional on incident energy).

---

## 1. Problem Statement

Normalizing flows provide a tractable change-of-variables density model:

$$p(x) = \pi(z) \left|\det J_F(z)\right|^{-1},\qquad z = F^{-1}(x)$$

where $\pi(z)$ is a simple base distribution (e.g., a standard Gaussian) and $F$ is an invertible neural network. When trained via maximum likelihood or reverse KL, flows learn to push forward the base distribution to match the target $p^*(x)$, or equivalently to map the target back to the base: $q(z) = F^{-1}_{\#}p^*(x)$ should approach $\pi(z)$.

**The topology problem.** Because $F$ is a homeomorphism, the support of $p(x)$ has the same topological structure as that of $\pi(z)$. If the target has disconnected modes (or more generally a support structure differing from a Gaussian's), the flow must either draw density filaments between modes (degrading sample quality) or become numerically non-invertible (Cornish et al., 2020).

For calorimeter simulation at $\sim 6500$ dimensions with conditioning on incident energy, this mismatch manifests as rare but systematic artifacts in generated showers.

---

## 2. Background: Resampled Base Distributions (LARS)

Stimper et al. (2022) proposed using Learned Accept/Reject Sampling (LARS; Bauer & Mnih, 2019) as a base distribution for normalizing flows. A learned acceptance function $a_\phi: \mathbb{R}^d \to [0,1]$ reweights a Gaussian proposal $\pi(z)$:

$$p_\phi(z) = \frac{\pi(z) a_\phi(z)}{Z},\qquad Z = \int \pi(z) a_\phi(z) \, dz$$

The base density enters the flow's log-likelihood as:

$$\log p(x) = \log\pi(z) + \log\left(\alpha_T + (1-\alpha_T)\frac{a_\phi(z)}{Z}\right) - \log|\det J_F(z)|$$

where $\alpha_T = (1-Z)^{T-1}$ arises from truncating rejection sampling after $T$ proposals.

### 2.1 The Curse of Dimensionality in LARS (Section 3.3)

The normalization constant $Z$ and its gradient must be estimated via Monte Carlo:

$$Z \approx \frac{1}{S}\sum_{s=1}^S a_\phi(z_s),\qquad z_s \sim \pi(z)$$

The relative variance of this estimator is:

$$\text{RelVar}(\hat Z) = \frac{\text{Var}_\pi[a_\phi]}{S\,Z^2} \lesssim \frac{1}{S\,Z}$$

where we used $a_\phi \in [0,1] \Rightarrow \text{Var}(a_\phi) \le \mathbb{E}[a_\phi] = Z$.

**Why this explodes with dimension.** $Z$ is the acceptance rate of a rejection sampler. In high dimensions, for the acceptance function to carve out the target's topological structure from a Gaussian proposal, it must suppress mass in most of the proposal's support. The acceptance rate decays exponentially: $Z \sim e^{-c d}$. Achieving fixed *relative* error then requires $S \gtrsim 1/Z \sim e^{c d}$.

This is not a contradiction with Monte Carlo's dimension-free $O(1/\sqrt{S})$ convergence rate: the rate is indeed dimension-free, but the constant (here $\text{Var}(a_\phi)/Z^2 \sim 1/Z$) grows exponentially with $d$. Rejection sampling between two high-dimensional distributions is cursed regardless of how the acceptance function is learned.

### 2.2 The Paper's Mitigation and Its Limits

Stimper et al. (2022) factorize the base into $<100$-dimension factors via the multiscale architecture:

- Squeeze feature maps until $H \times W < 100$
- Treat each channel as an independent factor
- Share a neural network per level with multiple outputs

**Ceiling reached:** the largest base in their experiments is 3072 dims (CIFAR-10 Glow). At $\sim 6500$ dims you would need $\sim 100$+ factors. And crucially, per-factor independent resampling can only model *independent per-factor modes* — correlated global mode structure (topological sectors, collective variable switching) cannot be represented, returning you to the original topology problem.

**Empirical gains shrink with capacity:** on CIFAR-10, gains are $0.004\text{--}0.007$ bits/dim, vanishing at 32 layers/level. Big wins are low-dimensional (2D toys, alanine dipeptide at 60 dims).

---

## 3. Proposed Method: Latent-Space Density Ratio via Classifier

Instead of learning an acceptance function directly (LARS), learn a density ratio between the pushed-back target and the base, and use it for post-hoc correction.

### 3.1 Overview

| Step | Description |
|------|-------------|
| 1 | Train a conditional normalizing flow $F_\theta$ on the target calorimeter data |
| 2 | Train a classifier $D_\psi(z, c)$ to distinguish $z = F^{-1}(x)$ (pushed-back target) from $z \sim \pi = \mathcal{N}(0,I)$, conditioned on incident energy $c$ |
| 3 | Recover $r(z,c) = q(z|c) / \pi(z)$ from the classifier output |
| 4 | Use $r$ for correction — either via rejection sampling or HMC in latent space |

### 3.2 Density Ratio from a Classifier

Train a binary classifier $D_\psi(z,c) \in [0,1]$ with cross-entropy loss on:

- **Class 1:** $z = F^{-1}(x)$ where $x \sim \text{target}$ (pushed-back data)
- **Class 0:** $z \sim \pi(z) = \mathcal{N}(0,I)$ (pure Gaussian)

Let $\rho_1 = n_1/(n_1+n_0)$ and $\rho_0 = n_0/(n_1+n_0)$ be the class fractions. At optimum:

$$D(z,c) = \frac{\rho_1\,q(z|c)}{\rho_1\,q(z|c) + \rho_0\,\pi(z)}$$

Solving for the ratio:

$$\frac{q(z|c)}{\pi(z)} = \frac{D(z,c)}{1 - D(z,c)} \cdot \frac{\rho_0}{\rho_1} = \frac{D(z,c)}{1 - D(z,c)} \cdot \frac{n_0}{n_1}$$

**In practice, work in logit space** to avoid sigmoid saturation:

$$r(z,c) = \exp\left(\text{logit}(z,c) + \log\frac{n_0}{n_1}\right)$$

```python
def density_ratio(logits, n_pos, n_neg):
    """Compute q/pi from classifier logits."""
    log_prior_correction = math.log(n_neg / n_pos)
    return torch.exp(logits + log_prior_correction)
```

Since $\pi$ samples are free (Gaussian draws), you can present balanced batches and skip the correction: $\rho_1 = \rho_0 = 1/2 \Rightarrow r = D/(1-D)$ directly.

### 3.3 Why This is Attractive

If $r(z,c)$ is exact and the bound $M \ge \sup_z r(z,c)$ is finite, then:

1. Accept $z \sim \pi$ with probability $r(z,c)/M$
2. The accepted $z$ follow $q(z|c)$ — the *true* pushed-back target
3. $F(z)$ follows the *exact* target distribution, **regardless of flow quality**

The flow's residual mismatch becomes a two-sample discrimination problem — the thing neural nets do best.

### 3.4 The Acceptance-Rate Problem (Rejection Sampling Variant)

If using rejection sampling:

$$\text{acceptance rate} = \frac{1}{M},\qquad M = \sup_z \frac{q(z|c)}{\pi(z)} = \exp\big(D_\infty(q\,\|\,\pi)\big)$$

where $D_\infty$ is the Rényi-$\infty$ divergence. The acceptance rate is $\alpha = \exp(-D_\infty)$. Since $D_\infty \ge \text{KL}$ and $\text{KL}(q\|\pi)$ grows ~linearly with $d$ for fixed-quality flows, $\alpha \sim e^{-c d}$ — **the same exponential curse as LARS** (§2.1), relocated from $Z$ to $M$.

The density ratio is learned more stably than LARS (supervised, decoupled from flow training), but rejection sampling itself is what's cursed, no matter how the acceptance function is obtained.

### 3.5 Improving M-Estimation (Rejection Sampling Only)

If one insists on rejection sampling, the following methods improve estimation of $M$:

#### Class-prior reweighting

With $n_0 \gg n_1$ ($\pi$-samples free), the prior correction $n_0/n_1$ is exact asymptotically. Weighted BCE generalizes to effective priors.

#### Defensive / heavier-tailed proposal

Instead of sampling the negative class from $\pi$, sample from a broadened proposal $\pi_0$ (inflated covariance or Student-t) and importance-weight:

$$w(z) = \frac{\pi(z)}{\pi_0(z)}$$

This reaches tail regions with far fewer samples. The correction formula becomes:

$$\frac{q}{\pi} = \frac{D}{1-D} \cdot \frac{w \cdot \rho_0}{\rho_1}$$

#### Radial stratification

For a $D$-dim Gaussian, $\|z\|^2 \sim \chi^2_D$ in closed form. Stratify sampling over radius bands, oversample rare large-radius shells, and reweight by the known $\chi^2$ density.

#### GPD / extreme value theory

Fit a Generalized Pareto Distribution to the upper tail of estimated ratio values (peaks-over-threshold). Extrapolate $M$ from the fitted tail rather than trusting the empirical maximum.

#### Tail calibration

Compute calibration in the top ratio deciles. Ensemble/bootstrapped classifiers give uncertainty; worst-case bin overestimation biases $M$ upward.

**Caveats:** these methods all improve the *estimator* of $M$, but $M$ is an extrapolation quantity — beyond data coverage, $M$ is whatever the classifier's inductive bias says. And even with a perfect $M$, $\alpha = 1/M$ is still exponentially small at $\sim 6500$ dims.

---

## 4. HMC for Latent Space Refinement

Hamiltonian Monte Carlo replaces rejection sampling and sidesteps the $M$-estimation problem entirely.

### 4.1 Mechanics

The unnormalized log-density of the target in latent space is:

$$\log q(z|c) = \log \pi(z) + \log r(z,c) = -\frac{1}{2}\|z\|^2 + \text{logit}(z,c) + \text{const}$$

The gradient for HMC is:

$$\nabla_z \log q(z|c) = -z + \nabla_z\,\text{logit}(z,c)$$

The classifier's logit (not sigmoid output) and its gradient via autograd provide everything HMC needs.

```python
def potential(z, c, logit_net):
    """Unnormalized negative log probability: U = -log q(z|c)."""
    log_pi = 0.5 * (z ** 2).sum(dim=-1)          # -log p(z) up to constant
    log_r = logit_net(z, c)                         # classifier logit(s)
    return log_pi - log_r                           # U(z) = -[log pi + log r]

def grad_potential(z, c, logit_net):
    """Gradient of U w.r.t. z."""
    z.requires_grad_(True)
    u = potential(z, c, logit_net)
    grad = torch.autograd.grad(u.sum(), z)[0]
    return grad
```

### 4.2 Why HMC Sidesteps the $D_\infty$ Curse

| Criterion | Rejection Sampling | HMC |
|-----------|-------------------|-----|
| Requires global bound $M$ | Yes — the killer at high $d$ | No |
| Acceptance rate | $\exp(-D_\infty) \sim e^{-c d}$ (fixed) | Tunable via $\varepsilon$, $L$ (typical $60$–$90\%$) |
| Exactness with perfect $r$ | Exact i.i.d. samples | Asymptotically exact (discretization bias controlled by $\varepsilon$) |
| With imperfect $r$ | Silently biased | Biased but **diagnosable** (ESS, $\hat R$, trace plots) |
| Multimodality | N/A (i.i.d.) | Can get trapped (mixing time grows with barrier height) |
| Parallelization | Embarrassingly (independent proposals) | Parallel chains, serial within chain |

The critical difference: HMC's per-sample cost grows with the *typical* log-ratio gradient, not the *sup*. At $\sim 6500$ dims, rejection's $M$ is dominated by one unlucky tail point; HMC only needs the gradient to be reasonable along typical trajectories.

### 4.3 Initialization

Push the training data through $F^{-1}$:

$$z_i^{\text{init}} = F^{-1}(x_i),\quad x_i \sim \text{target}$$

Each $z_i^{\text{init}}$ is a perfect warm start for a chain. This is one of the rare settings where MCMC initialization is genuinely solved by the model. Run one chain per data subsample (or per incident-energy bin) for diversity and mixing diagnostics.

### 4.4 Practical Tuning for $\sim 6500$ Dimensions

**Step size.** Standard scaling: $\varepsilon \propto d^{-1/4}$. For $d \approx 6500$, $\varepsilon \approx 0.01\text{--}0.05$ is typical.

**Trajectory length.** Fixed $L \sim 50\text{--}200$ leapfrog steps. NUTS (No-U-Turn Sampler) auto-tunes $L$ but can misbehave in very high $d$.

**Classifier smoothness.** HMC trusts $\nabla_z \log r$. The classifier must be smooth:

```python
class SmoothClassifier(nn.Module):
    def __init__(self, d_in, d_context, d_hidden=256, n_layers=3):
        super().__init__()
        layers = []
        in_dim = d_in + d_context
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(in_dim, d_hidden),
                nn.Softplus(),  # smooth activation
                nn.utils.spectral_norm_wrapper(...)
            ])
            in_dim = d_hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z, c):
        h = torch.cat([z, c], dim=-1)
        return self.net(h)  # logit, no sigmoid
```

**Gradient clipping** as a safety net:

```python
# During HMC leapfrog steps:
grad = torch.clamp(grad, -max_grad, max_grad)
```

### 4.5 Diagnostics

Standard MCMC diagnostics apply:

- **Acceptance rate:** target $0.6$–$0.9$; tune $\varepsilon$ accordingly
- **Effective sample size (ESS):** accounts for autocorrelation; target $\text{ESS}/N > 0.1$
- **$\hat R$ (Gelman-Rubin):** across chains, target $\hat R < 1.01$
- **Trace plots:** visual check for drift or poor mixing

If a single chain gets stuck (multimodal $q$ with large barriers):
- Run more chains from diverse $F^{-1}(x_i)$ initializations
- Use parallel tempering or replica exchange
- Or accept that rejection sampling wouldn't work there either

### 4.6 HMC Sampling Loop (Pseudocode)

```python
def hmc_sample(init_z, c, logit_net, eps, L, n_samples):
    """Run HMC to sample from q(z|c)."""
    z = init_z.clone()
    samples = []
    for _ in range(n_samples):
        # Momentum refresh
        p = torch.randn_like(z)
        z_prop, p_prop = z.clone(), p.clone()

        # Leapfrog integration
        p_prop = p_prop - 0.5 * eps * grad_potential(z_prop, c, logit_net)
        for _ in range(L - 1):
            z_prop = z_prop + eps * p_prop
            p_prop = p_prop - eps * grad_potential(z_prop, c, logit_net)
        z_prop = z_prop + eps * p_prop
        p_prop = p_prop - 0.5 * eps * grad_potential(z_prop, c, logit_net)

        # MH accept/reject
        u_current = potential(z, c, logit_net) + 0.5 * (p ** 2).sum()
        u_prop = potential(z_prop, c, logit_net) + 0.5 * (p_prop ** 2).sum()
        if torch.rand(1) < torch.exp(u_current - u_prop):
            z = z_prop
        samples.append(z.clone())
    return torch.stack(samples)
```

---

## 5. Prior Art

### 5.1 Latent Space Refinement (LaSeR / ELSA)

**Winterhalder, Bellagente & Nachman (2021)**, *Latent Space Refinement for Deep Generative Models*, arXiv:2106.00792. NeurIPS DGMs Workshop 2021.

- Train a classifier **in data space** to distinguish real vs. generated data
- Pull weights back to latent space via $F^{-1}$
- Use HMC ("LaSeR protocol") to refine samples in latent space
- Demonstrated on normalizing flows and GANs

**Nachman & Winterhalder (2023)**, *ELSA — Enhanced latent spaces for improved collider simulations*, arXiv:2305.07696. Eur. Phys. J. C 83, 843.

- Extended LaSeR to collider physics (W + jets matrix element surrogate simulations)
- Includes reweighting, pre-processing, latent space refinement, and augmented normalizing flows
- Code: `github.com/ramonpeter/elsa`
- Sub-percent precision across wide phase space

**How our proposal differs.** The LaSeR/ELSA papers train the classifier in *data space* and pull weights back. Our proposal trains the classifier **directly in latent space** ($F^{-1}(x)$ vs. $\pi$). This is cleaner when one distribution ($\pi$) is analytically known: the classifier's job is one-sample density estimation against a known reference, rather than two-sample. There is no Jacobian pullback through $F^{-1}$, and gradient through the classifier is direct.

### 5.2 Discriminator Rejection Sampling

**Azadi, Olsson, Darrell, Goodfellow & Odena (2019)**, *Discriminator Rejection Sampling*, arXiv:1810.06758. ICLR 2019.

- Use a trained GAN discriminator as the acceptance function for rejection sampling in **data space**
- Under strict assumptions (optimal discriminator, tight bound $M$), samples are drawn from the true data distribution
- Practical algorithm (DRS) uses clipping and empirical $M$ estimation
- Improved Inception Score on ImageNet from 52.52 to 76.08

Our proposal is the latent-space analogue of DRS. The latent space is better motivated for flows because:
- One side of the ratio ($\pi$) is known analytically, unlike the GAN case where neither side is known
- Exactness does not depend on flow quality (accept in $z$-space, push forward)
- The pushback $F^{-1}(x)$ is fast for MAF architectures

### 5.3 Calorimeter-Specific Work

**Krause & Shih (2021)**, *CaloFlow: Fast and Accurate Generation of Calorimeter Showers with Normalizing Flows*, arXiv:2106.05285. Phys. Rev. D 107, 113003.

- First demonstration of normalizing flows for multi-channel calorimeter showers
- Uses MAF-based flows; introduces classifier-based evaluation metric
- Mentions classifier-based reweighting of generated events

**Favaro et al. (2025)**, *CaloDREAM — Detector response emulation via attentive flow matching*, SciPost Physics.

- Flow matching for detector response emulation
- Operates on CaloChallenge datasets (DS2: 6480 dims, DS3: 40500 dims)
- Uses classifier weights for evaluation


### ErnList of papers on calorimeter fast simulation with flows:
**
- **CaloMan** (2023) — multiscale MAF for calorimeter  simulation
- **CaloDREAM** (2025) — flow matching for detector response emulation

**Butter, Plehn et al.**, *How to understand limitations of generative networks*, SciPost Physics 2024.

- Systematic study of classifier-based evaluation and reweighting for generative models in HEP

### 5.4 Density Ratio Estimation

**Sugiyama, Suzuki & Kanamori (2012)**, *Density Ratio Estimation in Machine Learning*. Cambridge University Press.

- Comprehensive treatment of classifier-based, moment-matching, and direct density-ratio estimation
- The class-prior correction identity ($r = D/(1-D) \cdot n_0/n_1$) is standard
- Covers least-squares importance fitting (LSIF), unconstrained LSIF, KLIEP, and related methods

### 5.5 Alternative Topology Fixes for Normalizing Flows

**Cornish, Caterini, Deligiannidis & Doucet (2020)**, *Relaxing Bijectivity Constraints with Continuously Indexed Normalising Flows*, arXiv:1909.13833. ICML 2020.

- Replace single bijection with a family indexed by continuous variable $u$
- The model becomes an infinite mixture of bijections → no longer injective
- Train with ELBO using encoder $q(u|x)$
- Proven not subject to the same topological limitations as standard flows
- Stimper et al. compare against CIF-NSF as a baseline on UCI tabular data

**Huang, Dinh & Courville (2020)**, *Augmented Normalizing Flows: Bridging the Gap Between Generative Flows and Latent Variable Models*, arXiv:2002.07101.

- Inflate input space with auxiliary noise dimensions $e \sim \mathcal{N}(0,I_k)$
- Flow operates on $(x, e) \in \mathbb{R}^{d+k}$; marginal in $x$ can be arbitrarily complex
- Train with ELBO: $p(x)$ intractable but bound is trainable
- Proven to approximate a Hamiltonian ODE as a universal transport map

**Wu, Köhler & Noé (2020)**, *Stochastic Normalizing Flows*, arXiv:2002.06707. NeurIPS 2020.

- Interleave bijective flow layers with stochastic MCMC/Langevin steps
- Bottom-up (inference) and top-down (generative) paths are both defined
- ELBO training, exact importance weights without marginalizing over stochastic blocks
- Designed for Boltzmann generators; demonstrated on molecular systems

**Nielsen, Jaini, Hoogeboom, Winther & Welling (2020)**, *SurVAE Flows: Surjections to Bridge the Gap between VAEs and Flows*, arXiv:2007.02731. NeurIPS 2020.

- Unified framework: surjective (many-to-one) layers with tractable likelihood contributions
- Deterministic forward, stochastic inverse; each layer contributes an ELBO-style term
- Includes dequantization and augmented flows as special cases
- Naturally handles zero-inflated data (relevant for calorimeter voxel sparsity)

### 5.6 Original LARS

**Bauer & Mnih (2019)**, *Resampled Priors for Variational Autoencoders*, arXiv:1810.11428. AISTATS 2019.

- Introduced Learned Accept/Reject Sampling (LARS)
- Applied as expressive prior for VAEs; trained jointly via ELBO
- Reported failure to beat factorized Gaussian prior when fully factorized
- Largest prior: 100 dimensions

**Stimper, Schölkopf & Hernández-Lobato (2022)**, *Resampling Base Distributions of Normalizing Flows*, arXiv:2110.15828. AISTATS 2022.

- Applied LARS as base distribution for normalizing flows
- Derived gradient estimators for both maximum likelihood and KL divergence training
- Introduced multiscale factorization to scale to 3072 dims (CIFAR-10 Glow)
- Demonstration on 2D densities, tabular data, image generation, and Boltzmann generators (alanine dipeptide, 60 dims)

---

## 6. Recommended Approach for Calorimeter Simulation

### 6.1 Diagnostic Phase (First — No New Training)

Before building any sampler, audit the existing trained conditional MAF:

1. **Train classifier** $D(z,c)$ on $F^{-1}(x_{\text{data}})$ vs. $\pi$ (conditional on $c$)
2. **Compute per-$c$ AUC:** if $\text{AUC} \approx 0.5$ everywhere, the flow is already good; correction is unnecessary
3. **Compute per-$c$ ratio ESS:** $\text{ESS} = (\sum w_i)^2 / \sum w_i^2$ where $w_i = r(z_i, c_i)$. If ESS is high ($>0.9$), the flow needs no base correction
4. **Where ESS collapses:** you've localized the failure region — for targeted fixes (more capacity, augmented dims, mixture base) rather than global correction

```python
def diagnostic(flow, classifier, data_loader):
    """Audit a trained flow via classifier in latent space."""
    for x, c in data_loader:
        z = flow.inverse(x, context=c)           # pushed-back data
        with torch.no_grad():
            logits_data = classifier(z, c)
            z_pi = torch.randn_like(z)
            logits_pi = classifier(z_pi, c)

        # AUC via rank statistics
        scores = torch.cat([logits_data.squeeze(), logits_pi.squeeze()])
        labels = torch.cat([torch.ones_like(logits_data), torch.zeros_like(logits_pi)])
        auc = compute_auc(scores, labels)

        # Ratio and effective sample size
        log_r = logits_data.squeeze() + math.log(n_pi / n_data)
        w = torch.exp(log_r)                     # importance weights
        ess = w.sum()**2 / (w**2).sum()

        print(f"c={c.item():.1f}: AUC={auc:.3f}, ESS/n={ess/len(w):.3f}")
```

### 6.2 Correction Phase (If Diagnostic Shows Mismatch)

**Option A: HMC in latent space** (recommended):
- Train smooth classifier with spectral norm + gradient penalty
- Initialize HMC chains from $F^{-1}(x_i)$
- Tune $\varepsilon$, $L$ on validation set
- Monitor ESS, $\hat R$, acceptance rate
- Sampling cost: $L \cdot (\text{fwd} + \text{grad})$ per sample — manageable at 6500 dims

**Option B: GMM base** (if exact likelihood is required):
- `normflows.distributions.base.GaussianMixture` is available in your installed library
- Exact $\log p$, exact sampling, $O(K d)$ cost
- Fixes global multimodality without any MC estimation
- Mitigate training instability: pretrain flow with Gaussian base, then fit/anneal GMM on fixed $z = F^{-1}(x)$

**Option C: LARS on coarsest multiscale level only:**
- Keep Gaussian base for factored-out levels, resample only the final (small, $\le 100$ dim) latent
- $Z$-estimation is bounded and does not grow with total dimensionality
- Global mode structure typically lives at the coarsest scale

### 6.3 Comparison of Approaches

| Method | Exact $\log p$ | Scales to $\sim 6500$d | Fixes topology | Implementation effort |
|--------|:---:|:---:|:---:|:---:|
| GMM base | ✓ | ✓ | Partial (finite mixture) | Low (built into normflows) |
| LARS (full factorization) | Approx | ✗ ($< 100$d per factor) | Per-factor only | Medium |
| LARS (coarsest-level only) | Approx | ✓ | Partial (coarse modes) | Medium |
| Classifier + rejection | Approx | ✗ ($M$ curse) | ✓ (if $r$ exact) | Low |
| Classifier + HMC | Approx | ✓ | ✓ (if $r$ exact) | Medium |
| Augmented flows | Bound | ✓ | ✓ | Low (pad inputs) |
| CIF | Bound | ✓ | ✓ | Medium (encoder needed) |
| SNF / SurVAE | Bound | ✓ | ✓ | High (new layers) |

### 6.4 The HEP Choice

For calorimeter simulation, the evaluation metric is typically **sample quality** (classifier AUC, shower observable histograms, CaloChallenge metrics) rather than exact likelihood. This means:

- ELBO-based methods (augmented, CIF, SNF, HMC-corrected) are fully acceptable — you don't need exact $\log p$
- The classifier diagnostic (§6.1) doubles as your evaluation metric — train a classifier to distinguish generated vs. real, just as CaloFlow (Krause & Shih 2021) did
- HMC correction with a latent-space classifier gives you a diagnostic-rich sampler whose bias you can measure and control

---

## 7. Summary

1. **LARS resampled base** (Stimper et al. 2022) fixes topology for low-dim targets, but the $Z$-estimation curse prevents scaling past $\sim 100$ dims per factor.

2. **Classifier-based density ratio in latent space** converts the flow's residual density mismatch into a two-sample discrimination problem. It inherits prior art from:
   - LaSeR / ELSA (classifier + HMC in latent space for collider simulations)
   - DRS (discriminator rejection sampling for GANs)
   - Density-ratio estimation (Sugiyama et al. 2012)

3. **Rejection sampling** using the ratio has acceptance rate $\exp(-D_\infty) \sim e^{-c d}$ — same curse as LARS.

4. **HMC** replaces rejection and sidesteps the $M$ curse. Per-sample cost is $L$ classifier evaluations (controllable), and all biases are diagnosable via standard MCMC tools. This is the LaSeR protocol applied natively in latent space.

5. **Diagnostic-first approach:** train a classifier on the existing flow, compute AUC and ESS per incident energy bin. If the flow is already good, no correction needed; if specific regions fail, targeted fixes beat global correction.

---

## References

1. **Stimper, V., Schölkopf, B. & Hernández-Lobato, J. M.** (2022). Resampling Base Distributions of Normalizing Flows. *AISTATS 2022*. arXiv:2110.15828.

2. **Bauer, M. & Mnih, A.** (2019). Resampled Priors for Variational Autoencoders. *AISTATS 2019*. arXiv:1810.11428.

3. **Winterhalder, R., Bellagente, M. & Nachman, B.** (2021). Latent Space Refinement for Deep Generative Models. *NeurIPS DGMs and Applications Workshop 2021*. arXiv:2106.00792.

4. **Nachman, B. & Winterhalder, R.** (2023). ELSA — Enhanced latent spaces for improved collider simulations. *Eur. Phys. J. C 83*, 843. arXiv:2305.07696.

5. **Azadi, S., Olsson, C., Darrell, T., Goodfellow, I. & Odena, A.** (2019). Discriminator Rejection Sampling. *ICLR 2019*. arXiv:1810.06758.

6. **Krause, C. & Shih, D.** (2021). CaloFlow: Fast and Accurate Generation of Calorimeter Showers with Normalizing Flows. *Phys. Rev. D 107*, 113003. arXiv:2106.05285.

7. **Favaro, L., Ore, A., Palacios Schweitzer, S. & Plehn, T.** (2025). CaloDREAM — Detector response emulation via attentive flow matching. *SciPost Physics*.

8. **Cornish, R., Caterini, A. L., Deligiannidis, G. & Doucet, A.** (2020). Relaxing Bijectivity Constraints with Continuously Indexed Normalising Flows. *ICML 2020*. arXiv:1909.13833.

9. **Huang, C.-W., Dinh, L. & Courville, A.** (2020). Augmented Normalizing Flows: Bridging the Gap Between Generative Flows and Latent Variable Models. arXiv:2002.07101.

10. **Wu, H., Köhler, J. & Noé, F.** (2020). Stochastic Normalizing Flows. *NeurIPS 2020*. arXiv:2002.06707.

11. **Nielsen, D., Jaini, P., Hoogeboom, E., Winther, O. & Welling, M.** (2020). SurVAE Flows: Surjections to Bridge the Gap between VAEs and Flows. *NeurIPS 2020*. arXiv:2007.02731.

12. **Sugiyama, M., Suzuki, T. & Kanamori, T.** (2012). *Density Ratio Estimation in Machine Learning*. Cambridge University Press.

13. **Butter, A. et al.** (2024). How to understand limitations of generative networks. *SciPost Physics*.

14. **Ding, X., Wang, Z. J. & Welch, W. J.** (2020). Subsampling Generative Adversarial Networks: Density Ratio Estimation in Feature Space with Softplus Loss. *IEEE Trans. Signal Processing*. arXiv:1909.10670.
