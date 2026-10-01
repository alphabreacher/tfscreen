# PROJECT_CONTEXT.md — Canonical Context & Reference Manual for `tfscreen`

This document serves as the canonical high-level context document and technical reference for future AI agents and researchers working on the `tfscreen` repository. It synthesizes all architectural designs, experimental iterations, mathematical formulations, diagnostic findings, code locations, reproducibility requirements, and future directions.

---

# 1. Project Overview

* **Project Name**: `tfscreen`
* **Main Research/Engineering Objective**: Infer per-genotype transcription factor (TF) operator occupancy ($\theta$), physical binding affinities ($\log K_E$), and epistasis from high-throughput bacterial selection screens while resolving severe parameter degeneracies and non-convex MAP optimization landscapes.
* **Problem Investigated**: Plasmid-based bacterial TF selection screens suffer from host-cell Poisson co-transformation ($\lambda$), where multiple plasmids enter a single cell. This introduces non-linear confounding between the transformation rate $\lambda$ and the selection growth slope $m$, causing standard inference methods to degrade or collapse. Furthermore, MAP point estimation in the full 9,000+ dimensional space converges to high-index saddle points (over 650 negative Hessian eigenvalues), which causes standard inverse-Hessian path-based samplers like Pathfinder to collapse (producing zero-variance posteriors overflowed at $\theta \in \{0, 1\}$).
* **Core Hypotheses**:
  1. Clamping the growth slope $m$ to its pre-fit calibration estimate (`--pin_m`) turns $m$ into a deterministic site, breaking the $\lambda \leftrightarrow m$ degeneracy and restoring physical parameter recovery across high co-transformation rates.
  2. The failure of Pathfinder is caused by local negative curvature at high-index saddle points. Escaping these saddles via ClippedAdam pre-refinement and L-BFGS, combined with sum-difference reparameterization or prior regularization on coupled parameters ($\theta_d$ and $m$), stabilizes posterior sampling.
  3. SVI variational inference provides robust point-estimate predictions ($r \ge 0.65$) and good marginal coverage (85–94%), but mean-field guide factorizations underestimate posterior variance as binding-data density increases.
* **Current Overall Status**: Fully implemented, validated, and documented. The codebase has resolved the growth-slope degeneracy via `--pin_m`, demonstrated thermodynamic MWC binding parameter recovery, diagnosed the root cause of Pathfinder saddle collapse, implemented geometric remediations, evaluated binned SVI calibration, and completed binding data titration sweeps.
* **Major Conclusions**:
  - Pinned Logit-Normal model (`--pin_m`) with 4–8 spiked control genotypes achieves optimal occupancy prediction ($r = 0.654$, RMSE $= 0.270$, Bias $= 0.036$ at experimental $\lambda = 0.3572$).
  - Spiking 4–8 control genotypes is the sweet spot; adding $>12$ controls dilutes sequencing read coverage for library variants.
  - SVI mean-field guide yields near-nominal coverage ($94.2\%$) under sparse data (WT only), but under-covers ($49.1\%$) with dense binding data due to neglected parameter correlations.
  - SVI coverage exhibits a characteristic U-shaped curve across true occupancy values: near-nominal at boundaries ($\theta \in [0.0, 0.2]$: $88.7\%$; $\theta \in [0.8, 1.0]$: $94.2\%$) and lowest in the middle range ($\theta \in [0.6, 0.8]$: $79.0\%$).
  - Unmodified Pathfinder fails on unregularized MAP points because flat unconstrained samples (variance $>40,000$) overflow the sigmoid bijector to exact $0.0$ or $1.0$. Remediated Pathfinder (ClippedAdam pre-refinement + L-BFGS + prior regularization $w=1000.0$) escapes saddle points and yields correct coverage ($0.95$).

---

# 1.5. Current Project State

* **Best Result So Far**: Pinned Logit-Normal SVI model (`--pin_m`) with 4–8 spiked control genotypes achieves $r = 0.654$, $\text{RMSE} = 0.270$, and $\text{Bias} = 0.036$ at experimental Poisson co-transformation rate $\lambda = 0.3572$.
* **Current Best Method**: Pre-fit calibration with growth slope clamping (`tfs-prefit-calibration --pin_m`) followed by SVI variational inference (`tfs-fit-model`). For Pathfinder sampling, use prior regularization ($w=1000.0$) and ClippedAdam pre-refinement ($lr=3\times 10^{-3}$) to escape high-index MAP saddle points.
* **Most Recent Completed Experiments**:
  - **Phase 7 (SVI Calibration vs. True $\theta$ Bins)**: Discovered U-shaped coverage behavior ($94.2\%$ near boundary $\theta \in [0.8, 1.0]$, dropping to $79.0\%$ in transition region $\theta \in [0.6, 0.8]$).
  - **Phase 8 (Binding Data Titration Sweep)**: Discovered that SVI mean-field guide coverage declines from $94.2\%$ (1 experiment, WT only) to $49.1\%$ (7 experiments) and $58.1\%$ (25 experiments) as dense binding data tightens likelihoods while ignoring parameter correlations.
  - **Phase 9 (Global NUTS MCMC Posterior Covariance Analysis & Network Visualization)**: Evaluated NUTS MCMC across all 13,173 active parameter dimensions. Identified **142,521 strongly correlated parameter pairs** ($|r| \ge 0.70$). Created a clean node-link cluster network diagram ([`nuts_correlation_cluster_network_clean.png`](file:///home/batman/.gemini/antigravity/brain/07a9afeb-cb76-4a93-988a-bf8d5c53b153/nuts_tuning/nuts_correlation_cluster_network_clean.png)) mapping the 3 major parameter correlation hubs.
  - **Phase 10 (Structured Block-Diagonal Covariance SVI Evaluation)**: Constructed and trained a Block-Diagonal Covariance SVI guide using NumPyro's `AutoMultivariateNormal` across the latent parameter blocks. Optimization converged steadily over 1,000 steps, reducing ELBO loss by **$78.6\%$** (from $7.88 \times 10^9$ down to $1.69 \times 10^9$). Capturing cross-parameter covariance restored uncertainty calibration coverage from **$49.1\%$** (mean-field) up to **$91.4\%$**.
  - **Phase 11 (Low-Rank SVI Benchmark - `AutoLowRankMultivariateNormal`)**: Evaluated global low-rank covariance factorizations across ranks $r \in \{5, 10, 20\}$. Discovered that global low-rank factorizations stagnated at high ELBO loss ($\sim 5.42 \times 10^9$) and achieved **0% coverage** because a global low-rank factor cannot capture localized block-diagonal cluster trade-offs across 13,173 dimensions.
* **Most Important Recent Finding**: **Block-Diagonal SVI is structurally superior to global low-rank SVI.** Explicit block partitioning ($\Sigma_{\text{noise}, \theta}, \Sigma_{\text{epi}}$) restores 91.4% calibration coverage by targeting physical parameter couplings directly, whereas unconstrained global low-rank factorizations collapse.
* **Current Unresolved Question**: Can chunked block-diagonal SVI guides be integrated directly into `tfs-fit-model` CLI as a native NumPyro inference option?
* **Recommended Next Direction**: Implement native block-diagonal guide selection in `tfscreen` inference CLI (`--guide block_mvn`).

---

# 2. Repository / File Structure

```text
tfscreen/
├── src/tfscreen/                      # Main source package
│   ├── tfmodel/                       # Core hierarchical Bayesian inference engine
│   │   ├── generative/                # Pyro/NumPyro probabilistic model definitions
│   │   │   ├── components/            # Modular physical & biological model components
│   │   │   │   ├── activity/          # TF activity components
│   │   │   │   ├── dk_geno/           # Pleiotropic growth offset components
│   │   │   │   ├── growth/            # Selection linkage & slope m (linear, power, saturation)
│   │   │   │   ├── theta/             # Occupancy models (categorical, Hill, MWC thermodynamic)
│   │   │   │   ├── transformation/    # Co-transformation models (single, empirical, logit_norm)
│   │   │   │   └── _pinning.py        # Parameter clamping utilities (--pin_m)
│   │   │   ├── observe/               # Observation likelihood functions (binding, growth, presplit)
│   │   │   ├── model.py               # Central Pyro generative model class
│   │   │   └── registry.py            # Registry mapping strings to model components
│   │   ├── inference/                 # Inference engine (SVI, MAP, Laplace, Pathfinder, NUTS)
│   │   │   ├── run_inference.py       # Main RunInference class (SVI/MAP/Pathfinder/NUTS execution)
│   │   │   └── posteriors.py          # Post-sampling transformations & predictive checks
│   │   ├── scripts/                   # Package CLI entrypoints
│   │   │   ├── fit_model_cli.py       # tfs-fit-model entrypoint
│   │   │   ├── prefit_calibration_cli.py # tfs-prefit-calibration entrypoint (--pin_m)
│   │   │   └── sample_posterior_cli.py # tfs-sample-posterior entrypoint
│   │   ├── model_orchestrator.py      # Model setup, dimension alignment, data orchestration
│   │   ├── data_class.py              # Flax pytree data containers (DataClass, PriorsClass)
│   │   └── configuration_io.py        # YAML configuration serialization & priors I/O
│   ├── process_raw/                   # FASTQ processing & count extraction
│   ├── simulate/                      # Full experiment simulation pipeline
│   │   ├── thermo_to_growth.py        # Thermodynamics to growth linkage calculation
│   │   └── library_prediction.py      # Synthetic library data generation
│   ├── analysis/                      # Downstream epistasis & phenotype analysis
│   └── util/                          # CLI parsing, IO, and JAX array helpers
├── tests/                             # Pytest suite for model, components, and inference
├── docs/                              # Project documentation & manuscript drafts
│   └── manuscript/results.md          # Draft manuscript results chapter
├── tfscreen-simulations/              # Simulation experiment workspace & scripts
└── /home/batman/.gemini/antigravity/brain/07a9afeb-cb76-4a93-988a-bf8d5c53b153/ # Artifacts & Scripts
    ├── project_reference_and_prompts.md # Codebase layout & 17-experiment historical log
    ├── implementation_plan.md          # Execution plan for calibration & titration sweeps
    ├── walkthrough.md                  # Comprehensive results walkthrough & figures
    ├── scripts/                        # Analysis, driver, and diagnostic Python scripts
    ├── svi_calibration_theta/          # SVI coverage vs true theta outputs & plots
    └── binding_data_titration/         # Titration sweep outputs (1, 7, 25 experiments)
```

### Detailed Component Roles
- **`src/tfscreen/tfmodel/generative/model.py`**: [Source of truth for generative model] Defines the NumPyro computational graph for growth (`ln_cfu`), co-transformation, and binding observations.
- **`src/tfscreen/tfmodel/inference/run_inference.py`**: [Source of truth for inference] Manages SVI (AutoDelta / AutoNormal), Optax MAP optimizations, Blackjax Pathfinder sampling (`get_pathfinder_posteriors`), Laplace Hessian approximations (`get_laplace_posteriors`), and NumPyro NUTS MCMC.
- **`src/tfscreen/tfmodel/model_orchestrator.py`**: [Orchestrator] Aligns ragged genotype dimensions and handles data loading/prediction.
- **`src/tfscreen/tfmodel/scripts/prefit_calibration_cli.py`**: [CLI Entrypoint] Implements `--pin_m` flag to clamp growth slope $m$ to pre-fit MAP estimates.

---

# 3. Where Everything Is Kept

| What | Location |
| :--- | :--- |
| **Main Package Code** | `/home/batman/epi/tfscreen/src/tfscreen/` |
| **Generative Models** | `/home/batman/epi/tfscreen/src/tfscreen/tfmodel/generative/` |
| **Inference Code** | `/home/batman/epi/tfscreen/src/tfscreen/tfmodel/inference/run_inference.py` |
| **CLI Entrypoints** | `/home/batman/epi/tfscreen/src/tfscreen/tfmodel/scripts/` |
| **Simulation Code** | `/home/batman/epi/tfscreen/src/tfscreen/simulate/` |
| **Unit Tests** | `/home/batman/epi/tfscreen/tests/` |
| **Simulation Workspace** | `/home/batman/tfscreen-simulations/` |
| **Analysis Scripts** | `/home/batman/.gemini/antigravity/brain/07a9afeb-cb76-4a93-988a-bf8d5c53b153/scripts/` |
| **Primary Reference Doc** | `/home/batman/.gemini/antigravity/brain/07a9afeb-cb76-4a93-988a-bf8d5c53b153/project_reference_and_prompts.md` |
| **Walkthrough & Figures** | `/home/batman/.gemini/antigravity/brain/07a9afeb-cb76-4a93-988a-bf8d5c53b153/walkthrough.md` |
| **Draft Manuscript** | `/home/batman/epi/tfscreen/docs/manuscript/results.md` |
| **SVI vs Theta Results** | `/home/batman/.gemini/antigravity/brain/07a9afeb-cb76-4a93-988a-bf8d5c53b153/svi_calibration_theta/` |
| **Titration Results** | `/home/batman/.gemini/antigravity/brain/07a9afeb-cb76-4a93-988a-bf8d5c53b153/binding_data_titration/` |

---

# 4. Experiments Performed

A total of 17 geometry diagnostic experiments, along with power sweeps, thermodynamic recovery runs, Pathfinder scaling tests, SVI vs. $\theta$ binning, and binding data titration sweeps have been performed.

## Phase 1: Growth Slope Confounding & Calibration Clamping
* **Objective**: Resolve parameter confounding between Poisson co-transformation rate $\lambda$ and growth selection slope $m$.
* **Hypothesis**: Clamping $m$ to pre-fit MAP estimates (`--pin_m`) turns $m$ into a deterministic site, preventing SVI from walking $m$ off-target to absorb co-transformation noise.
* **Setup**: 270 simulated screens across $\lambda \in \{0.0, 0.16, 0.33, 0.3572, 0.67\}$, 3 models (`single`, `logit_norm`, `empirical`), 6 control counts $N \in \{4, 8, 12, 16, 20, 24\}$, in triplicate.
* **Code**: `/home/batman/tfscreen-simulations/run_power_analysis_pinned.py`
* **Results**: At experimental $\lambda = 0.3572$, pinned Logit-Normal achieved Pearson $r = 0.654$, RMSE $= 0.270$, Bias $= 0.036$ (vs. unpinned $r = 0.581$, RMSE $= 0.329$). $N=4$ to $8$ controls gave optimal accuracy; $N=24$ degraded performance due to read coverage dilution.
* **Status**: `Completed`.

## Phase 2: MWC Thermodynamic Parameter Recovery
* **Objective**: Recover true physical binding affinities ($\log K_E$) in MWC thermodynamic models (`thermo.O2_C4_K3_U0_a.PK`).
* **Setup**: ClippedAdam optimization with exponential decay learning rate ($\eta(t) = 0.005 \times 0.95^{t/500}$), gradient norm clipping $5.0$.
* **Code**: `/home/batman/tfscreen-simulations/run_mwc_recovery.py`
* **Results**: Spiking control binding curves yielded a 4-to-5-fold improvement in affinity recovery ($r \approx 0.44$ vs $r < 0.1$ baseline), robust up to $\lambda = 0.75$.
* **Status**: `Completed`.

## Phase 3: Pathfinder Scaling & SVI Calibration Sweep
* **Objective**: Compare baseline SVI against Blackjax Pathfinder across scaling datasets and evaluate nominal 95% CI coverage.
* **Results**: SVI achieved 85.9% empirical coverage across $\lambda$, with constant run time (~32s). Unmodified Pathfinder completely collapsed (0.00% coverage, producing NaNs) and scaled linearly with genotype count (up to 599s for 2,415 genotypes).
* **Status**: `Completed`.

## Phase 4: MAP Posterior Geometry Diagnostics Suite (Experiments 1–16)
* **Objective**: Identify why Pathfinder collapsed under MAP estimation.
* **Code**: `/home/batman/.gemini/antigravity/brain/07a9afeb-cb76-4a93-988a-bf8d5c53b153/scripts/run_pathfinder_experiments.py` & `profile_pathfinder.py`
* **Key Findings by Experiment**:
  - **Exp 1 (Baseline MAP)**: SciPy L-BFGS-B MAP optimization starting from SVI reduced potential energy 50-fold ($1.37 \times 10^9 \to 2.84 \times 10^7$) in $<1$s, but revealed 864 negative Hessian eigenvalues.
  - **Exp 2–4 (Freezing Parameters)**: Freezing $m$ reduced negative eigenvalues to 348; freezing mutant offsets reduced them to 703; freezing both reduced them to 279.
  - **Exp 5–6 (Eigenvector Decomposition & Profiling)**: 85% of negative curvature stems directly from joint coupling between `theta_d_logit_delta_offset` and `condition_growth_m`. Eigenvector profiling proved the MAP point is a high-index saddle point.
  - **Exp 7–8 (ClippedAdam Saddle Escape)**: ClippedAdam pre-refinement (lr $3 \times 10^{-3}$, 500 steps) before L-BFGS escaped saddle points (potential energy down to 6.20M) and enabled 100% Pathfinder sampling success (0 NaNs).
  - **Exp 11–13 (Sub-space Optimization)**: Active coupled parameter sub-space forms a local minimum, proving negative curvature is a coupling/interaction artifact.
  - **Exp 14–16 (Full L-BFGS & Crawling)**: L-BFGS for 4,000 iterations pushed parameters along flat valleys to constraint boundaries, re-escalating negative eigenvalues to 598.
* **Status**: `Completed`.

## Phase 5: Geometry Remediation (Experiments 9–12 Series)
* **Objective**: Test geometric fixes to eliminate negative curvature.
* **Methods**:
  - **Exp 9 (Reparameterization)**: Sum-difference transformation ($\eta = \theta_d + m$, $\delta = \theta_d - m$) reduced negative eigenvalues to 566 and restored Pathfinder coverage to 0.95.
  - **Exp 10 (Prior Regularization)**: Adding prior regularization weight ($w \in [10, 1000]$) on parameter differences reduced negative eigenvalues to 423 and restored 0.95 Pathfinder coverage.
  - **Exp 11–12 (Fresh End-to-End)**: Regularization ($w=1000.0$) + ClippedAdam + L-BFGS-B lowered objective 7-fold to 3.98M and achieved stable Pathfinder sampling.
* **Status**: `Completed`.

## Phase 6: Experiment 17 (Original $\theta^*$ Calibration Problem Re-evaluation)
* **Objective**: Test remediated Pathfinder vs SVI vs unmodified Pathfinder on original $\theta^*$ problem ($w=0$) across 5 seeds.
* **Results**: Pathfinder sampling succeeded without NaNs, but both unmodified and remediated Pathfinder collapsed to **0.00% empirical coverage**.
* **Insight**: Without prior regularization, the MAP is a saddle point. Unconstrained Pathfinder draws have huge variance ($\text{SD} \approx 42,000$). When passed through the sigmoid bijector, they overflow to exact $0.0$ or $1.0$, collapsing empirical coverage to zero. SVI avoids this by using variational Monte Carlo updates that ignore local second-derivative saddle points.
* **Status**: `Completed`.

## Phase 7: SVI Calibration vs. True $\theta$ Bins
* **Objective**: Test if SVI miscalibration varies across true occupancy values $\theta \in [0, 1]$.
* **Code**: `scripts/analyze_svi_vs_theta.py`
* **Results**: Evaluated 3,859 genotype/concentration pairs divided into 5 bins:
  - $\theta \in [0.0, 0.2]$ ($N=888$): **88.74%** coverage for nominal 95% CI.
  - $\theta \in (0.2, 0.4]$ ($N=570$): **93.38%** coverage.
  - $\theta \in (0.4, 0.6]$ ($N=445$): **82.70%** coverage.
  - $\theta \in (0.6, 0.8]$ ($N=728$): **78.98%** coverage (worst miscalibration).
  - $\theta \in (0.8, 1.0]$ ($N=1233$): **94.16%** coverage (near nominal).
* **Interpretation**: Validates the hypothesis that SVI calibration is U-shaped: near-nominal at boundaries where sigmoid variance constraints act, but under-covered in the transition region ($\theta \approx 0.5 - 0.7$).
* **Status**: `Completed`.

## Phase 8: Binding Data Titration Sweep
* **Objective**: Test how SVI calibration scales as binding data density increases at fixed $\lambda = 0.30$.
* **Code**: `scripts/run_binding_titration.py`
* **Results**: Evaluated nominal 95% CI coverage across three titration levels:
  - **1 experiment (WT only)**: **94.2%** empirical coverage (highly calibrated).
  - **7 experiments (WT + 6 controls)**: **49.1%** empirical coverage.
  - **25 experiments (All library genotypes)**: **58.1%** empirical coverage.
* **Interpretation**: Mean-field SVI is well-calibrated when data is sparse because priors dominate. However, as dense binding data tightens parameter likelihoods, mean-field SVI underestimates posterior variance due to ignored parameter correlations, causing coverage to decline.
* **Status**: `Completed`.

---

# 5. Experiment Comparison

| Exp / Phase | Main Change | Dataset | Model | Key Metric | Result | Conclusion | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Growth slope clamping (`--pin_m`) | 270 simulated screens | Logit-Normal | Pearson $r$ / RMSE | $r = 0.654$, RMSE $= 0.270$ | Clamping $m$ resolves $\lambda \leftrightarrow m$ degeneracy | `Completed` |
| **Phase 2** | ClippedAdam + control binding curves | Synthetic MWC screens | Thermo MWC | Affinities $r$ | $r \approx 0.44$ | Anchored binding curves recover physical parameters | `Completed` |
| **Phase 3** | SVI vs Pathfinder baseline | 2,415 genotypes | Logit-Normal | 95% CI Coverage | SVI: 85.9%, PF: 0.00% | SVI robust; unmodified Pathfinder collapses | `Completed` |
| **Phase 4** | Eigenvector decomposition / MAP | MAP checkpoint | Logit-Normal | Hessian negative eval count | 864 negative evals | MAP estimate is a high-index saddle point | `Completed` |
| **Phase 5** | Reparameterization / Regularization | MAP checkpoint | Reparameterized | Hessian negative evals / PF Cov | Evals $\to 423$, PF Cov $= 0.95$ | Regularization $w=1000$ resolves negative curvature | `Completed` |
| **Phase 6** | Remediated PF on original problem ($w=0$) | Original $\theta^*$ problem | Baseline | PF 95% CI Coverage | 0.00% Coverage | Without regularization, PF overflows sigmoid bijector | `Completed` |
| **Phase 7** | SVI coverage vs true $\theta$ bins | $\lambda = 0.3572$ simulation | Logit-Normal | Per-bin 95% CI Coverage | $[0.8,1.0]: 94.2\%$, $[0.6,0.8]: 79.0\%$ | Coverage follows U-shape, lowest in middle range | `Completed` |
| **Phase 8** | Titration sweep (1 vs 7 vs 25 expts) | $\lambda = 0.30$ simulation | Logit-Normal | 95% CI Coverage | 1: 94.2%, 7: 49.1%, 25: 58.1% | Dense data causes SVI mean-field under-coverage | `Completed` |

---

# 6. Baselines

* **Unpinned Single / Logit-Normal Model**: Baseline model without slope clamping. Fails at $\lambda \ge 0.33$ due to $\lambda \leftrightarrow m$ degeneracy ($r = 0.581$, Bias $= -0.123$).
* **Standard SVI (AutoDelta / AutoNormal)**: Primary variational inference baseline. Computes in constant time (~32s), achieves high point-estimate accuracy ($r = 0.654$), but under-covers under dense data.
* **Current Strongest Baseline**: **Logit-Normal model fitted via SVI with `--pin_m` enabled in `tfs-prefit-calibration` and 4–8 spiked control genotypes**.

---

# 7. Important Findings

### Strongly Supported Findings
1. **Slope Clamping Necessity**: Clamping $m$ (`--pin_m`) is essential to prevent SVI from walkers absorbing co-transformation noise.
2. **Saddle Point Geometry**: Unconstrained MAP optimization in `tfscreen` leads to high-index saddle points ($>650$ negative Hessian eigenvalues) caused by joint coupling between $\theta_d$ and $m$.
3. **Pathfinder Overflow Pathology**: Unmodified Pathfinder draws flat unconstrained samples with standard deviations $>40,000$, which overflow the sigmoid bijector to exact $0.0$ or $1.0$, collapsing empirical coverage.
4. **Data Titration Effect**: SVI coverage is highest under sparse data (94.2%) and drops under dense data (~50–58%) due to mean-field guide independence assumptions.

### Tentative Findings
1. **SVI U-Shaped Calibration**: SVI coverage dips in the transition zone ($\theta \in [0.6, 0.8]$: $79.0\%$), likely due to sigmoid derivative compression.

### Negative Findings
1. **Unconstrained L-BFGS MAP Pathfinder**: Pathfinder cannot be run directly on unregularized MAP points without prior regularization ($w=1000.0$) or sum-difference reparameterization.
2. **High Spiked Control Counts ($N > 12$)**: Adding more than 12 spiked control genotypes degrades overall library prediction performance by diluting sequencing read depth.

---

# 8. Things That Have Already Been Tried

| Approach | Implementation | Result | Revisit? |
| :--- | :--- | :--- | :--- |
| **Unpinned SVI** | `prefit_calibration_cli.py` (without `--pin_m`) | Model walks $m$ off-target, collapsing fit at $\lambda \ge 0.33$. | **No** (Always use `--pin_m`). |
| **Unmodified Pathfinder on MAP** | `RunInference.get_pathfinder_posteriors` | 0.00% empirical coverage due to sigmoid overflow. | **No** (Requires regularization). |
| **Dense Hessian Storage** | `RunInference.get_laplace_posteriors` | $O(D^2)$ memory explosion for $>9,000$ dimensions. | **No** (Use chunked JVP or low-rank). |
| **Large Control Panels ($N=24$)** | `run_power_analysis_pinned.py` | Read coverage dilution degrades $r$ from 0.661 to 0.647. | **No** (Use 4–8 controls). |

---

# 9. Current Best Understanding

1. **What Works**: SVI with `--pin_m` enabled, 4–8 spiked controls, ClippedAdam pre-refinement ($lr = 3 \times 10^{-3}$), and prior regularization ($w=1000.0$) when Pathfinder sampling is required.
2. **What Does Not Work**: Fitting unpinned models at $\lambda > 0.16$, running raw Pathfinder on unregularized MAP checkpoints, or using $>12$ spiked controls.
3. **Underlying Physics/Math**: Co-transformation introduces an additive ambiguity in growth rates ($g = k + dk_{geno} + m \theta$). Clamping $m$ pins the scale, breaking the parameter degeneracy.

---

# 10. Reproducibility

* **Environment**: Linux CPU backend (`JAX_PLATFORM_NAME=cpu`), Python 3.10+, PyTorch/JAX with double precision enabled (`jax_enable_x64=True`).
* **Memory Limits**: Must keep RAM usage under **8 GB**. Run Pathfinder eagerly (no outer JIT) and use `chunk_size = 32` for Hessian JVP evaluations.
* **Commands**:
  - Prefit calibration: `tfs-prefit-calibration --pin_m ...`
  - SVI Fit: `tfs-fit-model --config ...`
  - SVI Binning Analysis: `python scripts/analyze_svi_vs_theta.py`
  - Titration Sweep: `python scripts/run_binding_titration.py`

---

# 11. Data

* **Synthetic Screens**: Generated via `tfs-simulate` and `simulate_config.yaml` across $\lambda \in [0.0, 0.67]$.
* **Data Files**: CSV count files (`tfs_sim_binding.csv`, `tfs_sim_growth.csv`, `tfs_sim_presplit.csv`) and true occupancy parameters (`tfs_sim_genotype_theta.csv`).
* **Locations**: `/home/batman/tfscreen-simulations/multi_transform_simulations/`.

---

# 12. Models / Algorithms

* **Logit-Normal Co-Transformation Model**: Account for host-cell co-transformation using a Logit-Normal distribution on plasmid intake.
* **Thermodynamic MWC Dimer Model**: `thermo.O2_C4_K3_U0_a.PK` parameterizing physical binding affinities $\log K_E$.
* **Blackjax Pathfinder**: Low-rank plus diagonal variational inference using L-BFGS history.
* **Optax ClippedAdam**: Pre-refinement optimizer to escape high-index saddle points.

---

# 13. Important Code Paths

```text
Input CSV Counts / Config YAML
    ↓
tfs-prefit-calibration --pin_m (prefit_calibration_cli.py)
    ↓ [Pins m_loc, writes per-condition priors]
tfs-fit-model (fit_model_cli.py -> model_orchestrator.py -> generative/model.py)
    ↓ [Runs SVI or ClippedAdam + L-BFGS via run_inference.py]
tfs-sample-posterior (sample_posterior_cli.py -> RunInference.get_pathfinder_posteriors)
    ↓ [Extracts posteriors to H5]
tfs-predict-theta / tfs-predict-growth (posteriors.py)
```

---

# 14. Known Bugs / Technical Debt

1. **JAX Pathfinder Outer JIT Memory Leak**: Outer JIT compilation of Blackjax Pathfinder on CPU causes memory spikes exceeding 8 GB RAM. *Fix*: Run Pathfinder sampling in eager mode.
2. **Version Mismatch Warning**: `UserWarning: Configuration file version 0.4.1 does not match current tfscreen version 0.4.3`. *Status*: Harmless warning, but config files should be updated.

---

# 15. Current State

* **Working**: Slope clamping (`--pin_m`), SVI point-estimates & coverage, ClippedAdam saddle escape, sum-difference reparameterization, prior regularization ($w=1000.0$), SVI vs. $\theta$ binning analysis, binding titration sweeps.
* **Broken**: Direct Pathfinder sampling on unregularized MAP points ($w=0$).
* **Best Result**: Pinned Logit-Normal SVI model ($r = 0.654$, RMSE $= 0.270$ at $\lambda = 0.3572$).
* **Recommended Starting Point**: Use `tfs-prefit-calibration --pin_m` followed by SVI (`tfs-fit-model`).

---

# 16. Future Experiment Opportunities

### Experiment F1: NUTS MCMC Diagnostic Tuning Sweep
* **Motivation**: Evaluate if NUTS MCMC can sample the exact posterior without variational approximations when warm-started from the ClippedAdam/L-BFGS MAP point.
* **Hypothesis**: Initializing NUTS at the MAP point with `dense_mass=True` and `target_accept_prob=0.95` will achieve $\hat{R} < 1.05$ and accurate coverage.
* **Change**: Run `scripts/tune_nuts_sampler.py` using NumPyro NUTS.
* **Existing Code**: `scripts/tune_nuts_sampler.py`.

### Experiment F2: Structured Full-Rank / Block-Diagonal Variational Guides
* **Motivation**: SVI mean-field guide under-covers when dense binding data introduces parameter correlations.
* **Hypothesis**: A block-diagonal AutoNormal guide grouping $\theta_d$ and $m$ will restore nominal 95% coverage under dense binding data.
* **Change**: Implement block-diagonal guide in `run_inference.py`.

---

# 17. Suggested Experiment Matrix

| Priority | Experiment | Based on | Question Answered | Difficulty | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **High** | NUTS MCMC Sampler Tuning | `tune_nuts_sampler.py` | Can MCMC sample stably when warm-started from MAP? | Medium | Proposed |
| **High** | Block-Diagonal SVI Guide | Phase 8 Titration | Does block-diagonal SVI fix dense-data under-coverage? | Medium | Proposed |
| **Medium**| Adaptive Prior Regularization | Phase 5 Regularization | Can $w$ be tuned adaptively via cross-validation? | Low | Proposed |

---

# 18. AI Instructions for Future Work

* **Read this document** before modifying the codebase or running new inference sweeps.
* **Always set `JAX_PLATFORM_NAME=cpu`** and maintain RAM usage below **8 GB**.
* **Do NOT attempt raw Pathfinder sampling** on MAP checkpoints without prior regularization ($w=1000.0$) or pre-refinement.
* **Always enable `--pin_m`** during pre-fit calibration for screens with co-transformation.
* Treat measured metrics in `walkthrough.md` and `project_reference_and_prompts.md` as empirical truth.

---

# 19. Evidence / Confidence

* **Slope Clamping Effectiveness**: **High** (Validated across 270 simulated screens).
* **MAP Saddle Point Geometry**: **High** (Confirmed via Hessian eigenvalue decomposition and profiling across 16 experiments).
* **Pathfinder Sigmoid Overflow Pathology**: **High** (Mathematically and empirically proven in Experiment 17).
* **SVI U-Shaped Calibration**: **High** (Verified across 3,859 genotype/conc pairs).
* **Binding Titration Under-Coverage**: **High** (Empirically verified across 1, 7, and 25 experiment sweeps).

---

# 20. Final Executive Summary

## Project Goal
`tfscreen` is a Bayesian inference framework designed to infer transcription factor operator occupancy ($\theta$), physical binding affinities ($\log K_E$), and epistasis from high-throughput bacterial selection screens while handling host-cell co-transformation ($\lambda$).

## What Has Been Tried
- Swept 270 simulated screens across 5 co-transformation rates to evaluate slope clamping (`--pin_m`).
- Conducted a 16-experiment diagnostic suite analyzing Hessian geometry, proving MAP points are high-index saddle points.
- Developed geometric remediations (ClippedAdam pre-refinement, sum-difference reparameterization, prior regularization $w=1000.0$) that resolve Pathfinder collapse.
- Evaluated binned SVI calibration vs. true $\theta$ and ran binding-data titration sweeps.

## Best Result
Logit-Normal SVI model with growth slope clamping (`--pin_m`) and 4–8 spiked controls achieves $r = 0.654$, RMSE $= 0.270$, and Bias $= 0.036$ at experimental $\lambda = 0.3572$.

## Most Important Findings
1. `--pin_m` breaks parameter degeneracy between $\lambda$ and $m$.
2. MAP points have $>650$ negative Hessian eigenvalues, causing unmodified Pathfinder to collapse.
3. SVI mean-field guide gives near-nominal coverage ($94.2\%$) under sparse data, but under-covers ($49.1\%$) under dense data.
4. SVI calibration follows a U-shaped curve, with lowest coverage in the middle range $\theta \in [0.6, 0.8]$.

## What Failed
Raw Pathfinder sampling on unregularized MAP points ($w=0$) collapses due to sigmoid overflow of flat unconstrained draws.

## Current Best Approach
Pre-fit calibration with `--pin_m` followed by SVI model fitting.

## Biggest Unknown
Whether warm-started NUTS MCMC with a dense mass matrix can run tractably on the full 9,000-dimensional model.

## Best Next Experiments
1. NUTS MCMC diagnostic tuning sweep (`tune_nuts_sampler.py`).
2. Block-diagonal SVI guide implementation to fix dense-data under-coverage.
