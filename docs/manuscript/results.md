# Results (draft)

This section reports the results of our simulation-based power analysis evaluating (1) the impact of calibration clamping (`--pin_m`), (2) the effect of spiked control count ($N$), and (3) the relative performance of different congression correction models across a range of co-transformation rates ($\lambda$).

## Impact of Calibration Clamping (`--pin_m`)

In the standard fitting workflow, stochastic variational inference (SVI) optimizes the variational guide over millions of growth observations. When the co-transformation rate $\lambda$ is non-zero, the model must simultaneously infer the cell growth rates and correct for congression. Under the unpinned setup, SVI is free to adjust the occupancy-to-growth slope $m$ in response to the likelihood. However, this introduces a parameter degeneracy where SVI can adjust $m$ to absorb experimental noise and co-transformation confounding, pulling it away from its MAP calibration value.

By enabling `--pin_m` during `tfs-prefit-calibration`, the slope $m$ is hard-clamped to the Map estimate obtained from clean, control-only calibration data, preventing SVI from walking off during the full joint fit.

Table 1 summarizes the performance of the Logit-Normal congression correction model under the pinned vs. unpinned setups across all simulated $\lambda$ values (averaged over replicates and spiked control counts).

### Table 1: Calibration Clamping Comparison (Logit-Normal Model)

| Simulated Co-transformation ($\lambda$) | Setup | Pearson Correlation ($r$) | RMSE | Bias (Mean Error) |
|---|---|---|---|---|
| **$\lambda = 0.0000$** (No Congression) | Unpinned | **0.916** | **0.146** | -0.057 |
| | Pinned | 0.689 | 0.247 | **0.054** |
| **$\lambda = 0.1600$** | Unpinned | **0.816** | **0.213** | -0.079 |
| | Pinned | 0.665 | 0.255 | **0.043** |
| **$\lambda = 0.3300$** | Unpinned | 0.627 | 0.310 | -0.119 |
| | Pinned | **0.675** | **0.261** | **0.033** |
| **$\lambda = 0.3572$** (MAP Experimental Estimate) | Unpinned | 0.581 | 0.329 | -0.123 |
| | Pinned | **0.654** | **0.270** | **0.036** |
| **$\lambda = 0.6700$** (95% CI Upper Bound) | Unpinned | 0.467 | 0.376 | -0.126 |
| | Pinned | **0.504** | **0.334** | **-0.013** |

### Key Findings
- **Low Congression Regime ($\lambda \le 0.16$)**: When co-transformation is low, the unpinned setup yields higher Pearson correlation ($r = 0.916$ vs $r = 0.689$ at $\lambda = 0.0$). In this regime, growth trajectories contain direct, clean signal about growth rates, allowing SVI to fine-tune $m$ to maximize fit quality.
- **High Congression Regime ($\lambda \ge 0.33$)**: Under realistic co-transformation levels ($\lambda = 0.3572$), unpinned SVI collapses due to parameter confounding, resulting in high error (RMSE = 0.329) and significant negative bias (Bias = -0.123). Pinning $m$ breaks this degeneracy, yielding substantially better prediction accuracy (Pearson $r$ increases by **+0.073** to **0.654**, RMSE drops by **-0.059** to **0.270**) and reduces bias to near-zero (**0.036**).

---

## Effect of Spiked Control Count ($N$) on Model Performance

To evaluate the experimental design trade-off, we simulated screens with varying numbers of spiked control genotypes ($N \in \{4, 8, 12, 16, 20, 24\}$). These controls are exempted from congression and serve to anchor the pre-fit growth calibration.

Table 2 shows the test set prediction metrics for the pinned Logit-Normal model at the experimental MAP co-transformation rate ($\lambda = 0.3572$) as a function of $N$.

### Table 2: Performance vs. Spiked Control Count ($N$) at $\lambda = 0.3572$

| Number of Spiked Controls ($N$) | Pearson Correlation ($r$) | RMSE | Bias (Mean Error) |
|---|---|---|---|
| **$N = 4$** | **0.661** | 0.269 | 0.045 |
| **$N = 8$** | **0.661** | **0.268** | 0.043 |
| **$N = 12$** | 0.657 | 0.269 | 0.036 |
| **$N = 16$** | 0.651 | 0.270 | 0.035 |
| **$N = 20$** | 0.646 | 0.271 | 0.029 |
| **$N = 24$** | 0.647 | 0.271 | **0.028** |

### Key Findings
- **Diminishing Returns**: Increasing the number of spiked control genotypes from 4 to 24 does not improve prediction accuracy. A small panel of $N = 4$ to $8$ controls is sufficient to anchor the calibration pre-fit.
- **Coverage Dilution**: We observe a very minor drop in Pearson correlation (from 0.661 to 0.647) at higher control counts. Because controls are spiked in at higher concentrations, adding more control genotypes dilutes the read coverage of the library genotypes, slightly increasing the sequencing noise on the library test set.
- **Recommendation**: For high-throughput transcription factor screens, we recommend spiking in a small panel of **4 to 8 control genotypes** rather than a larger set.

---

## Comparison of Congression Correction Models

We compared three models for handling transformation congression:
1. **Single (Null)**: Puts all genotypes in the single-plasmid regime (no correction).
2. **Empirical**: Corrects for congression by max-pooling occupancy over the empirical population distribution of $\theta$.
3. **Logit-Normal**: Corrects for congression using a logit-normal parameterization of the background $\theta$ distribution.

Table 3 compares these models under the pinned slope $m$ setup at the experimental co-transformation rate ($\lambda = 0.3572$).

### Table 3: Model Comparison at $\lambda = 0.3572$ (Pinned $m$)

| Model | Pearson Correlation ($r$) | RMSE | Bias (Mean Error) |
|---|---|---|---|
| **Logit-Normal** | **0.654** | **0.270** | 0.036 |
| **Single (Null)** | 0.588 | 0.275 | **0.029** |
| **Empirical** | 0.493 | 0.331 | 0.090 |

### Key Findings
- **Logit-Normal Superiority**: The Logit-Normal model achieves the highest prediction accuracy ($r = 0.654$, RMSE = 0.270). By assuming a smooth distribution of background genotypes on the logit scale, it acts as a regularizer, avoiding overfitting.
- **Empirical Model Deficit**: The Empirical model performs poorly ($r = 0.493$, RMSE = 0.331). The empirical max-pooling correction is sensitive to outliers in the minibatch during forward propagation, inflating the background occupancy and destabilizing SVI updates.
