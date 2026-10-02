# Ratio Metric Treatment Effect Estimation with Bivariate Regression

This repository implements covariate-adjusted treatment effect estimation for **ratio metrics** (e.g., Click-Through Rate, Revenue per Order, Conversions per Visit) using a bivariate linear regression framework and the delta method.

Reference: [Jeff Wong: Ratio Metrics with Covariate Adjustment](https://jeffwong.github.io/posts/ratio_metrics/)

---

## 1. Problem Formulation & Target Estimand

In experiment evaluation, ratio metrics are typically defined as the ratio of expected outcomes:

$$R_T = \frac{\mathbb{E}[N \mid T=1]}{\mathbb{E}[D \mid T=1]}, \quad R_C = \frac{\mathbb{E}[N \mid T=0]}{\mathbb{E}[D \mid T=0]}$$

The average treatment effect on the ratio metric ($\tau$) is:

$$\tau = R_T - R_C$$

where:
- $N$: Numerator metric (e.g., clicks, spend, converted actions)
- $D$: Denominator metric (e.g., impressions, visits, opportunities)
- $T \in \{0, 1\}$: Treatment assignment indicator
- $X$: Pre-experiment baseline covariates (used for confounding adjustment and variance reduction)

---

## 2. Pitfalls of Naive Estimators

### A. Average of Per-Unit Ratios ($\tau_{\text{naive}}$)
A common pitfall is calculating the average of unit-level ratios:

$$\hat{\tau}_{\text{naive}} = \frac{1}{n_T} \sum_{i: T_i=1} \frac{N_i}{D_i} - \frac{1}{n_C} \sum_{i: T_i=0} \frac{N_i}{D_i}$$

By Jensen's inequality and Taylor approximation:

$$\mathbb{E}\left[\frac{N}{D}\right] \approx \frac{\mathbb{E}[N]}{\mathbb{E}[D]} + \frac{\mathbb{E}[N]}{\mathbb{E}[D]}\left(\frac{\mathrm{Var}(D)}{\mathbb{E}[D]^2} - \frac{\mathrm{Cov}(N, D)}{\mathbb{E}[N]\mathbb{E}[D]}\right)$$

This introduces severe finite-sample bias whenever $\mathrm{CV}_D$ is moderate to large, or when $N_i$ and $D_i$ are correlated.

### B. Unadjusted Ratio of Means ($\tau_{\text{unadjusted}}$)

$$\hat{\tau}_{\text{unadjusted}} = \frac{\bar{N}_T}{\bar{D}_T} - \frac{\bar{N}_C}{\bar{D}_C}$$

While asymptotically consistent in randomized A/B tests, it cannot correct for selection bias / confounding if treatment assignment $T$ depends on covariates $X$, and it fails to leverage variance reduction (ANCOVA/CUPED-style efficiency gains).

---

## 3. The Bivariate Regression Model

We jointly model the numerator $N$ and denominator $D$ conditional on the design matrix $Z = [1, T, X]$:

$$N_i = \beta_{0N} + \beta_{TN} T_i + \beta_{XN} X_i + \varepsilon_{Ni}$$

$$D_i = \beta_{0D} + \beta_{TD} T_i + \beta_{XD} X_i + \varepsilon_{Di}$$

where the residual error vector has covariance $\Sigma_\varepsilon$:

$$\begin{pmatrix} \varepsilon_{Ni} \\ \varepsilon_{Di} \end{pmatrix} \sim \mathcal{N}\left(\mathbf{0}, \, \Sigma_\varepsilon\right), \quad \Sigma_\varepsilon = \begin{pmatrix} \sigma_N^2 & \sigma_{ND} \\ \sigma_{ND} & \sigma_D^2 \end{pmatrix}$$

Because both equations share identical regressors $Z$, equation-by-equation Ordinary Least Squares (OLS) is equivalent to Seemingly Unrelated Regressions (SUR) and Maximum Likelihood Estimation (MANOVA).

The sample residual covariance matrix is:

$$\hat{\Sigma}_\varepsilon = \frac{1}{n - k} \sum_{i=1}^n \hat{\boldsymbol{\varepsilon}}_i \hat{\boldsymbol{\varepsilon}}_i^T$$

where $k = 3$ is the number of regressors.

---

## 4. Predicted Ratios & The Delta Method

Evaluating at the sample covariate mean $\bar{X}$:
- Treatment prediction vector: $b_T = [1, 1, \bar{X}]^T$
- Control prediction vector: $b_C = [1, 0, \bar{X}]^T$

Predicted group means:
$$\hat{\boldsymbol{\mu}}_T = \begin{pmatrix} \hat{\mu}_{NT} \\ \hat{\mu}_{DT} \end{pmatrix} = \begin{pmatrix} b_T^T \hat{\boldsymbol{\beta}}_N \\ b_T^T \hat{\boldsymbol{\beta}}_D \end{pmatrix}, \quad \hat{\boldsymbol{\mu}}_C = \begin{pmatrix} \hat{\mu}_{NC} \\ \hat{\mu}_{DC} \end{pmatrix} = \begin{pmatrix} b_C^T \hat{\boldsymbol{\beta}}_N \\ b_C^T \hat{\boldsymbol{\beta}}_D \end{pmatrix}$$

The point estimates for the group ratios and treatment effect are:

$$\hat{R}_T = \frac{\hat{\mu}_{NT}}{\hat{\mu}_{DT}}, \quad \hat{R}_C = \frac{\hat{\mu}_{NC}}{\hat{\mu}_{DC}}, \quad \hat{\tau} = \hat{R}_T - \hat{R}_C$$

### Asymptotic Variance via Delta Method

Let $g(\mu_N, \mu_D) = \frac{\mu_N}{\mu_D}$. The gradient vector is:

$$\nabla g(\mu_N, \mu_D) = \begin{pmatrix} \frac{1}{\mu_D} \\ -\frac{\mu_N}{\mu_D^2} \end{pmatrix}$$

The covariance between design predictions is governed by $c_{ij} = b_i^T (Z^T Z)^{-1} b_j$:
- $\mathrm{Cov}(\hat{\boldsymbol{\mu}}_T) = c_T \Sigma_\varepsilon$, where $c_T = b_T^T (Z^T Z)^{-1} b_T$
- $\mathrm{Cov}(\hat{\boldsymbol{\mu}}_C) = c_C \Sigma_\varepsilon$, where $c_C = b_C^T (Z^T Z)^{-1} b_C$
- $\mathrm{Cov}(\hat{\boldsymbol{\mu}}_T, \hat{\boldsymbol{\mu}}_C) = c_{TC} \Sigma_\varepsilon$, where $c_{TC} = b_T^T (Z^T Z)^{-1} b_C$

Applying the multivariate delta method:

$$\widehat{\mathrm{Var}}(\hat{R}_T) = c_T \cdot \nabla g(\hat{\boldsymbol{\mu}}_T)^T \hat{\Sigma}_\varepsilon \nabla g(\hat{\boldsymbol{\mu}}_T)$$

$$\widehat{\mathrm{Var}}(\hat{R}_C) = c_C \cdot \nabla g(\hat{\boldsymbol{\mu}}_C)^T \hat{\Sigma}_\varepsilon \nabla g(\hat{\boldsymbol{\mu}}_C)$$

$$\widehat{\mathrm{Cov}}(\hat{R}_T, \hat{R}_C) = c_{TC} \cdot \nabla g(\hat{\boldsymbol{\mu}}_T)^T \hat{\Sigma}_\varepsilon \nabla g(\hat{\boldsymbol{\mu}}_C)$$

The estimated variance and standard error of $\hat{\tau}$ are:

$$\widehat{\mathrm{Var}}(\hat{\tau}) = \widehat{\mathrm{Var}}(\hat{R}_T) + \widehat{\mathrm{Var}}(\hat{R}_C) - 2 \widehat{\mathrm{Cov}}(\hat{R}_T, \hat{R}_C)$$

$$\mathrm{SE}(\hat{\tau}) = \sqrt{\widehat{\mathrm{Var}}(\hat{\tau})}$$

Hypothesis testing ($H_0: \tau = 0$) and $(1 - \alpha)$ confidence intervals follow standard normal asymptotics:

$$z = \frac{\hat{\tau}}{\mathrm{SE}(\hat{\tau})}, \quad \mathrm{CI}_{1-\alpha} = \left[\hat{\tau} \pm z_{1 - \alpha / 2} \cdot \mathrm{SE}(\hat{\tau})\right]$$

---

## 5. Usage & Testing

### Example Usage

```python
import pandas as pd
from model import estimate_tau, sim

# Generate synthetic data
N, D, T, X = sim(n=1000, bn=3, bd=2, an=10, ad=60)
data = pd.DataFrame({'N': N, 'D': D, 'T': T, 'X': X})

# Estimate treatment effect
res = estimate_tau(data, alpha=0.05)
print(f"tau_hat: {res['tau_hat']:.4f}")
print(f"SE:      {res['se']:.4f}")
print(f"95% CI:  [{res['ci'][0]:.4f}, {res['ci'][1]:.4f}]")
print(f"p-value: {res['p_value']:.4e}")
```

### Running Unit Tests

```bash
python -m unittest test_model.py
```