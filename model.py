
"""

Treatment effect for ratio metric with bivariate model
with a covariate
controlling for cov 

https://jeffwong.github.io/posts/ratio_metrics/
"""

import numpy as np
from statsmodels.multivariate.manova import MANOVA
import pandas as pd
import pymc as pm
from scipy import stats
import statsmodels.formula.api as smf

def sim(n=500, bn=3, bd=2, an=10, ad=60):
  X = np.random.normal(0, 1, n)
  # 1. Selection bias / Confounding: T depends on covariate X
  prob_T = 1 / (1 + np.exp(-1.5 * X))
  T = np.random.binomial(1, prob_T, n)
  # 2. Large variance & high correlation between N and D residuals
  S = np.array([[4.0, 3.5], [3.5, 4.0]])
  eps = np.random.multivariate_normal(np.zeros(2), S, n)
  # 3. Smaller denominator mean (CV_D is ~40% instead of ~1%)
  N = an + bn * T + 3.0 * X + eps[:, 0]
  D = ad + bd * T + 1.5 * X + eps[:, 1]
  return N, D, T, X


def estimate_tau(data, alpha=0.05):
  """Estimate treatment effect on ratio metric using bivariate regression and delta method."""
  fit_N = smf.ols('N ~ T + X', data=data).fit()
  fit_D = smf.ols('D ~ T + X', data=data).fit()

  residuals = pd.DataFrame({"N": fit_N.resid, "D": fit_D.resid})
  k = len(fit_N.params)
  n = len(data)
  Sigma_eps_hat = residuals.cov() * (n - 1) / (n - k)

  x_mean = data['X'].mean()
  b_T = np.array([1, 1, x_mean])
  b_C = np.array([1, 0, x_mean])

  mu_N_T_hat = b_T.dot(fit_N.params)
  mu_D_T_hat = b_T.dot(fit_D.params)
  mu_N_C_hat = b_C.dot(fit_N.params)
  mu_D_C_hat = b_C.dot(fit_D.params)

  R_T_hat = mu_N_T_hat / mu_D_T_hat
  R_C_hat = mu_N_C_hat / mu_D_C_hat
  tau_hat = R_T_hat - R_C_hat

  Z = np.column_stack([np.ones(n), data['T'].values, data['X'].values])
  ZtZ_inv = np.linalg.inv(Z.T @ Z)
  c_T = b_T.dot(ZtZ_inv).dot(b_T)
  c_C = b_C.dot(ZtZ_inv).dot(b_C)
  c_TC = b_T.dot(ZtZ_inv).dot(b_C)

  grad_g_T = np.array([1 / mu_D_T_hat, -mu_N_T_hat / (mu_D_T_hat ** 2)])
  grad_g_C = np.array([1 / mu_D_C_hat, -mu_N_C_hat / (mu_D_C_hat ** 2)])

  var_R_T_hat = c_T * grad_g_T.dot(Sigma_eps_hat).dot(grad_g_T)
  var_R_C_hat = c_C * grad_g_C.dot(Sigma_eps_hat).dot(grad_g_C)
  cov_R_T_R_C_hat = c_TC * grad_g_T.dot(Sigma_eps_hat).dot(grad_g_C)

  var_tau_hat = var_R_T_hat + var_R_C_hat - 2 * cov_R_T_R_C_hat
  se_tau_hat = np.sqrt(var_tau_hat)

  z_stat = tau_hat / se_tau_hat
  p_val = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
  z_crit = stats.norm.ppf(1 - alpha / 2)

  return {
    "tau_hat": tau_hat,
    "se": se_tau_hat,
    "z_stat": z_stat,
    "p_value": p_val,
    "ci": (tau_hat - z_crit * se_tau_hat, tau_hat + z_crit * se_tau_hat),
    "R_T_hat": R_T_hat,
    "R_C_hat": R_C_hat,
  }


if __name__ == "__main__":
  bn, bd = 3, 2
  an, ad = 10, 60
  N, D, T, X = sim(bn=bn, bd=bd)

  true_R_T = (an + bn) / (ad + bd)
  true_R_C = an / ad
  true_tau = true_R_T - true_R_C

  data = pd.DataFrame({'N': N, 'D': D, 'T': T, 'X': X})
  res = estimate_tau(data)

  # Naive (biased) estimator: difference of per-unit ratios (E[N/D] != E[N]/E[D])
  tau_naive = (data.loc[data['T'] == 1, 'N'] / data.loc[data['T'] == 1, 'D']).mean() - \
              (data.loc[data['T'] == 0, 'N'] / data.loc[data['T'] == 0, 'D']).mean()

  # Unadjusted ratio of sample means (omitting covariate X)
  tau_unadjusted = (data.loc[data['T'] == 1, 'N'].mean() / data.loc[data['T'] == 1, 'D'].mean()) - \
                   (data.loc[data['T'] == 0, 'N'].mean() / data.loc[data['T'] == 0, 'D'].mean())
  print(f"true_tau: {true_tau:.5f}, tau_hat: {res['tau_hat']:.5f}, tau_naive: {tau_naive:.5f}, tau_unadjusted: {tau_unadjusted:.5f}")
  print(f"SE: {res['se']:.5f}, z: {res['z_stat']:.3f}, p-value: {res['p_value']:.5e}")
  print(f"95% CI: [{res['ci'][0]:.5f}, {res['ci'][1]:.5f}]")