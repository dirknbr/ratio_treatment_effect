import numpy as np
import pandas as pd
import unittest
from model import estimate_tau, sim


class TestEstimateTau(unittest.TestCase):

  def setUp(self):
    np.random.seed(55)
    N, D, T, X = sim(n=1000, bn=3, bd=2, an=10, ad=60)
    self.sample_data = pd.DataFrame({'N': N, 'D': D, 'T': T, 'X': X})

  def test_estimate_tau_keys(self):
    res = estimate_tau(self.sample_data)
    expected_keys = {"tau_hat", "se", "z_stat", "p_value", "ci", "R_T_hat", "R_C_hat"}
    self.assertTrue(expected_keys.issubset(res.keys()))
    self.assertGreater(res["se"], 0)
    self.assertTrue(0.0 <= res["p_value"] <= 1.0)
    self.assertLess(res["ci"][0], res["tau_hat"])
    self.assertLess(res["tau_hat"], res["ci"][1])

  def test_estimate_tau_consistency(self):
    res = estimate_tau(self.sample_data)
    self.assertAlmostEqual(res["tau_hat"], res["R_T_hat"] - res["R_C_hat"], places=9)
    self.assertAlmostEqual(res["z_stat"], res["tau_hat"] / res["se"], places=9)

  def test_estimate_tau_null_effect(self):
    np.random.seed(123)
    # bn=0, bd=0 -> null treatment effect (true_tau = 0)
    N, D, T, X = sim(n=3000, bn=0, bd=0, an=10, ad=60)
    df = pd.DataFrame({'N': N, 'D': D, 'T': T, 'X': X})
    res = estimate_tau(df)
    self.assertLess(abs(res["tau_hat"]), 0.01)
    self.assertLessEqual(res["ci"][0], 0.0)
    self.assertGreaterEqual(res["ci"][1], 0.0)

  def test_estimate_tau_covers_truth(self):
    np.random.seed(999)
    bn, bd, an, ad = 3, 2, 10, 60
    true_tau = (an + bn) / (ad + bd) - (an / ad)
    N, D, T, X = sim(n=4000, bn=bn, bd=bd, an=an, ad=ad)
    df = pd.DataFrame({'N': N, 'D': D, 'T': T, 'X': X})
    res = estimate_tau(df)
    self.assertLessEqual(res["ci"][0], true_tau)
    self.assertGreaterEqual(res["ci"][1], true_tau)

  def test_estimate_tau_custom_alpha(self):
    res_95 = estimate_tau(self.sample_data, alpha=0.05)
    res_99 = estimate_tau(self.sample_data, alpha=0.01)
    width_95 = res_95["ci"][1] - res_95["ci"][0]
    width_99 = res_99["ci"][1] - res_99["ci"][0]
    self.assertGreater(width_99, width_95)


if __name__ == "__main__":
  unittest.main()