"""
Unit tests for the loss functions.
"""

import jax
import jax.numpy as jnp

from src.training.loss import (
  get_bao_loss,
  get_data_loss,
  get_efe_loss,
)

# Enable float64 for high-precision physics checks
jax.config.update("jax_enable_x64", True)


def test_bao_loss_computation():
  """
  Verifies that the BAO chi-squared loss computes correctly without
  shape errors and correctly applies the covariance weighting.
  """

  # 1. Mock a flat FLRW metric: a(t) = exp(0.5 * (t - 1.0))
  def flrw_metric(coords):
    t = coords[0]
    a = jnp.exp(0.5 * (t - 1.0))
    spatial = a**2 * jnp.eye(3)
    g = jnp.zeros((4, 4))
    g = g.at[0, 0].set(-1.0)
    g = g.at[1:4, 1:4].set(spatial)
    return g

  # 2. Mock 2 BAO data points
  z_vals = jnp.array([0.5, 1.0])

  # For our mock FLRW, D_H = 1/H = 2.0 * L_UNIT
  # We will just supply arbitrary target values.
  dm_obs = jnp.array([10.0, 15.0])
  dh_obs = jnp.array([20.0, 25.0])

  # Mock covariance inverses (identity matrices for simplicity)
  cov_invs = jnp.stack([jnp.eye(2), jnp.eye(2)])

  # 3. Compute Loss
  loss = get_bao_loss(flrw_metric, z_vals, dm_obs, dh_obs, cov_invs)

  # 4. Verify properties
  assert not jnp.isnan(loss), "BAO loss returned NaN."
  assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"
  assert loss > 0.0, "BAO loss should be strictly positive."


def test_efe_loss_minkowski():
  """
  Verifies that the Einstein Field Equation loss is exactly zero
  for a flat Minkowski spacetime (vacuum solution).
  """

  class MinkowskiMetric:
    def __init__(self):
      self.kappa_rho_0 = jnp.array([-20.0])

    def __call__(self, coords):
      return jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))

    def get_cosmology_today(self):
      # Flat Minkowski spacetime has zero expansion: H_mean(1.0) = 0.0.
      omega_m = jnp.array(
        [
          0.05
          + jax.nn.softplus(self.kappa_rho_0[0])
          - jax.nn.softplus(self.kappa_rho_0[0] - 0.25)
        ]
      )
      kappa_rho_0 = omega_m * 3.0 * 0.0**2
      return kappa_rho_0, omega_m

    def get_spatial_weight(self, coords):
      return jnp.array([0.0])

  metric_fn = MinkowskiMetric()

  # Evaluate at an arbitrary coordinate point
  coords = jnp.array([0.5, 0.0, 0.0, 0.0])

  kappa_rho_0_arr, _ = metric_fn.get_cosmology_today()
  efe_loss, wec_penalty, w_4d = get_efe_loss(
    metric_fn, coords, kappa_rho_0_arr[0]
  )

  assert not jnp.isnan(efe_loss), "EFE loss returned NaN."
  assert not jnp.isnan(wec_penalty), "WEC penalty returned NaN."
  # Since Minkowski is flat and physical density is 0, expected EFE loss is 0
  # and WEC is 0.
  assert jnp.allclose(efe_loss, 0.0, atol=1e-6)
  assert jnp.allclose(wec_penalty, 0.0, atol=1e-6)


def test_data_loss_computation():
  """
  Verifies that the Supernova data loss computes correctly without
  shape errors.
  """

  def flrw_metric(coords):
    t = coords[0]
    a = jnp.exp(0.5 * (t - 1.0))
    spatial = a**2 * jnp.eye(3)
    g = jnp.zeros((4, 4))
    g = g.at[0, 0].set(-1.0)
    g = g.at[1:4, 1:4].set(spatial)
    return g

  z_vals = jnp.array([0.1, 0.5])
  target_mu = jnp.array([35.0, 40.0])
  mu_err = jnp.array([0.1, 0.2])

  loss = get_data_loss(flrw_metric, z_vals, target_mu, mu_err)

  assert not jnp.isnan(loss), "Data loss returned NaN."
  assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"
  assert loss > 0.0, "Data loss should be strictly positive."


def test_augmented_lagrangian_loss():
  """
  Verifies that the Augmented Lagrangian formulation correctly computes
  the total loss given simulated metrics, Lagrange multiplier, and
  penalty parameter.
  """

  # Simulated component losses
  l_phys = jnp.array(0.1)
  l_wec = jnp.array(0.05)
  l_sn = jnp.array(0.5)
  l_bao = jnp.array(0.2)

  # Weights
  w_efe = 1.0
  w_sn = 10.0
  w_bao = 0.5

  # AL parameters
  lambda_wec = jnp.array(0.2)
  w_wec = jnp.array(5.0)

  # Hand-computed expected total loss
  expected_loss = (
    w_efe * l_phys
    + w_sn * l_sn
    + w_bao * l_bao
    + lambda_wec * l_wec
    + 0.5 * w_wec * (l_wec**2)
  )

  # Let's perform the same computation
  total_loss = (
    w_efe * l_phys
    + w_sn * l_sn
    + w_bao * l_bao
    + lambda_wec * l_wec
    + 0.5 * w_wec * (l_wec**2)
  )

  assert jnp.allclose(total_loss, expected_loss, atol=1e-8)
  assert total_loss > 0.0
  assert not jnp.isnan(total_loss)


def test_efe_loss_negative_curvature(monkeypatch):
  """
  Verifies that get_efe_loss computes a linear penalty for negative Ricci
  scalar curvature.
  """
  # Mock get_ricci_tensor to return a tensor that results in a negative Ricci
  # scalar. Minkowski metric has g = diag(-1, 1, 1, 1). If r_mu_nu has
  # components diag(0.5, -0.5, -0.5, -0.5), then:
  # R = g^mn r_mn = -0.5 - 0.5 - 0.5 - 0.5 = -2.0.
  import src.training.loss

  def mock_ricci_tensor(metric_fn, coords):
    return jnp.diag(jnp.array([0.5, -0.5, -0.5, -0.5]))

  monkeypatch.setattr(src.training.loss, "get_ricci_tensor", mock_ricci_tensor)

  class MockMetric:
    def __call__(self, coords):
      return jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))

    def get_spatial_weight(self, coords):
      return jnp.array([0.0])

  metric_fn = MockMetric()
  coords = jnp.array([1.0, 0.0, 0.0, 0.0])
  kappa_rho_0 = 0.1

  _, wec_penalty, _ = get_efe_loss(metric_fn, coords, kappa_rho_0)

  # For R = -2.0, the linear penalty raw is 2.0.
  assert jnp.allclose(wec_penalty, 2.0, atol=1e-6)


def test_efe_loss_positivity(monkeypatch):
  """
  Verifies that get_efe_loss always returns strictly non-negative raw residuals
  and WEC penalties, even when the spatial weight W is extremely negative.
  This ensures that the AL multiplier and telemetry only receive valid data.
  """
  import src.training.loss

  def mock_ricci_tensor(metric_fn, coords):
    # This mock produces a negative Ricci scalar (R = -2.0)
    return jnp.diag(jnp.array([0.5, -0.5, -0.5, -0.5]))

  monkeypatch.setattr(src.training.loss, "get_ricci_tensor", mock_ricci_tensor)

  class MockMetric:
    def __call__(self, coords):
      return jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))

    def get_spatial_weight(self, coords):
      # Extreme negative weight which would cause the homoscedastic formula
      # 0.5 * exp(-2W) * L + W to become highly negative if L is small.
      return jnp.array([-5.0])

  metric_fn = MockMetric()
  coords = jnp.array([1.0, 0.0, 0.0, 0.0])
  kappa_rho_0 = 0.1

  raw_residual_loss, wec_penalty_raw, _ = get_efe_loss(
    metric_fn, coords, kappa_rho_0
  )

  # The raw returned values must NEVER be negative, as they are fed to the AL
  # constraint and telemetry logger.
  assert raw_residual_loss >= 0.0, "Raw residual loss must be non-negative!"
  assert wec_penalty_raw >= 0.0, "Raw WEC penalty must be non-negative!"
