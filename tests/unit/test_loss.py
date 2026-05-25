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

  def minkowski_metric(coords):
    return jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))

  # Minkowski is a vacuum solution, so matter density must be 0
  minkowski_metric.kappa_rho_0 = jnp.array([0.0])

  # Evaluate at an arbitrary coordinate point
  coords = jnp.array([0.5, 0.0, 0.0, 0.0])

  loss = get_efe_loss(minkowski_metric, coords)

  assert not jnp.isnan(loss), "EFE loss returned NaN."
  # Minkowski is flat, but get_efe_loss enforces a hard baryonic matter floor
  # of kappa_rho_0 = 0.15 (Omega_m = 0.05 * 3). In Minkowski, the residual
  # is G_00 - T_00 = 0 - 0.15 = -0.15, leading to a mean squared loss of
  # (-0.15)^2 / 16 = 0.00140625.
  expected_loss = (0.15**2) / 16.0
  assert jnp.allclose(
    loss, expected_loss, atol=1e-6
  ), f"Expected {expected_loss} loss, got {loss}"


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
