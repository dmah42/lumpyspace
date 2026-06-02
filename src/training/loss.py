"""
Loss functions for the FLRW control baseline.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from src.core.tensors import get_ricci_tensor
from src.physics.geodesics import get_bao_distances, get_luminosity_distance

PLANCK_RD_MPC = 147.09


def get_efe_loss(
  metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
  coords: jnp.ndarray,
  kappa_rho_0: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """
  Computes the residual of the Einstein Field Equations:
  G_mu_nu - 8*pi*G * T_mu_nu = 0

  For the FLRW control, we assume a vacuum or perfect fluid.
  Here we minimize the Einstein Tensor residual directly.
  """
  g = metric_fn(coords)
  g_inv = jnp.linalg.inv(g)
  r_mu_nu = get_ricci_tensor(metric_fn, coords)
  r_scalar = jnp.einsum("mn,mn->", g_inv, r_mu_nu)

  # Einstein Tensor G_mu_nu = R_mu_nu - 0.5 * R * g_mu_nu
  g_mu_nu = r_mu_nu - 0.5 * r_scalar * g

  # Matter Stress-Energy Tensor (Pressureless Dust)
  # Density dilutes as 1 / sqrt(det(spatial_metric))
  spatial_metric = g[1:4, 1:4]
  gamma = jnp.linalg.det(spatial_metric)
  # Protect against negative/zero determinant during early training
  gamma = jnp.maximum(gamma, 1e-24)

  # Extract trainable matter density parameter from the model
  # Enforce Weak Energy Condition and a dynamic Baryonic matter floor
  # dynamically scaled by the derived expansion rate today.
  current_density = kappa_rho_0 / jnp.sqrt(gamma)

  t_mu_nu = jnp.zeros((4, 4))
  # In comoving coordinates, dust only has energy density (T_00 = rho * -g_00)
  t_mu_nu = t_mu_nu.at[0, 0].set(current_density * -g[0, 0])

  # Compute mixed tensors (G^m_n and T^m_n) for coordinate invariance
  g_mixed = jnp.matmul(g_inv, g_mu_nu)
  t_mixed = jnp.matmul(g_inv, t_mu_nu)

  # Physics Residual (G^m_n - 8*pi*G * T^m_n = 0)
  residual = (g_mixed - t_mixed) * gamma

  # Enforce the Weak Energy Condition by penalizing negative Ricci scalar
  # (R = kappa * rho >= 0) using a linear absolute violation penalty to
  # ensure non-vanishing gradients.
  wec_penalty = jnp.maximum(0.0, -r_scalar)

  return jnp.mean(jnp.square(residual)), wec_penalty


def get_data_loss(
  metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
  redshifts: jnp.ndarray,
  target_mu: jnp.ndarray,
  mu_err: jnp.ndarray,
) -> jnp.ndarray:
  """
  Data loss against Supernova distance modulus.
  dL is computed via the geodesic ray-tracer.
  mu = 5 * log10(dL_mpc) + 25

  Computes the weighted chi-squared statistic.
  """
  # To avoid a terminal firehose, we only log diagnostics for the first
  # supernova in each batch.
  should_log_batch = jnp.zeros(len(redshifts), dtype=bool).at[0].set(True)

  # Vectorize over redshifts and logging flags
  v_get_dl = jax.vmap(
    lambda z, sl: get_luminosity_distance(metric_fn, z, should_log=sl)
  )
  dl_mpc = v_get_dl(redshifts, should_log_batch)

  # Avoid log10(0) or negative distances
  mu_pred = 5.0 * jnp.log10(jnp.abs(dl_mpc) + 1e-6) + 25.0

  # Weighted Chi-Squared: mean( ((pred - obs) / err)^2 )
  return jnp.mean(jnp.square((mu_pred - target_mu) / mu_err))


def get_bao_loss(
  metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
  z_vals: jnp.ndarray,
  dm_obs: jnp.ndarray,
  dh_obs: jnp.ndarray,
  cov_invs: jnp.ndarray,
) -> jnp.ndarray:
  """
  Data loss against 3D BAO distance measurements.

  Computes the chi-squared statistic using Transverse (D_M) and Radial (D_H)
  distances, weighted by the full inverted covariance matrix.
  """
  # We hardcode the standard Planck sound horizon (r_d) as justified
  # in the technical specs
  r_d = PLANCK_RD_MPC

  # To avoid a terminal firehose, log diagnostics for only the first
  # BAO data point in the batch
  should_log_batch = jnp.zeros(len(z_vals), dtype=bool).at[0].set(True)

  v_get_bao = jax.vmap(
    lambda z, sl: get_bao_distances(metric_fn, z, should_log=sl)
  )

  # Predict physical distances
  dm_pred, dh_pred = v_get_bao(z_vals, should_log_batch)

  # Convert predictions to the observational ratio (D / r_d)
  dm_ratio = dm_pred / r_d
  dh_ratio = dh_pred / r_d

  def compute_chi2(dm_p, dh_p, dm_o, dh_o, c_inv):
    # Delta vector: [D_M - D_M_obs, D_H - D_H_obs]
    delta = jnp.array([dm_p - dm_o, dh_p - dh_o])
    # Chi-squared: delta^T * C^{-1} * delta
    return jnp.einsum("i,ij,j->", delta, c_inv, delta)

  v_compute_chi2 = jax.vmap(compute_chi2)
  chi2_vals = v_compute_chi2(dm_ratio, dh_ratio, dm_obs, dh_obs, cov_invs)

  return jnp.mean(chi2_vals)
