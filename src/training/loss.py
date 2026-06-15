"""
Loss functions for the FLRW control baseline.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from src.core.metric import MetricNN
from src.core.tensors import get_ricci_tensor
from src.physics.geodesics import get_bao_distances, get_luminosity_distance

METRIC_LOSS = "loss"
METRIC_PHYS = "l_phys"
METRIC_WEC = "l_wec"
METRIC_EXPAND = "l_expand"
METRIC_SHEAR = "l_shear"
METRIC_SPATIAL = "l_spatial"
METRIC_SN = "l_sn"
METRIC_BAO = "l_bao"
METRIC_OMEGA_M = "omega_m"
METRIC_MEAN_W_SPATIAL = "mean_w_spatial"

CONSTRAINT_METRICS = (METRIC_WEC, METRIC_EXPAND, METRIC_SHEAR, METRIC_SPATIAL)

PLANCK_RD_MPC = 147.09


def apply_spatial_weight(loss_val: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
  """Applies homoscedastic uncertainty weighting."""
  return 0.5 * jnp.exp(-2.0 * w) * loss_val + w


def get_efe_loss(
  model: MetricNN,
  coords: jnp.ndarray,
  kappa_rho_0: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """
  Computes the residual of the Einstein Field Equations:
  G_mu_nu - 8*pi*G * T_mu_nu = 0

  For the FLRW control, we assume a vacuum or perfect fluid.
  Here we minimize the Einstein Tensor residual directly.
  """
  g = model(coords)
  g_inv = jnp.linalg.inv(g)
  r_mu_nu = get_ricci_tensor(model, coords)
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

  # Evaluate spatial weight
  w_4d = model.get_spatial_weight(coords)[0]

  # Raw EFE residual
  raw_residual_loss = jnp.mean(jnp.square(residual))

  # Enforce the Weak Energy Condition by penalizing negative Ricci scalar
  wec_penalty_raw = jnp.maximum(0.0, -r_scalar)

  return raw_residual_loss, wec_penalty_raw, w_4d


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


def get_cmb_loss(
  model: MetricNN,
  coords: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """
  Computes the three CMB constraints at the deep past boundary.

  Expansion, Shear (Isotropy), and Spatial Homogeneity.
  Returns raw loss values before Augmented Lagrangian weighting.
  """
  # Extract metric and spatial metric
  g = model(coords)
  g_spatial = g[1:4, 1:4]
  g_spatial_inv = jnp.linalg.inv(g_spatial)

  # Lapse function N = sqrt(-g_00)
  lapse = jnp.sqrt(-g[0, 0])

  # Compute Jacobian to get time and spatial derivatives
  # shape of jac_g: (4, 4, 4) -> (mu, nu, k) where k is the derivative index
  jac_g = jax.jacfwd(model)(coords)

  # Time derivative of spatial metric (dot_g_ij)
  # index 0 is time 't'
  dot_g_spatial = jac_g[1:4, 1:4, 0]

  # Extrinsic curvature K_ij = -1/(2N) * dot_g_ij
  k_ij = -1.0 / (2.0 * lapse) * dot_g_spatial

  # 1. Expansion Penalty: Enforce expansion (no contraction) on all three axes
  # H_i = -K^i_i (diagonal of K_mixed).
  # Computed efficiently to avoid full matmul.
  h_dir = -jnp.sum(g_spatial_inv * jnp.transpose(k_ij), axis=-1)
  l_expand = jnp.sum(jnp.maximum(0.0, -h_dir) ** 2)

  # 2. Shear Penalty: max(0, sigma^2 - 1e-5)^2
  # sigma_ij = K_ij - 1/3 * Tr(K) * g_ij
  # Tr(K) is exactly -sum(h_dir)
  sigma_ij = k_ij + 1.0 / 3.0 * jnp.sum(h_dir) * g_spatial

  # sigma^2 = 1/2 * sigma_ij * g^{ia} g^{jb} sigma_ab
  # which is 1/2 * Tr( (g_spatial_inv @ sigma_ij)^2 )
  sigma_mixed = jnp.matmul(g_spatial_inv, sigma_ij)
  # Optimize Tr(A @ A) -> sum(A * A.T) to save a matmul
  sigma_sq = 0.5 * jnp.sum(sigma_mixed * jnp.transpose(sigma_mixed))
  l_shear = jnp.maximum(0.0, sigma_sq - 1e-5) ** 2

  # 3. Spatial Homogeneity Penalty
  # sum over spatial derivatives (k=1,2,3) and all metric components
  spatial_grads = jac_g[:, :, 1:4]
  sum_spatial_grads_sq = jnp.sum(jnp.square(spatial_grads))
  l_spatial = jnp.maximum(0.0, sum_spatial_grads_sq - 1e-5) ** 2

  return l_expand, l_shear, l_spatial
