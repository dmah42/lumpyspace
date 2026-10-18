"""
Loss functions for the FLRW control baseline.
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from src.core.kinematics import comoving_kinematics
from src.core.metric import MetricNN
from src.core.tensors import get_ricci_tensor
from src.physics.geodesics import get_bao_distances, get_luminosity_distance

METRIC_BAO = "l_bao"
METRIC_EXPAND = "l_expand"
METRIC_LOSS = "loss"
METRIC_OMEGA_M = "omega_m"
METRIC_PHYS = "l_phys"
METRIC_SHEAR = "l_shear"
METRIC_SN = "l_sn"
METRIC_SN_COVER = "l_sn_cover"
METRIC_SPATIAL = "l_spatial"
METRIC_WEC = "l_wec"
METRIC_MEAN_W_SPATIAL = "mean_w_spatial"

CONSTRAINT_METRICS = (METRIC_WEC, METRIC_EXPAND, METRIC_SHEAR, METRIC_SPATIAL)

PLANCK_RD_MPC = 147.09


def apply_spatial_weight(loss_val: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
  """Applies homoscedastic uncertainty weighting."""
  return 0.5 * jnp.exp(-2.0 * w) * loss_val + w


class EfeDiagnostics(NamedTuple):
  """Per-point field-equation quantities for the selection score."""

  # Mean square of G^mu_nu - kappa T^mu_nu, without the volume weighting the
  # loss applies.
  residual_sq: jnp.ndarray
  # Local matter term kappa * rho.
  kappa_rho: jnp.ndarray
  ricci_scalar: jnp.ndarray


def get_efe_loss(
  model: MetricNN,
  coords: jnp.ndarray,
  kappa_rho_0: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, EfeDiagnostics]:
  """
  Computes the residual of the Einstein Field Equations:
  G_mu_nu - 8*pi*G * T_mu_nu = 0

  For the FLRW control, we assume a vacuum or perfect fluid.
  Here we minimize the Einstein Tensor residual directly.

  Returns (raw_residual_loss, wec_penalty_raw, w_4d, diagnostics), where
  diagnostics feeds the selection score and plays no part in the loss.
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

  # Compute mixed Einstein tensor G^m_n for coordinate invariance
  g_mixed = jnp.matmul(g_inv, g_mu_nu)

  # In comoving coordinates, dust 4-velocity is u^mu = (1/sqrt(-g_00), 0, 0, 0).
  # The mixed Stress-Energy Tensor T^mu_nu = rho * u^mu * u_nu:
  # T^0_0 = -rho
  # T^0_i = rho * g_0i / (-g_00)
  # T^i_0 = 0, T^i_j = 0
  # Protect against a near-zero lapse collapsing during early training.
  neg_g00 = jnp.maximum(-g[0, 0], 1e-12)
  t_mixed = jnp.zeros((4, 4))
  t_mixed = t_mixed.at[0, 0].set(-current_density)
  t_mixed = t_mixed.at[0, 1:4].set(current_density * g[0, 1:4] / neg_g00)

  # Physics Residual (G^m_n - 8*pi*G * T^m_n = 0)
  unweighted_residual = g_mixed - t_mixed
  residual = unweighted_residual * gamma

  # Evaluate spatial weight
  w_4d = model.get_spatial_weight(coords)[0]

  # Raw EFE residual
  raw_residual_loss = jnp.mean(jnp.square(residual))

  # Enforce the Weak Energy Condition by penalizing negative Ricci scalar
  wec_penalty_raw = jnp.maximum(0.0, -r_scalar)

  diagnostics = EfeDiagnostics(
    residual_sq=jnp.mean(jnp.square(unweighted_residual)),
    kappa_rho=current_density,
    ricci_scalar=r_scalar,
  )
  return raw_residual_loss, wec_penalty_raw, w_4d, diagnostics


def get_data_loss(
  metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
  redshifts: jnp.ndarray,
  target_mu: jnp.ndarray,
  mu_err: jnp.ndarray,
  directions: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """
  Data loss against Supernova distance modulus.
  dL is computed via the geodesic ray-tracer along each source's own sky
  direction. mu = 5 * log10(dL_mpc) + 25

  Returns (l_sn, l_sn_cover, cover_sq):
    l_sn:       weighted chi-squared. Unreachable supernovae contribute zero
                to the numerator but STILL COUNT in the denominator - with a
                moving denominator the model could shed a badly-fit source by
                making it unreachable, which is the same perverse incentive
                the old compass had.
    l_sn_cover: mean redshift shortfall over all supernovae. A hinge, so it
                is exactly zero once every ray reaches its source.
    cover_sq:   mean squared shortfall, for the selection score. Plays no part
                in the loss.

  These are kept separate on purpose. Folding shortfall into the returned
  distance (the old "Optimizer Compass") made the two failure modes fungible
  inside one scalar, and the optimizer settled in a basin where neither was
  solved - see Task 4.12 in docs/technical_design.md.
  """
  # To avoid a terminal firehose, we only log diagnostics for the first
  # supernova in each batch.
  should_log_batch = jnp.zeros(len(redshifts), dtype=bool).at[0].set(True)

  # Vectorize over redshifts, directions and logging flags
  v_get_dl = jax.vmap(
    lambda z, d, sl: get_luminosity_distance(
      metric_fn, z, direction=d, should_log=sl
    )
  )
  dl_mpc, z_reached = v_get_dl(redshifts, directions, should_log_batch)

  # A source is only fit if its ray actually got there. No abs() - a negative
  # distance is a failure, not something to quietly rectify.
  reached = z_reached >= redshifts
  mu_pred = 5.0 * jnp.log10(jnp.maximum(dl_mpc, 1e-6)) + 25.0
  chi2 = jnp.square((mu_pred - target_mu) / mu_err)

  l_sn = jnp.sum(jnp.where(reached, chi2, 0.0)) / len(redshifts)
  shortfall = jnp.maximum(0.0, redshifts - z_reached)
  l_sn_cover = jnp.mean(shortfall)
  cover_sq = jnp.mean(jnp.square(shortfall))

  return l_sn, l_sn_cover, cover_sq


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
  g = model(coords)

  # Compute Jacobian to get time and spatial derivatives
  # shape of jac_g: (4, 4, 4) -> (mu, nu, k) where k is the derivative index
  jac_g = jax.jacfwd(model)(coords)

  # Expansion and shear of the comoving matter, including the shift terms.
  kinematics = comoving_kinematics(g, jac_g)

  # 1. Expansion Penalty: Enforce expansion (no contraction) on all three axes
  l_expand = jnp.sum(jnp.maximum(0.0, -kinematics.h_dir) ** 2)

  # 2. Shear Penalty: max(0, sigma^2 - 1e-5)^2
  l_shear = jnp.maximum(0.0, kinematics.sigma_sq - 1e-5) ** 2

  # 3. Spatial Homogeneity Penalty
  # sum over spatial derivatives (k=1,2,3) and all metric components
  spatial_grads = jac_g[:, :, 1:4]
  sum_spatial_grads_sq = jnp.sum(jnp.square(spatial_grads))
  l_spatial = jnp.maximum(0.0, sum_spatial_grads_sq - 1e-5) ** 2

  return l_expand, l_shear, l_spatial
