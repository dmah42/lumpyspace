"""
Loss functions for the FLRW control baseline.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from src.core.tensors import get_ricci_scalar, get_ricci_tensor
from src.physics.geodesics import get_luminosity_distance


def get_efe_loss(
  metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
  coords: jnp.ndarray,
) -> jnp.ndarray:
  """
  Computes the residual of the Einstein Field Equations:
  G_mu_nu - 8*pi*G * T_mu_nu = 0

  For the FLRW control, we assume a vacuum or perfect fluid.
  Here we minimize the Einstein Tensor residual directly.
  """
  g = metric_fn(coords)
  r_mu_nu = get_ricci_tensor(metric_fn, coords)
  r_scalar = get_ricci_scalar(metric_fn, coords)

  # Einstein Tensor G_mu_nu = R_mu_nu - 0.5 * R * g_mu_nu
  g_mu_nu = r_mu_nu - 0.5 * r_scalar * g

  # Physics Residual
  # residual = G_mu_nu
  # (Ignoring T_mu_nu for the simplest control baseline)
  residual = g_mu_nu

  return jnp.mean(jnp.square(residual))


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
