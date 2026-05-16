"""
Loss functions for the FLRW control baseline.
"""

import jax
import jax.numpy as jnp

from src.core.tensors import get_ricci_scalar, get_ricci_tensor
from src.physics.geodesics import get_luminosity_distance


def get_efe_loss(metric_fn, coords, lam=0.7, g_newton=1.0):
  """
  Computes the residual of the Einstein Field Equations:
  G_mu_nu + Lambda * g_mu_nu - 8*pi*G * T_mu_nu = 0

  For the FLRW control, we assume a vacuum + Lambda or perfect fluid.
  Here we minimize the Einstein Tensor residual directly.
  """
  g = metric_fn(coords)
  r_mu_nu = get_ricci_tensor(metric_fn, coords)
  r_scalar = get_ricci_scalar(metric_fn, coords)

  # Einstein Tensor G_mu_nu = R_mu_nu - 0.5 * R * g_mu_nu
  g_mu_nu = r_mu_nu - 0.5 * r_scalar * g

  # Physics Residual with Cosmological Constant Lambda
  # residual = G_mu_nu + Lambda * g_mu_nu
  # (Ignoring T_mu_nu for the simplest control baseline)
  residual = g_mu_nu + lam * g

  return jnp.mean(jnp.square(residual))


def get_data_loss(metric_fn, redshifts, target_mu):
  """
  Data loss against Supernova distance modulus.
  dL is computed via the geodesic ray-tracer.
  mu = 5 * log10(dL_mpc) + 25
  """
  # Vectorize over redshifts
  v_get_dl = jax.vmap(lambda z: get_luminosity_distance(metric_fn, z))
  dl_mpc = v_get_dl(redshifts)

  # Avoid log10(0)
  mu_pred = 5.0 * jnp.log10(dl_mpc + 1e-6) + 25.0

  return jnp.mean(jnp.square(mu_pred - target_mu))
