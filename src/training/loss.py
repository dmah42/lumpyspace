"""
Loss functions for the FLRW control baseline.
"""

import jax.numpy as jnp

from src.core.tensors import get_ricci_scalar, get_ricci_tensor


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
  Note: This currently uses a placeholder for dL(z) until Phase 3 is ready.
  For the FLRW control, we can enforce a(t) matching directly.
  """
  # TODO: Integrate with Phase 3 Geodesics
  return 0.0
