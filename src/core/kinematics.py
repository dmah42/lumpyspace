"""
Kinematics of the comoving congruence: the expansion and shear measured by
observers at fixed spatial coordinates, u^mu = (1 / sqrt(-g_00), 0, 0, 0).
This is the same congruence the comoving dust stress-energy tensor in
src/training/loss.py describes.
"""

from typing import NamedTuple

import jax.numpy as jnp

# Floor on -g_00, matching get_efe_loss, so a lapse that collapses during
# early training stays finite.
_NEG_G00_FLOOR = 1e-12


class Kinematics(NamedTuple):
  """Strain rate of the comoving congruence at one point."""

  # Mixed strain-rate tensor D^i_j, positive for expansion.
  strain_rate: jnp.ndarray
  # Diagonal of strain_rate: the expansion rate along x, y and z.
  h_dir: jnp.ndarray
  # theta / 3, the rate of change of the cube root of comoving volume.
  h_mean: jnp.ndarray
  # Shear scalar sigma^2 = 1/2 sigma^i_j sigma^j_i.
  sigma_sq: jnp.ndarray


def comoving_spatial_metric(g: jnp.ndarray) -> jnp.ndarray:
  """
  Spatial metric in the rest frame of observers at fixed spatial coordinates,
  h_ij = g_ij - g_0i g_0j / g_00. It equals g_ij when the shift g_0i is zero.
  """
  neg_g00 = jnp.maximum(-g[0, 0], _NEG_G00_FLOOR)
  shift = g[0, 1:4]
  return g[1:4, 1:4] + jnp.outer(shift, shift) / neg_g00


def comoving_kinematics(g: jnp.ndarray, dg: jnp.ndarray) -> Kinematics:
  """
  Expansion and shear of the comoving congruence.

  g is the metric at a point and dg its coordinate derivatives,
  dg[mu, nu, k] = d g_{mu nu} / d x^k, as returned by jax.jacfwd(metric_fn).

  Observers at fixed spatial coordinates measure proper separations with the
  rest-frame spatial metric h_ij and proper time d tau = sqrt(-g_00) dt, so
  their strain rate is D_ij = d_t h_ij / (2 sqrt(-g_00)) (Landau & Lifshitz,
  The Classical Theory of Fields, section 84). With zero shift this is
  d_t g_ij / (2 sqrt(-g_00)).
  """
  neg_g00 = jnp.maximum(-g[0, 0], _NEG_G00_FLOOR)
  shift = g[0, 1:4]
  h = g[1:4, 1:4] + jnp.outer(shift, shift) / neg_g00

  dg_dt = dg[:, :, 0]
  dshift_dt = dg_dt[0, 1:4]
  dh_dt = (
    dg_dt[1:4, 1:4]
    + (jnp.outer(dshift_dt, shift) + jnp.outer(shift, dshift_dt)) / neg_g00
    + jnp.outer(shift, shift) * dg_dt[0, 0] / neg_g00**2
  )

  strain_rate = jnp.linalg.solve(h, dh_dt) / (2.0 * jnp.sqrt(neg_g00))
  h_dir = jnp.diag(strain_rate)
  h_mean = jnp.trace(strain_rate) / 3.0
  sigma = strain_rate - h_mean * jnp.eye(3)
  sigma_sq = 0.5 * jnp.sum(sigma * sigma.T)
  return Kinematics(
    strain_rate=strain_rate, h_dir=h_dir, h_mean=h_mean, sigma_sq=sigma_sq
  )
