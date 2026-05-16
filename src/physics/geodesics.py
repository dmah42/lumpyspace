"""
Null geodesic equations for ray-tracing in arbitrary metrics.
"""

import diffrax
import jax
import jax.numpy as jnp

from src.core.tensors import get_christoffel_symbols


def geodesic_system(affine_param, state, args):
  """
  Defines the first-order ODE system for geodesics:
  dx^mu/dl = k^mu
  dk^mu/dl = -Gamma^mu_{alpha beta} * k^alpha * k^beta

  state: [t, x, y, z, kt, kx, ky, kz]
  args: (metric_fn)
  """
  coords = state[:4]
  k = state[4:]
  metric_fn = args

  # Get Christoffel symbols at current coordinates
  gamma = get_christoffel_symbols(metric_fn, coords)

  # dx/dl = k
  dcoords_dl = k

  # dk/dl = -Gamma * k * k
  dk_dl = -jnp.einsum("abc,b,c->a", gamma, k, k)

  return jnp.concatenate([dcoords_dl, dk_dl])


def get_redshift(metric_fn, state_obs, state_source):
  """
  Calculates redshift 1+z = (u . k)_source / (u . k)_obs.
  Assumes observers are comoving with the fluid: u = [-1/sqrt(|g00|), 0, 0, 0].
  """
  coords_obs = state_obs[:4]
  k_obs = state_obs[4:]
  g_obs = metric_fn(coords_obs)

  coords_src = state_source[:4]
  k_src = state_source[4:]
  g_src = metric_fn(coords_src)

  # u_obs = [1/sqrt(|g00|), 0, 0, 0] (Lowered index)
  # (u . k)_obs = u_mu k^mu = u_0 * k^0
  u0_obs = -jnp.sqrt(jnp.abs(g_obs[0, 0]))
  uk_obs = u0_obs * k_obs[0]

  u0_src = -jnp.sqrt(jnp.abs(g_src[0, 0]))
  uk_src = u0_src * k_src[0]

  return uk_src / uk_obs


def get_luminosity_distance(metric_fn, z_target):
  """
  Calculates dL(z) by integrating null geodesics backwards from observer.
  dL = (1+z) * r_prop
  """
  # 1. Setup Observer
  g_obs = metric_fn(jnp.zeros(4))
  kt = 1.0
  # Null condition: -|g00|*kt^2 + gxx*kx^2 = 0
  kx = kt * jnp.sqrt(jnp.abs(g_obs[0, 0]) / g_obs[1, 1])
  initial_k = jnp.array([kt, kx, 0.0, 0.0])
  initial_state = jnp.concatenate([jnp.zeros(4), initial_k])

  # 2. Setup Integrator
  term = diffrax.ODETerm(geodesic_system)
  solver = diffrax.Tsit5()

  # 3. Integrate to find redshift
  # We integrate backwards in affine parameter l (l < 0)
  # until we hit z_target.
  # For now, we integrate for a range of l and interpolate.
  # TODO: Use diffrax.Event for precise z-triggering
  sol = diffrax.diffeqsolve(
    term,
    solver,
    t0=0.0,
    t1=-5.0,  # Integrate backwards
    dt0=-0.1,
    y0=initial_state,
    args=metric_fn,
    saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, -5.0, 50)),
  )

  # 4. Map l to Redshift
  def compute_z_at_l(state):
    return get_redshift(metric_fn, initial_state, state) - 1.0

  redshifts = jax.vmap(compute_z_at_l)(sol.ys)

  # 5. Extract dL = (1+z) * r_prop
  # r_prop = sqrt(x^2 + y^2 + z^2) at the source
  r_prop = jnp.sqrt(jnp.sum(jnp.square(sol.ys[:, 1:4]), axis=1))
  dl_values = (1.0 + redshifts) * r_prop

  # 6. Interpolate to get dL at z_target
  return jnp.interp(z_target, redshifts[::-1], dl_values[::-1])
