"""
Null geodesic equations for ray-tracing in arbitrary metrics.
"""

from typing import Any, Callable, Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp


def geodesic_system(
  affine_param: jnp.ndarray, state: jnp.ndarray, args: Tuple[Any, ...]
) -> jnp.ndarray:
  """
  Defines the first-order ODE system for geodesics:
  dx^mu/dl = k^mu
  dk^mu/dl = -Gamma^mu_{alpha beta} * k^alpha * k^beta
  """
  coords = state[:4]
  k = state[4:]
  metric_fn, jac_fn, _, _ = args

  # Regularize metric to ensure invertibility during early training
  g = metric_fn(coords)
  g_reg = g + jnp.eye(4) * 1e-6
  g_inv = jnp.linalg.inv(g_reg)
  dg = jac_fn(coords)

  # Gamma^a_{bc} = 0.5 * g^{ad} (d_b g_{cd} + d_c g_{bd} - d_d g_{bc})
  gamma = 0.5 * (
    jnp.einsum("ad,cdb->abc", g_inv, dg)
    + jnp.einsum("ad,bdc->abc", g_inv, dg)
    - jnp.einsum("ad,bcd->abc", g_inv, dg)
  )

  dcoords_dl = k
  dk_dl = -jnp.einsum("abc,b,c->a", gamma, k, k)

  jax.debug.print("Step: l={l} t={t}", l=affine_param, t=coords[0])

  return jnp.concatenate([dcoords_dl, dk_dl])


def get_redshift(
  metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
  state_obs: jnp.ndarray,
  state_source: jnp.ndarray,
) -> jnp.ndarray:
  """
  Calculates redshift 1+z = (u . k)_source / (u . k)_obs.
  """
  coords_obs = state_obs[:4]
  k_obs = state_obs[4:]
  g_obs = metric_fn(coords_obs)

  coords_src = state_source[:4]
  k_src = state_source[4:]
  g_src = metric_fn(coords_src)

  u0_obs = -jnp.sqrt(jnp.abs(g_obs[0, 0]))
  uk_obs = u0_obs * k_obs[0]

  u0_src = -jnp.sqrt(jnp.abs(g_src[0, 0]))
  uk_src = u0_src * k_src[0]

  return uk_src / uk_obs


def _check_redshift_termination(
  t: jnp.ndarray, state: jnp.ndarray, args: Tuple[Any, ...], **kwargs
) -> jnp.ndarray:
  metric_fn, _, initial_state, z_limit = args
  # Terminate if state becomes NaN to prevent infinite streams of errors
  is_nan = jnp.any(jnp.isnan(state))
  z = get_redshift(metric_fn, initial_state, state) - 1.0
  return jnp.logical_or(is_nan, z > z_limit)


@eqx.filter_jit
def _integrate_geodesic(
  metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
  initial_state: jnp.ndarray,
  l_max: float,
  z_limit: float,
) -> diffrax.Solution:
  # Pre-calculate Jacobian function once
  jac_fn = jax.jacfwd(metric_fn)

  term = diffrax.ODETerm(geodesic_system)
  solver = diffrax.Tsit5()
  stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)

  # Termination event to prevent crashing into singularities
  event = diffrax.Event(_check_redshift_termination)

  sol = diffrax.diffeqsolve(
    term,
    solver,
    t0=0.0,
    t1=l_max,
    dt0=-1.0,
    y0=initial_state,
    args=(metric_fn, jac_fn, initial_state, z_limit),
    stepsize_controller=stepsize_controller,
    event=event,
    saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, l_max, 50)),
    max_steps=4096,
  )
  return sol


def get_luminosity_distance(
  metric_fn: Callable[[jnp.ndarray], jnp.ndarray], z_target: float
) -> jnp.ndarray:
  """
  Calculates dL(z) by integrating null geodesics backwards from observer.
  Observer is placed at t=1.0 (today) in the normalized training domain.
  """
  # 1. Setup Observer at t=1.0 (Present day)
  obs_coords = jnp.array([1.0, 0.0, 0.0, 0.0])
  g_obs = metric_fn(obs_coords)
  kt = 1.0
  # Ensure kx is null: g00*kt^2 + g11*kx^2 = 0 => kx = kt * sqrt(-g00/g11)
  kx = kt * jnp.sqrt(jnp.abs(g_obs[0, 0]) / g_obs[1, 1])
  initial_k = jnp.array([kt, kx, 0.0, 0.0])
  initial_state = jnp.concatenate([obs_coords, initial_k])

  # 2. Integrate
  # Use a safety margin for redshift termination
  z_limit = jnp.maximum(2.5, z_target + 0.2)
  l_max = -3000.0
  sol = _integrate_geodesic(metric_fn, initial_state, l_max, z_limit)

  # 3. Map trajectory to Redshift and Distance
  def compute_z_dl(state):
    z = get_redshift(metric_fn, initial_state, state) - 1.0
    r_prop = jnp.sqrt(jnp.sum(jnp.square(state[1:4])))
    dl = (1.0 + z) * r_prop
    return z, dl

  z_vals, dl_vals = jax.vmap(compute_z_dl)(sol.ys)

  # 4. Result via interpolation
  return jnp.interp(z_target, z_vals, dl_vals)
