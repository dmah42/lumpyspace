"""
Null geodesic equations for ray-tracing in arbitrary metrics.
"""

import os
from collections.abc import Callable
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# Mirrors diffrax's RealScalarLike, which is not exported from its public
# API (it lives in the private diffrax._custom_types). Spelled out here
# rather than imported from a private path that may move between versions.
RealScalarLike = int | float | jnp.ndarray | np.ndarray

# PHYSICAL SCALE:
# PHYSICAL SCALE:
# To maintain numerical stability in the PINN, we use dimensionless
# coordinates where 1.0 unit = 1000 Mpc. This ensures the solver
# stays within the network's stable domain while light travels
# cosmological distances.
L_UNIT = 1000.0
# We integrate null geodesics backwards out to 10,000 Mpc
L_MAX = -10000.0 / L_UNIT


def geodesic_system(
  affine_param: RealScalarLike,
  state: jnp.ndarray,
  args: tuple[Any, ...],
) -> jnp.ndarray:
  """
  Defines the first-order ODE system for geodesics:
  dx^mu/dl = k^mu
  dk^mu/dl = -Gamma^mu_{alpha beta} * k^alpha * k^beta
  """
  coords = state[:4]
  k = state[4:]
  metric_fn, jac_fn, _, _, should_log = args

  # Regularize metric to ensure invertibility during early training
  g = metric_fn(coords)
  g_reg = g + jnp.eye(4) * 1e-12
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

  # Only print in debug mode AND if this ray is selected for logging
  if os.environ.get("LUMPY_DEBUG") == "1":
    jax.lax.cond(
      should_log,
      lambda: jax.debug.print(
        "Step: l={l} t={t} | g00={g00} | gamma_max={gamma}",
        l=affine_param,
        t=coords[0],
        g00=g[0, 0],
        gamma=jnp.max(jnp.abs(gamma)),
      ),
      lambda: None,
    )

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

  # Use epsilon to avoid singular gradients at sqrt(0)
  u0_obs = -jnp.sqrt(jnp.abs(g_obs[0, 0]) + 1e-9)
  uk_obs = u0_obs * k_obs[0]

  u0_src = -jnp.sqrt(jnp.abs(g_src[0, 0]) + 1e-9)
  uk_src = u0_src * k_src[0]

  return uk_src / uk_obs


def _check_redshift_termination(
  t: jnp.ndarray, state: jnp.ndarray, args: tuple[Any, ...], **kwargs
) -> jnp.ndarray:
  metric_fn, _, initial_state, z_limit, should_log = args
  # Terminate if state becomes non-finite (NaN or Inf)
  is_invalid = jnp.logical_not(jnp.all(jnp.isfinite(state)))

  # Redshift-based termination
  z = get_redshift(metric_fn, initial_state, state) - 1.0
  over_z = z > z_limit

  # Domain-based termination (Task 4.1 safeguard)
  # Prevent solver from wandering outside the physical training domain [-4, 1]
  # where the neural metric is undefined/singular.
  out_of_bounds = state[0] < -5.0

  # DEBUG: Monitor event triggers for selected ray
  if os.environ.get("LUMPY_DEBUG") == "1":
    jax.lax.cond(
      should_log,
      lambda: jax.debug.print(
        "Event Check: t={t} z={z} | Invalid={inv} OverZ={oz} Bounds={oob}",
        t=state[0],
        z=z,
        inv=is_invalid,
        oz=over_z,
        oob=out_of_bounds,
      ),
      lambda: None,
    )

  return jnp.logical_or(jnp.logical_or(is_invalid, over_z), out_of_bounds)


@eqx.filter_jit
def _integrate_geodesic(
  metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
  initial_state: jnp.ndarray,
  l_max: float,
  z_limit: jnp.ndarray,
  should_log: bool = False,
) -> diffrax.Solution:
  # Pre-calculate Jacobian function once
  jac_fn = jax.jacfwd(metric_fn)

  term = diffrax.ODETerm(geodesic_system)
  solver = diffrax.Tsit5()
  stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-6)

  # Termination event to prevent crashing into singularities
  event = diffrax.Event(_check_redshift_termination)

  sol = diffrax.diffeqsolve(
    term,
    solver,
    t0=0.0,
    t1=l_max,
    dt0=-1.0,
    y0=initial_state,
    args=(metric_fn, jac_fn, initial_state, z_limit, should_log),
    stepsize_controller=stepsize_controller,
    event=event,
    saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, l_max, 50)),
    max_steps=4096,
  )
  return sol


def make_null_direction(
  g_obs: jnp.ndarray, direction: jnp.ndarray, kt: float = 1.0
) -> jnp.ndarray:
  """
  Builds a past-directed null 4-vector pointing along a sky direction.

  Solves g_mu_nu k^mu k^nu = 0 for k = (kt, -s * n_hat), which is a quadratic
  in s. The previous implementation assumed the ray ran along +x and used only
  g_00/g_11, silently ignoring the shift terms g_0i - so the initial ray was
  not actually null whenever the metric had off-diagonal time-space
  components.

  The spatial part is negated because the affine parameter runs backwards
  (l: 0 -> L_MAX < 0), so position advances along +n_hat into the past.
  """
  n = direction / jnp.sqrt(jnp.sum(jnp.square(direction)) + 1e-12)

  # g_mu_nu k^mu k^nu = A s^2 + B s + C with k_spatial = -s * n
  a = jnp.einsum("ij,i,j->", g_obs[1:4, 1:4], n, n)
  b = -2.0 * kt * jnp.einsum("i,i->", g_obs[0, 1:4], n)
  c = g_obs[0, 0] * kt**2

  disc = jnp.maximum(b**2 - 4.0 * a * c, 1e-12)
  s = (-b + jnp.sqrt(disc)) / (2.0 * a + 1e-12)

  return jnp.concatenate([jnp.array([kt]), -s * n])


def get_luminosity_distance(
  metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
  z_target: float,
  direction: jnp.ndarray,
  should_log: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """
  Calculates dL(z) by integrating null geodesics backwards from observer.

  Observer is placed at t=1.0 (today) in the normalized training domain.
  `direction` is a required unit 3-vector giving the source's position on
  the sky. It has no default: every caller must say where it is looking.

  Returns (dL_mpc, z_reached). dL_mpc is only meaningful where
  z_reached >= z_target - callers MUST check. This function no longer
  fabricates a distance for unreachable targets.
  """
  # 1. Setup Observer at t=1.0 (Present day)
  obs_coords = jnp.array([1.0, 0.0, 0.0, 0.0])
  g_obs = metric_fn(obs_coords)
  initial_k = make_null_direction(g_obs, direction)
  initial_state = jnp.concatenate([obs_coords, initial_k])

  # 2. Integrate
  # Use a safety margin for redshift termination
  z_limit = jnp.maximum(2.5, z_target + 0.2)
  sol = _integrate_geodesic(
    metric_fn, initial_state, L_MAX, z_limit, should_log=should_log
  )

  # 3. Physical Sampling (Robust against termination)
  # Sample exactly up to the termination point sol.t1.
  # This ensures all states are finite and physical.
  def compute_z_dl(state):
    z = get_redshift(metric_fn, initial_state, state) - 1.0
    # Use epsilon to avoid singular gradients at sqrt(0)
    r_prop = jnp.sqrt(jnp.sum(jnp.square(state[1:4])) + 1e-9)
    # Unitless luminosity distance
    dl_unit = (1.0 + z) * r_prop
    return z, dl_unit

  # Prevent JAX backward pass from evaluating 0.0 * Inf = NaN by masking inputs
  # first.
  # diffrax types ys as optional (it is None when SaveAt requests nothing);
  # we always pass SaveAt(ts=...), so this is an invariant, not a fallback.
  assert sol.ys is not None
  is_finite_state = jnp.all(jnp.isfinite(sol.ys), axis=-1)
  safe_ys = jnp.where(is_finite_state[:, None], sol.ys, initial_state)

  z_vals, dl_vals = jax.vmap(compute_z_dl)(safe_ys)

  # z along the ray is NOT monotonic in general - a contracting or lumpy
  # region produces blueshift segments, and jnp.interp silently returns
  # garbage on unsorted xp. Take a running maximum so the interpolation
  # precondition actually holds, and carry invalid tail states forward at
  # their last valid value rather than at the global max (which previously
  # made interp return that max for every z_target).
  z_masked = jnp.where(is_finite_state, z_vals, -jnp.inf)
  z_monotonic = jax.lax.cummax(z_masked)
  # Any leading -inf (shouldn't happen: state 0 is the observer) -> 0.0
  safe_z_vals = jnp.where(jnp.isfinite(z_monotonic), z_monotonic, 0.0)

  # Reachability is "how far did this ray actually get", i.e. the redshift at
  # the last finite state - NOT the best it managed anywhere along the path,
  # which would reward a ray that redshifts briefly then blueshifts back.
  # Note this must read z_vals, not the cummax above: safe_z_vals[-1] is by
  # construction the running maximum, which is exactly what we don't want.
  last_finite = jnp.max(
    jnp.where(is_finite_state, jnp.arange(len(is_finite_state)), 0)
  )
  z_reached = z_vals[last_finite]

  dl_masked = jnp.where(is_finite_state, dl_vals, 0.0)
  safe_dl_vals = jax.lax.cummax(dl_masked)

  # Diagnostics
  if os.environ.get("LUMPY_DEBUG") == "1":
    jax.lax.cond(
      should_log,
      lambda: jax.debug.print(
        "\n--- dL Diagnostics ---\n"
        "sol.result: {res}\n"
        "z_target: {z}\nz_reached: {zr}\n"
        "finite states: {nf}/{nt}",
        res=sol.result,
        z=z_target,
        zr=z_reached,
        nf=jnp.sum(is_finite_state),
        nt=len(is_finite_state),
      ),
      lambda: None,
    )

  # 4. Result via interpolation over the (now monotonic) reachable range.
  # For z_target > z_reached this clamps, and the value is meaningless - the
  # caller is responsible for masking on z_reached.
  result = jnp.interp(z_target, safe_z_vals, safe_dl_vals)

  # Convert back to physical Mpc. Reachability is reported separately rather
  # than folded into the distance, so the two failure modes (wrong distances
  # vs. insufficient redshift range) cannot trade off inside one scalar.
  return result * L_UNIT, z_reached


def get_bao_distances(
  metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
  z_target: float,
  should_log: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """
  Calculates the BAO Transverse (D_M) and Radial (D_H) distances in Mpc.
  To account for anisotropy, it shoots rays along the 3 principal spatial axes
  and returns the angular average.
  """
  # 1. Setup Observer at t=1.0 (Present day)
  obs_coords = jnp.array([1.0, 0.0, 0.0, 0.0])
  g_obs = metric_fn(obs_coords)
  kt = 1.0

  def shoot_ray(axis_idx):
    k_spatial = kt * jnp.sqrt(
      jnp.abs(g_obs[0, 0]) / jnp.abs(g_obs[axis_idx + 1, axis_idx + 1]) + 1e-9
    )
    initial_k = jnp.zeros(4)
    initial_k = initial_k.at[0].set(kt)
    initial_k = initial_k.at[axis_idx + 1].set(k_spatial)

    initial_state = jnp.concatenate([obs_coords, initial_k])

    # 2. Integrate backwards
    z_limit = jnp.maximum(2.5, z_target + 0.2)
    sol = _integrate_geodesic(
      metric_fn, initial_state, L_MAX, z_limit, should_log=should_log
    )

    # 3. Find Transverse Distance (D_M)
    def compute_z_dm(state):
      z = get_redshift(metric_fn, initial_state, state) - 1.0
      # D_M = r_prop for null geodesics in code units
      r_prop = jnp.sqrt(jnp.sum(jnp.square(state[1:4])) + 1e-9)
      return z, r_prop

    assert sol.ys is not None
    is_finite_state = jnp.all(jnp.isfinite(sol.ys), axis=-1)
    safe_ys = jnp.where(is_finite_state[:, None], sol.ys, initial_state)
    z_vals, dm_vals = jax.vmap(compute_z_dm)(safe_ys)

    safe_z_for_max = jnp.where(is_finite_state, z_vals, -1.0)
    max_z_found = jnp.max(safe_z_for_max)
    safe_z_vals = jnp.where(is_finite_state, z_vals, max_z_found)

    safe_dm_for_max = jnp.where(is_finite_state, dm_vals, 0.0)
    max_dm_found = jnp.max(safe_dm_for_max)
    safe_dm_vals = jnp.where(is_finite_state, dm_vals, max_dm_found)

    dm_unit = jnp.interp(z_target, safe_z_vals, safe_dm_vals)

    # 4. Find Radial Distance (D_H) at the target point
    t_vals = safe_ys[:, 0]
    t_target = jnp.interp(z_target, safe_z_vals, t_vals)

    coords_target = jnp.array([t_target, 0.0, 0.0, 0.0])
    g_target = metric_fn(coords_target)
    g_spatial = g_target[1:4, 1:4]
    g_spatial_inv = jnp.linalg.inv(g_spatial)

    dg_dt = jax.jacfwd(metric_fn)(coords_target)[:, :, 0]
    dg_spatial_dt = dg_dt[1:4, 1:4]

    # H_tensor with respect to coordinate time
    h_tensor = 0.5 * jnp.matmul(g_spatial_inv, dg_spatial_dt)

    # Convert to proper time (physical expansion rate)
    u0 = jnp.sqrt(jnp.abs(g_target[0, 0]) + 1e-9)
    h_tensor_phys = h_tensor / u0

    # Directional Hubble parameter
    h_dir = h_tensor_phys[axis_idx, axis_idx]

    h_recip = 1.0 / (jnp.abs(h_dir) + 1e-9)
    dh_unit = jnp.minimum(h_recip, -L_MAX) + 0.01 * jnp.maximum(
      0.0, h_recip - (-L_MAX)
    )

    # Apply penalty if target z was unreachable
    penalty = jnp.where(
      max_z_found < z_target, 10 * (z_target - max_z_found), 0.0
    )

    return dm_unit + penalty, dh_unit + penalty

  # Shoot rays along the 3 principal axes
  dm_x, dh_x = shoot_ray(0)
  dm_y, dh_y = shoot_ray(1)
  dm_z, dh_z = shoot_ray(2)

  dm_mean = (dm_x + dm_y + dm_z) / 3.0
  dh_mean = (dh_x + dh_y + dh_z) / 3.0

  return dm_mean * L_UNIT, dh_mean * L_UNIT
