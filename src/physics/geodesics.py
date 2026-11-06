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

from src.core.kinematics import comoving_four_velocity, comoving_spatial_metric

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


def geodesic_deviation_system(
  affine_param: RealScalarLike,
  state: jnp.ndarray,
  args: tuple[Any, ...],
) -> jnp.ndarray:
  """
  A null geodesic and the deviation vectors to its neighbouring rays.

  state is the ray (x^mu, k^mu) followed by one (xi^mu, d xi^mu / dl) per
  neighbour. The deviations evolve by the geodesic flow linearised about the
  ray, which is the geodesic deviation equation in coordinates; the
  curvature enters through the derivatives of the Christoffel symbols.
  """
  ray = state[:8]
  deviations = state[8:].reshape(-1, 8)
  flow, linear_flow = jax.linearize(
    lambda s: geodesic_system(affine_param, s, args), ray
  )
  return jnp.concatenate([flow, jax.vmap(linear_flow)(deviations).reshape(-1)])


def comoving_frequency(g: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
  """Frequency -u . k of the photon k seen by the comoving observer u."""
  return -jnp.einsum("mn,m,n->", g, comoving_four_velocity(g), k)


def get_redshift(
  metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
  state_obs: jnp.ndarray,
  state_source: jnp.ndarray,
) -> jnp.ndarray:
  """
  Calculates redshift 1+z = (u . k)_source / (u . k)_obs, with u the
  4-velocity of the comoving matter at each end.
  """
  coords_obs = state_obs[:4]
  k_obs = state_obs[4:8]
  coords_src = state_source[:4]
  k_src = state_source[4:8]
  return comoving_frequency(metric_fn(coords_src), k_src) / comoving_frequency(
    metric_fn(coords_obs), k_obs
  )


def _check_redshift_termination(
  t: jnp.ndarray, state: jnp.ndarray, args: tuple[Any, ...], **kwargs
) -> jnp.ndarray:
  metric_fn, _, initial_ray, z_limit, should_log = args
  # Terminate if state becomes non-finite (NaN or Inf)
  is_invalid = jnp.logical_not(jnp.all(jnp.isfinite(state)))

  # Redshift-based termination
  z = get_redshift(metric_fn, initial_ray, state[:8]) - 1.0
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

  # A state longer than the ray itself carries deviation vectors.
  system = (
    geodesic_deviation_system if initial_state.shape[0] > 8 else geodesic_system
  )
  term = diffrax.ODETerm(system)
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
    args=(metric_fn, jac_fn, initial_state[:8], z_limit, should_log),
    stepsize_controller=stepsize_controller,
    event=event,
    saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, l_max, 50)),
    max_steps=4096,
  )
  return sol


def make_null_direction(
  g_obs: jnp.ndarray, direction: jnp.ndarray
) -> jnp.ndarray:
  """
  The null 4-vector of light reaching the observer from a sky direction.

  The observer is the comoving matter, with 4-velocity u, and its sky is laid
  out on a spatial triad orthonormal in the rest-frame metric h. The Cholesky
  factor h = L L^T gives the triad L^-T, which lies along the coordinate axes
  when h is isotropic. `direction` is a 3-vector in that triad, normalised
  here. With m the unit vector towards the source, orthogonal to u,
  k = u - m is null and has unit frequency for the observer, -u . k = 1.

  The spatial part of k points away from the source. The affine parameter
  runs backwards (l: 0 -> L_MAX < 0), so the ray advances towards the source
  into the past.
  """
  n = direction / jnp.sqrt(jnp.sum(jnp.square(direction)) + 1e-12)
  chol = jnp.linalg.cholesky(comoving_spatial_metric(g_obs))
  m_spatial = jnp.linalg.solve(chol.T, n)
  # Orthogonal to u: g_0mu m^mu = 0.
  m_time = -jnp.dot(g_obs[0, 1:4], m_spatial) / g_obs[0, 0]
  m = jnp.concatenate([m_time[None], m_spatial])
  return comoving_four_velocity(g_obs) - m


def _sky_axes(n: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Two unit 3-vectors orthogonal to the unit vector n and to each other."""
  helper = jnp.where(
    jnp.abs(n[0]) < 0.9, jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0])
  )
  e1 = jnp.cross(n, helper)
  e1 = e1 / jnp.sqrt(jnp.sum(jnp.square(e1)))
  return e1, jnp.cross(n, e1)


def area_distance(
  metric_fn: Callable[[jnp.ndarray], jnp.ndarray], state: jnp.ndarray
) -> jnp.ndarray:
  """
  The area distance d_A at a point on a ray traced with two deviation
  vectors, each started at the observer with zero separation and unit rate
  per radian of sky angle.

  The deviations are projected onto the screen at that point: the 2-plane
  orthogonal to the comoving u and to the ray's direction d in u's rest
  frame, k = omega (u + d). On the screen their inner products are
    G_AB = xi_A . xi_B + (u . xi_A)(u . xi_B) - (d . xi_A)(d . xi_B),
  and d_A^2 = sqrt(det G), the screen area per unit solid angle.
  """
  coords = state[:4]
  k = state[4:8]
  xi = state[8:].reshape(-1, 8)[:, :4]
  g = metric_fn(coords)
  u = comoving_four_velocity(g)
  d = k / comoving_frequency(g, k) - u

  xi_u = xi @ (g @ u)
  xi_d = xi @ (g @ d)
  gram = xi @ g @ xi.T + jnp.outer(xi_u, xi_u) - jnp.outer(xi_d, xi_d)
  det = gram[0, 0] * gram[1, 1] - gram[0, 1] * gram[1, 0]
  # The floor keeps the gradient finite at the observer, where d_A = 0.
  return jnp.sqrt(jnp.sqrt(jnp.maximum(det, 0.0) + 1e-30))


def _trace_bundle(
  metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
  z_target: float,
  direction: jnp.ndarray,
  should_log: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """
  Traces the ray from the observer at t=1.0 (today) towards a sky direction,
  with the deviation vectors to its neighbours along two orthogonal sky
  angles.

  Returns (states, is_finite_state, z, d_A) at the saved affine parameters.
  Non-finite states are replaced by the initial state, so z and d_A are
  finite everywhere and meaningful only where is_finite_state holds.
  """
  obs_coords = jnp.array([1.0, 0.0, 0.0, 0.0])
  g_obs = metric_fn(obs_coords)
  n = direction / jnp.sqrt(jnp.sum(jnp.square(direction)))
  e1, e2 = _sky_axes(n)

  # The initial rate of each deviation is the change in k per radian.
  k0, dk1 = jax.jvp(lambda m: make_null_direction(g_obs, m), (n,), (e1,))
  _, dk2 = jax.jvp(lambda m: make_null_direction(g_obs, m), (n,), (e2,))
  # One constant rescales the whole bundle so that dt/dl = 1 at the observer,
  # which spreads the saved points over the ray's coordinate time; L_MAX is
  # measured in it. Redshift and d_A do not depend on the affine scale.
  affine_scale = 1.0 / k0[0]
  k0, dk1, dk2 = k0 * affine_scale, dk1 * affine_scale, dk2 * affine_scale
  zeros = jnp.zeros(4)
  initial_ray = jnp.concatenate([obs_coords, k0])
  initial_state = jnp.concatenate([initial_ray, zeros, dk1, zeros, dk2])

  # Use a safety margin for redshift termination
  z_limit = jnp.maximum(2.5, z_target + 0.2)
  sol = _integrate_geodesic(
    metric_fn, initial_state, L_MAX, z_limit, should_log=should_log
  )

  # Prevent JAX backward pass from evaluating 0.0 * Inf = NaN by masking inputs
  # first.
  # diffrax types ys as optional (it is None when SaveAt requests nothing);
  # we always pass SaveAt(ts=...), so this is an invariant, not a fallback.
  assert sol.ys is not None
  is_finite_state = jnp.all(jnp.isfinite(sol.ys), axis=-1)
  safe_ys = jnp.where(is_finite_state[:, None], sol.ys, initial_state)

  z_vals = jax.vmap(lambda s: get_redshift(metric_fn, initial_ray, s) - 1.0)(
    safe_ys
  )
  d_a_vals = jax.vmap(lambda s: area_distance(metric_fn, s))(safe_ys)
  return safe_ys, is_finite_state, z_vals, d_a_vals


def get_luminosity_distance(
  metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
  z_target: float,
  direction: jnp.ndarray,
  should_log: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """
  Calculates dL(z) by integrating null geodesics backwards from observer.

  Observer is placed at t=1.0 (today) in the normalized training domain.
  `direction` is a required 3-vector giving the source's position on the
  observer's sky. It has no default: every caller must say where it is
  looking.

  The distance is the model's own luminosity distance,
  dL = (1 + z)^2 d_A (Etherington reciprocity), with the area distance d_A
  from the focusing and shear of the ray bundle.

  Returns (dL_mpc, z_reached). dL_mpc is only meaningful where
  z_reached >= z_target - callers MUST check.
  """
  _, is_finite_state, z_vals, d_a_vals = _trace_bundle(
    metric_fn, z_target, direction, should_log
  )
  dl_vals = jnp.square(1.0 + z_vals) * d_a_vals

  # z along the ray is NOT monotonic in general - a contracting or lumpy
  # region produces blueshift segments, and jnp.interp silently returns
  # garbage on unsorted xp. Take a running maximum so the interpolation
  # precondition holds.
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

  # dL need not be monotonic in a lumpy metric, so invalid tail states carry
  # the last finite distance.
  safe_dl_vals = jnp.where(is_finite_state, dl_vals, dl_vals[last_finite])

  # Diagnostics
  if os.environ.get("LUMPY_DEBUG") == "1":
    jax.lax.cond(
      should_log,
      lambda: jax.debug.print(
        "\n--- dL Diagnostics ---\n"
        "z_target: {z}\nz_reached: {zr}\n"
        "finite states: {nf}/{nt}",
        z=z_target,
        zr=z_reached,
        nf=jnp.sum(is_finite_state),
        nt=len(is_finite_state),
      ),
      lambda: None,
    )

  # Result via interpolation over the (now monotonic) reachable range.
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

  def shoot_ray(axis_idx):
    safe_ys, is_finite_state, z_vals, d_a_vals = _trace_bundle(
      metric_fn, z_target, jnp.eye(3)[axis_idx], should_log
    )

    # 3. Transverse comoving distance D_M = (1 + z) d_A
    dm_vals = (1.0 + z_vals) * d_a_vals

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
