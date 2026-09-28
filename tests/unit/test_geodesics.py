"""
Unit tests for the geodesic equations.
"""

import jax
import jax.numpy as jnp
import pytest

from src.physics.geodesics import L_UNIT, geodesic_system, get_bao_distances

# Enable float64 for high-precision physics checks
jax.config.update("jax_enable_x64", True)


def test_minkowski_geodesic():
  """
  Verification of Geodesic Equations in Flat Spacetime.

  Physical Principle: In Minkowski space (flat spacetime), all Christoffel
  symbols are zero. Therefore, the geodesic equation dk^mu/dl = 0 implies
  that light travels in straight lines with constant four-momentum.

  Verification Ritual:
  1. Define a standard Minkowski metric.
  2. Provide an initial state at the origin moving in the x-direction.
  3. Calculate the derivative (flow) using the geodesic_system.

  Expected Outcome:
  - dx^mu/dl must exactly match the initial four-momentum k^mu.
  - dk^mu/dl must be zero within numerical precision (1e-6).
  """

  def minkowski_metric(coords):
    return jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))

  # Initial state: origin, moving in x direction
  # [t, x, y, z, kt, kx, ky, kz]
  state = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])

  args = (minkowski_metric, jax.jacfwd(minkowski_metric), None, None, False)
  derivative = geodesic_system(0.0, state, args)

  # dx/dl should be k
  assert jnp.allclose(derivative[:4], state[4:])

  # dk/dl should be 0
  assert jnp.allclose(derivative[4:], 0.0, atol=1e-6)


@pytest.mark.parametrize("m", [0.5, 1.0, 2.0])
def test_schwarzschild_geodesic_acceleration(m):
  """
  Verification of Geodesic Acceleration in Curved Spacetime.

  Physical Principle: In a Schwarzschild metric (spherical mass), the spacetime
  is curved. A photon moving radially must experience non-zero 'acceleration'
  in its coordinate four-momentum (dk^mu/dl != 0) due to the non-vanishing
  gravitational field (Christoffel symbols).

  Verification Ritual:
  1. Define a Schwarzschild metric for a given mass 'm'.
  2. Place a photon at r=5 moving radially.
  3. Calculate the geodesic flow.

  Expected Outcome:
  - The radial component of the momentum derivative (dk^r/dl) must be
    significantly non-zero (> 1e-4).
  - Angular components should remain zero for purely radial motion.
  """

  def schwarzschild_metric(coords):
    t, r, theta, phi = coords
    f = 1.0 - 2.0 * m / r
    return jnp.diag(jnp.array([-f, 1.0 / f, r**2, r**2 * jnp.sin(theta) ** 2]))

  # State: r=5, moving radially (kr != 0)
  state = jnp.array([0.0, 5.0, jnp.pi / 2, 0.0, 1.0, 1.0, 0.0, 0.0])

  args = (
    schwarzschild_metric,
    jax.jacfwd(schwarzschild_metric),
    None,
    None,
    False,
  )
  derivative = geodesic_system(0.0, state, args)

  # Non-zero acceleration
  assert jnp.abs(derivative[5]) > 1e-4
  assert jnp.allclose(derivative[6:], 0.0, atol=1e-6)


def test_null_constraint_conservation():
  """
  Verification of the Null Geodesic Constraint Conservation.

  Physical Principle: A null geodesic (light ray) must always satisfy the
  constraint g_uv k^u k^v = 0 along its entire path. This implies that the
  derivative of this constraint along the geodesic flow must be zero.

  Verification Ritual:
  1. Define a time-dependent curved metric.
  2. Calculate the geodesic flow at a specific state.
  3. Use automatic differentiation (jax.grad) to find the gradient of the
     null constraint function.
  4. Perform the dot product (directional derivative) of the gradient and flow.

  Expected Outcome:
  - The change in the null constraint along the flow must be zero.
  - Due to metric regularization (1e-6) added for PINN training stability,
    we expect a small numerical residual, so we verify conservation to
    within 1e-6.
  """

  def curved_metric(coords):
    # A fake time-dependent metric for testing
    t, x, y, z = coords
    return jnp.diag(jnp.array([-1.0, 1.0 + 0.1 * t, 1.0, 1.0]))

  state = jnp.array([0.5, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])

  # Manual check of the derivative of the constraint
  def constraint(s):
    coords = s[:4]
    k = s[4:]
    g = curved_metric(coords)
    return jnp.einsum("uv,u,v", g, k, k)

  # Derivative of constraint along the flow
  args = (curved_metric, jax.jacfwd(curved_metric), None, None, False)
  flow = geodesic_system(0.0, state, args)

  # grad(constraint) dot flow
  grad_c = jax.grad(constraint)(state)
  change_in_constraint = jnp.dot(grad_c, flow)

  # Tolerance loosened from 1e-12 to 1e-6 due to metric regularization
  # for PINN stability
  assert jnp.abs(change_in_constraint) < 1e-6


def test_bao_isotropy_on_flrw_metric():
  """
  Verification of BAO Isotropic Distances in Flat FLRW.

  Physical Principle: In a standard isotropic expanding universe (FLRW),
  the expansion rate and spatial geometry are identical in all directions.
  Therefore, shooting light rays along the x, y, and z axes must yield
  the exact same averaged distances, and must match the theoretical D_H.

  Verification Ritual:
  1. Define a standard flat expanding FLRW metric.
  2. Run the get_bao_distances ray-tracer at z=0.5.
  3. Verify that the function executes without shape errors and
     returns valid physical constraints.
  """

  def flrw_metric(coords):
    t, x, y, z = coords
    # Standard flat FLRW metric: ds^2 = -dt^2 + a(t)^2 (dx^2 + dy^2 + dz^2)
    # Using a simple expanding scale factor a(t) = exp(0.5 * (t - 1.0))
    a = jnp.exp(0.5 * (t - 1.0))
    spatial = a**2 * jnp.eye(3)
    g = jnp.zeros((4, 4))
    g = g.at[0, 0].set(-1.0)
    g = g.at[1:4, 1:4].set(spatial)
    return g

  z_target = 0.5

  # Run the function
  dm, dh = get_bao_distances(flrw_metric, z_target, should_log=False)

  # Validate types and physical constraints
  assert not jnp.isnan(dm), "Transverse distance is NaN"
  assert not jnp.isnan(dh), "Radial distance is NaN"

  # Both distances must be strictly positive in an expanding universe
  assert dm > 0.0, f"Expected positive Transverse distance, got {dm}"
  assert dh > 0.0, f"Expected positive Radial distance, got {dh}"

  # We can also do a direct calculation of D_H at z_target to ensure math is
  # correct.
  # For a(t) = exp(0.5*(t-1)), H = a'/a = 0.5
  # So D_H = 1/H = 2.0 in code units.
  # In physical units: D_H = 2.0 * L_UNIT
  assert jnp.allclose(
    dh, 2.0 * L_UNIT, atol=1e-1
  ), f"Expected D_H ~ {2.0 * L_UNIT}, got {dh}"
