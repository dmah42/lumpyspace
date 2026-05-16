"""
Unit tests for the geodesic equations.
"""

import jax.numpy as jnp
import pytest

from src.physics.geodesics import geodesic_system


def test_minkowski_geodesic():
  """
  In Minkowski space, Gamma is zero, so dk/dl should be zero (straight lines).
  """

  def minkowski_metric(coords):
    return jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))

  # Initial state: origin, moving in x direction
  # [t, x, y, z, kt, kx, ky, kz]
  state = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])

  derivative = geodesic_system(0.0, state, minkowski_metric)

  # dx/dl should be k
  assert jnp.allclose(derivative[:4], state[4:])

  # dk/dl should be 0
  assert jnp.allclose(derivative[4:], 0.0, atol=1e-6)


@pytest.mark.parametrize("m", [0.5, 1.0, 2.0])
def test_schwarzschild_geodesic_acceleration(m):
  """
  In Schwarzschild space, a photon moving radially should experience
  acceleration (dk/dl != 0).
  """

  def schwarzschild_metric(coords):
    t, r, theta, phi = coords
    f = 1.0 - 2.0 * m / r
    return jnp.diag(jnp.array([-f, 1.0 / f, r**2, r**2 * jnp.sin(theta) ** 2]))

  # State: r=5, moving radially (kr != 0)
  state = jnp.array([0.0, 5.0, jnp.pi / 2, 0.0, 1.0, 1.0, 0.0, 0.0])

  derivative = geodesic_system(0.0, state, schwarzschild_metric)

  # Non-zero acceleration
  assert jnp.abs(derivative[5]) > 1e-4
  assert jnp.allclose(derivative[6:], 0.0, atol=1e-6)


def test_null_constraint_conservation():
  """
  A null geodesic must maintain g_uv k^u k^v = 0.
  We check if the derivative of the null constraint is zero:
  d/dl (g_uv k^u k^v) = 0
  """

  def curved_metric(coords):
    # A fake time-dependent metric for testing
    t, x, y, z = coords
    return jnp.diag(jnp.array([-1.0, 1.0 + 0.1 * t, 1.0, 1.0]))

  state = jnp.array([0.5, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])

  # This is a bit more complex since we need to verify if the DK_dl
  # preserves the null condition.
  # d/dl (g_uv k^u k^v) = (d_r g_uv) k^r k^u k^v + 2 g_uv k^u (dk^v/dl)
  # But dk^v/dl = -Gamma^v_ab k^a k^b
  # The geodesic equation is derived precisely to make this zero.

  # Manual check of the derivative of the constraint
  import jax

  def constraint(s):
    coords = s[:4]
    k = s[4:]
    g = curved_metric(coords)
    return jnp.einsum("uv,u,v", g, k, k)

  # Derivative of constraint along the flow
  flow = geodesic_system(0.0, state, curved_metric)

  # grad(constraint) dot flow
  grad_c = jax.grad(constraint)(state)
  change_in_constraint = jnp.dot(grad_c, flow)

  assert jnp.abs(change_in_constraint) < 1e-6
