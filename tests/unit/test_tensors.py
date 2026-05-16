"""
Unit tests for the tensor calculus engine.
Verifies the Ricci tensor implementation against analytic solutions.
"""

import jax
import jax.numpy as jnp

from src.core.tensors import get_ricci_tensor

# Enable float64 for higher precision tensor math
jax.config.update("jax_enable_x64", True)


def test_ricci_minkowski():
  """Minkowski metric should have zero Ricci tensor."""

  def minkowski(coords):
    return jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))

  coords = jnp.array([1.0, 2.0, 3.0, 4.0])
  ricci = get_ricci_tensor(minkowski, coords)

  assert jnp.allclose(
    ricci, 0.0, atol=1e-5
  ), f"Ricci tensor for Minkowski should be zero, got {ricci}"


def test_ricci_schwarzschild():
  """
  Schwarzschild metric is a vacuum solution; its Ricci tensor must be zero.
  Using spherical coordinates (t, r, theta, phi).
  """

  def schwarzschild(coords):
    t, r, theta, phi = coords
    m = 1.0
    f = 1.0 - 2.0 * m / r

    g = jnp.zeros((4, 4))
    g = g.at[0, 0].set(-f)
    g = g.at[1, 1].set(1.0 / f)
    g = g.at[2, 2].set(r**2)
    g = g.at[3, 3].set(r**2 * jnp.sin(theta) ** 2)
    return g

  # Test at a point outside the horizon (r > 2M)
  # coords: t=0, r=5, theta=pi/2, phi=0
  coords = jnp.array([0.0, 5.0, jnp.pi / 2.0, 0.0])
  ricci = get_ricci_tensor(schwarzschild, coords)

  # Success Criterion: All components of Ricci must be zero to within 1e-5
  assert jnp.allclose(
    ricci, 0.0, atol=1e-5
  ), f"Ricci tensor for Schwarzschild should be zero, got {ricci}"
