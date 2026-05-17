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
  """
  Verification of the Ricci Tensor in Flat Spacetime.

  Physical Principle: Minkowski space is flat, meaning there is no curvature.
  The Ricci tensor, which represents a contraction of the Riemann curvature
  tensor, must be identically zero in any flat spacetime.

  Verification Ritual:
  1. Define a Minkowski metric.
  2. Calculate the Ricci tensor at a random coordinate point using the
     tensor calculus engine.

  Expected Outcome:
  - All components of the Ricci tensor must be zero within numerical
    precision (1e-5).
  """

  def minkowski(coords):
    return jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))

  coords = jnp.array([1.0, 2.0, 3.0, 4.0])
  ricci = get_ricci_tensor(minkowski, coords)

  assert jnp.allclose(
    ricci, 0.0, atol=1e-5
  ), f"Ricci tensor for Minkowski should be zero, got {ricci}"


def test_ricci_schwarzschild():
  """
  Verification of the Ricci Tensor in Schwarzschild Vacuum.

  Physical Principle: The Schwarzschild metric is a solution to the vacuum
  Einstein Field Equations (R_uv = 0). This represents the spacetime
  outside a spherical mass. Even though the spacetime is curved (non-zero
  Riemann tensor), the Ricci tensor must vanish in the vacuum.

  Verification Ritual:
  1. Define a Schwarzschild metric in spherical coordinates.
  2. Calculate the Ricci tensor at a point outside the event horizon.

  Expected Outcome:
  - All components of the Ricci tensor must be zero within numerical
    precision (1e-5). This verifies that our automated tensor algebra
    correctly handles complex coordinate dependencies and derivatives.
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
