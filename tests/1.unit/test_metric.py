"""
Unit tests for the MetricNN architecture.
"""

import jax
import jax.numpy as jnp

from src.core.metric import MetricNN


def test_metric_signature():
  """
  Verification of the Lorentzian Signature Enforcement.

  Physical Principle: For a spacetime to be physical, the metric must have
  a Lorentzian signature (-, +, +, +). This ensures that we have one
  time dimension and three spatial dimensions, preserving the causal
  structure of General Relativity.

  Verification Ritual:
  1. Initialize a MetricNN with a random seed.
  2. Evaluate the metric at the origin (0, 0, 0, 0).

  Expected Outcome:
  - The g00 component (time) must be negative.
  - The g11, g22, and g33 components (space) must be positive.
  - The tensor must be symmetric (g_uv = g_vu).
  """
  key = jax.random.PRNGKey(42)
  model = MetricNN(key)

  # Test at origin
  coords = jnp.zeros(4)
  g = model(coords)

  # Check diagonals
  assert g[0, 0] < 0, f"g00 should be negative, got {g[0,0]}"
  assert g[1, 1] > 0, f"g11 should be positive, got {g[1,1]}"
  assert g[2, 2] > 0, f"g22 should be positive, got {g[2,2]}"
  assert g[3, 3] > 0, f"g33 should be positive, got {g[3,3]}"

  # Check symmetry
  assert jnp.allclose(g, g.T), "Metric tensor should be symmetric"


def test_metric_output_shape():
  """
  Verification of Metric Tensor Dimensionality.

  Verification Ritual:
  1. evaluates the MetricNN at a non-zero coordinate point.

  Expected Outcome:
  - The output must be a 4x4 matrix, representing the full 4D metric tensor.
  """
  key = jax.random.PRNGKey(0)
  model = MetricNN(key)
  coords = jnp.array([1.0, 2.0, 3.0, 4.0])
  g = model(coords)

  assert g.shape == (4, 4)
