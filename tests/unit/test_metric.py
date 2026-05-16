"""
Unit tests for the MetricNN architecture.
"""

import jax
import jax.numpy as jnp

from src.core.metric import MetricNN


def test_metric_signature():
  """
  Verify that the metric always enforces a Lorentzian signature (-, +, +, +).
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
  Verify the output shape is (4, 4).
  """
  key = jax.random.PRNGKey(0)
  model = MetricNN(key)
  coords = jnp.array([1.0, 2.0, 3.0, 4.0])
  g = model(coords)

  assert g.shape == (4, 4)
