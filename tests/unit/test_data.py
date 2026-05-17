"""
Unit tests for data loading and coordinate normalization.
"""

import jax.numpy as jnp

from src.training.data import normalize_coordinates


def test_normalize_coordinates_boundaries():
  """Verifies that redshift is mapped correctly using t = 1 - z."""
  # z=0 should map to t=1 (the "present")
  assert jnp.isclose(normalize_coordinates(0.0), 1.0)

  # z=1.0 should map to t=0
  assert jnp.isclose(normalize_coordinates(1.0), 0.0)

  # z=2.0 should map to t=-1.0
  assert jnp.isclose(normalize_coordinates(2.0), -1.0)


def test_normalize_coordinates_array():
  """Verifies normalization works on JAX arrays."""
  z = jnp.array([0.0, 1.0, 2.0])
  t = normalize_coordinates(z)
  expected = jnp.array([1.0, 0.0, -1.0])
  assert jnp.allclose(t, expected)
