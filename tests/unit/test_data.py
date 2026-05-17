"""
Unit tests for data loading and coordinate normalization.
"""

import jax.numpy as jnp

from src.training.data import normalize_coordinates


def test_normalize_coordinates_boundaries():
  """Verifies that redshift is mapped correctly to [-1, 1]."""
  z_max = 2.5

  # z=0 should map to t=1 (the "present")
  assert jnp.isclose(normalize_coordinates(0.0, z_max), 1.0)

  # z=z_max should map to t=-1 (the "past boundary")
  assert jnp.isclose(normalize_coordinates(z_max, z_max), -1.0)

  # z=z_max/2 should map to t=0
  assert jnp.isclose(normalize_coordinates(z_max / 2, z_max), 0.0)


def test_normalize_coordinates_array():
  """Verifies normalization works on JAX arrays."""
  z = jnp.array([0.0, 1.25, 2.5])
  t = normalize_coordinates(z, z_max=2.5)
  expected = jnp.array([1.0, 0.0, -1.0])
  assert jnp.allclose(t, expected)
