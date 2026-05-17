"""
Integration tests for Pantheon+ data ingestion.
"""

import os

import jax.numpy as jnp
import pytest

from src.training.data import load_pantheon_plus


@pytest.mark.skipif(
  not os.path.exists("data/pantheon_plus.dat"),
  reason="Pantheon+ data not found. Run scripts/download_pantheon.py.",
)
def test_load_pantheon_plus_integrity():
  """Verifies the real dataset loads with expected dimensions and values."""
  z, mu, err = load_pantheon_plus("data/pantheon_plus.dat")

  # Check dimensions (1701 supernovae in Pantheon+)
  assert len(z) == 1701
  assert len(mu) == 1701
  assert len(err) == 1701

  # Check for NaNs
  assert not jnp.any(jnp.isnan(z))
  assert not jnp.any(jnp.isnan(mu))
  assert not jnp.any(jnp.isnan(err))

  # Check redshift range
  assert jnp.all(z > 0)
  assert z.max() > 2.0

  # Check distance modulus range (typical values 30-45)
  assert mu.min() > 20.0
  assert mu.max() < 50.0
