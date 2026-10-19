"""
Unit tests for the MetricNN architecture.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from src.core.kinematics import comoving_kinematics
from src.core.metric import (
  KAPPA_RHO_BARYON,
  KAPPA_RHO_MATTER,
  MetricNN,
  clip_matter_density,
)


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


def test_matter_density_bounds_match_the_physical_densities():
  # kappa rho = 8 pi G rho / c^2 in units of Gpc^-2: 4.17e-28 kg/m^3 of
  # baryons and 2.63e-27 kg/m^3 of total matter.
  assert KAPPA_RHO_BARYON == pytest.approx(0.00741, rel=1e-3)
  assert KAPPA_RHO_MATTER == pytest.approx(0.0467, rel=1e-3)


def test_matter_density_starts_at_the_baryon_floor():
  model = MetricNN(jax.random.PRNGKey(0))

  kappa_rho_0, _ = model.get_cosmology_today()

  assert kappa_rho_0[0] == pytest.approx(KAPPA_RHO_BARYON, rel=1e-6)


def test_omega_m_is_the_density_over_the_critical_density():
  model = MetricNN(jax.random.PRNGKey(0))
  coords_today = jnp.array([1.0, 0.0, 0.0, 0.0])
  h_today = comoving_kinematics(
    model(coords_today), jax.jacfwd(model)(coords_today)
  ).h_mean

  kappa_rho_0, omega_m = model.get_cosmology_today()

  assert omega_m[0] == pytest.approx(
    float(kappa_rho_0[0] / (3.0 * h_today**2)), rel=1e-4
  )


@pytest.mark.parametrize(
  "raw, expected",
  [(-1.0, KAPPA_RHO_BARYON), (0.5, None), (10.0, KAPPA_RHO_MATTER)],
)
def test_clip_matter_density_keeps_baryons_to_total_matter(raw, expected):
  model = MetricNN(jax.random.PRNGKey(0))
  model = eqx.tree_at(lambda m: m.kappa_rho_0_raw, model, jnp.array([raw]))

  kappa_rho_0, _ = clip_matter_density(model).get_cosmology_today()

  if expected is None:
    expected = KAPPA_RHO_BARYON * float(jnp.exp(raw))
  assert kappa_rho_0[0] == pytest.approx(expected, rel=1e-5)
