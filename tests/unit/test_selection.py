import jax.numpy as jnp
import pytest

from src.training.loss import EfeDiagnostics
from src.training.selection import (
  CMB_ACCURACY,
  SCORE_TERMS,
  SIGMA_EFE,
  SIGMA_WEC,
  SIGMA_Z,
  TERM_EFE,
  TERM_EXPAND,
  TERM_HOMOGENEITY,
  TERM_SHEAR,
  TERM_SN,
  TERM_SN_COVER,
  TERM_WEC,
  cmb_point_ratios,
  efe_point_ratios,
  selection_score,
  selection_terms,
)


def _flrw(h):
  def metric(coords):
    a_sq = jnp.exp(2.0 * h * coords[0])
    return jnp.diag(jnp.array([-1.0, a_sq, a_sq, a_sq]))

  return metric


def test_expanding_flrw_meets_every_cmb_condition():
  shear, contraction, gradient = cmb_point_ratios(
    _flrw(0.3), jnp.array([-3.995, 0.2, -0.4, 0.6])
  )

  assert jnp.allclose(shear, 0.0, atol=1e-8)
  assert jnp.allclose(contraction, 0.0, atol=1e-8)
  assert jnp.allclose(gradient, 0.0, atol=1e-8)


def test_contracting_flrw_contracts_on_every_axis():
  # Every directional rate is contraction, so the fraction is 1.
  shear, contraction, _ = cmb_point_ratios(
    _flrw(-0.3), jnp.array([-3.995, 0.0, 0.0, 0.0])
  )

  assert jnp.allclose(shear, 0.0, atol=1e-8)
  assert jnp.allclose(contraction, 1.0, atol=1e-6)


def test_kasner_shear_and_contraction_fractions():
  # Kasner p = (-1/3, 2/3, 2/3) at t = 2: H_i = (-1/6, 1/3, 1/3), H = 1/6,
  # sigma^2 = 1/12. Shear: (1/12) / (1/36 + 1/12) = 3/4. Contraction:
  # (1/36) / (1/36 + 1/9 + 1/9) = 1/9.
  p = jnp.array([-1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])

  def metric(coords):
    return jnp.diag(jnp.concatenate([jnp.array([-1.0]), coords[0] ** (2 * p)]))

  shear, contraction, _ = cmb_point_ratios(metric, jnp.array([2.0, 0, 0, 0]))

  assert jnp.allclose(shear, 0.75, atol=1e-5)
  assert jnp.allclose(contraction, 1.0 / 9.0, atol=1e-5)


def test_spatial_gradient_is_the_squared_metric_derivative():
  eps = 0.01

  def metric(coords):
    return jnp.diag(jnp.array([-1.0, 1.0 + eps * coords[1], 1.0, 1.0]))

  _, _, gradient = cmb_point_ratios(metric, jnp.array([0.0, 0.3, 0.0, 0.0]))

  assert jnp.allclose(gradient, eps**2, rtol=1e-4)


def test_minkowski_scores_zero_on_the_cmb_conditions():
  def metric(coords):
    return jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))

  ratios = cmb_point_ratios(metric, jnp.array([-3.995, 0.1, 0.2, 0.3]))

  for r in ratios:
    assert jnp.allclose(r, 0.0, atol=1e-20)


def test_efe_point_ratios_are_relative_to_the_matter_term():
  efe, wec = efe_point_ratios(
    EfeDiagnostics(
      residual_sq=jnp.array(4e-4),
      kappa_rho=jnp.array(0.1),
      ricci_scalar=jnp.array(-0.002),
    )
  )

  assert jnp.allclose(efe, 4e-4 / 0.1**2)
  assert jnp.allclose(wec, (0.002 / 0.1) ** 2)


def test_efe_point_ratios_vacuum_scores_zero():
  efe, wec = efe_point_ratios(
    EfeDiagnostics(
      residual_sq=jnp.array(0.0),
      kappa_rho=jnp.array(0.0),
      ricci_scalar=jnp.array(0.0),
    )
  )

  assert efe == 0.0
  assert wec == 0.0


def test_selection_terms_are_in_accuracy_units():
  terms = selection_terms(
    l_sn=0.9,
    cover_sq=SIGMA_Z**2,
    shear_fraction=CMB_ACCURACY**2,
    contraction_fraction=4 * CMB_ACCURACY**2,
    gradient_sq=9 * CMB_ACCURACY**2,
    efe_rel_sq=SIGMA_EFE**2,
    wec_rel_sq=0.25 * SIGMA_WEC**2,
  )

  assert set(terms) == set(SCORE_TERMS)
  assert terms[TERM_SN] == 0.9
  assert terms[TERM_SN_COVER] == pytest.approx(1.0)
  assert terms[TERM_SHEAR] == pytest.approx(1.0)
  assert terms[TERM_EXPAND] == pytest.approx(4.0)
  assert terms[TERM_HOMOGENEITY] == pytest.approx(9.0)
  assert terms[TERM_EFE] == pytest.approx(1.0)
  assert terms[TERM_WEC] == pytest.approx(0.25)


def test_selection_score_is_the_sum_of_terms():
  terms = {name: float(i) for i, name in enumerate(SCORE_TERMS)}

  assert selection_score(terms) == pytest.approx(sum(range(len(SCORE_TERMS))))


def test_lower_score_wins_whichever_terms_differ():
  # A better fit with an unbalanced field equation loses to a slightly worse
  # fit that balances it.
  base = dict(
    cover_sq=0.0,
    shear_fraction=0.0,
    contraction_fraction=0.0,
    gradient_sq=0.0,
    wec_rel_sq=0.0,
  )
  good_fit = selection_terms(l_sn=0.9, efe_rel_sq=SIGMA_EFE**2 * 10, **base)
  good_physics = selection_terms(l_sn=1.1, efe_rel_sq=SIGMA_EFE**2, **base)

  assert selection_score(good_physics) < selection_score(good_fit)
