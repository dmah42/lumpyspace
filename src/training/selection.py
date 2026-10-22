"""
Accuracy-normalised selection score for choosing the best checkpoint.

Each term is a mean squared deviation in units of how precisely that quantity
is known, so a term of order 1 means "as good as the accuracy allows", the way
chi^2/N is for the supernovae. The score is their sum. It decides which step
is kept; the training loss does not see it. See Task 4.13 in
docs/technical_design.md.
"""

from collections.abc import Callable

import jax.numpy as jnp

from src.training.loss import CMB_ACCURACY, EfeDiagnostics, cmb_fractions

# Redshift scatter from peculiar velocities (~300 km/s).
SIGMA_Z = 1e-3
# The field equations and the weak energy condition must hold to 1% of the
# local matter term: an order of magnitude below the ~10% density effects that
# inhomogeneous alternatives to Lambda predict, so an effect of that size
# cannot be confused with an unbalanced equation.
SIGMA_EFE = 1e-2
SIGMA_WEC = 1e-2

# Small-denominator floor, matching get_efe_loss. Vacuum has kappa rho = 0,
# and with this floor it scores zero on the field-equation and energy
# condition terms.
_DENOM_FLOOR = 1e-12

METRIC_SCORE = "score"

# Per-step raw means returned by the training step.
RAW_COVER_SQ = "cover_sq"
RAW_SHEAR_FRACTION = "shear_fraction"
RAW_CONTRACTION_FRACTION = "contraction_fraction"
RAW_GRADIENT_SQ = "gradient_sq"
RAW_EFE_REL_SQ = "efe_rel_sq"
RAW_WEC_REL_SQ = "wec_rel_sq"

# Score terms, in accuracy units.
TERM_SN = "sn"
TERM_SN_COVER = "sn_cover"
TERM_SHEAR = "shear"
TERM_EXPAND = "expand"
TERM_HOMOGENEITY = "homogeneity"
TERM_EFE = "efe"
TERM_WEC = "wec"
SCORE_TERMS = (
  TERM_SN,
  TERM_SN_COVER,
  TERM_SHEAR,
  TERM_EXPAND,
  TERM_HOMOGENEITY,
  TERM_EFE,
  TERM_WEC,
)


def cmb_point_ratios(
  metric_fn: Callable[[jnp.ndarray], jnp.ndarray], coords: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """
  CMB-boundary quantities at one point: the shear and contraction fractions
  the CMB loss constrains (see cmb_fractions), which approximate (sigma/H)^2
  and the squared contraction-to-expansion ratio while those are small, and
  the squared perturbation amplitude.
  """
  shear, contraction, gradient = cmb_fractions(metric_fn, coords)
  return shear, contraction, gradient**2


def efe_point_ratios(
  diagnostics: EfeDiagnostics,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """
  Field-equation residual and weak-energy-condition violation relative to the
  local matter term: (residual / kappa rho)^2 and (max(0, -R) / kappa rho)^2.
  """
  kappa_rho = jnp.maximum(diagnostics.kappa_rho, _DENOM_FLOOR)
  efe = diagnostics.residual_sq / kappa_rho**2
  wec = (jnp.maximum(0.0, -diagnostics.ricci_scalar) / kappa_rho) ** 2
  return efe, wec


def selection_terms(
  l_sn: float,
  cover_sq: float,
  shear_fraction: float,
  contraction_fraction: float,
  gradient_sq: float,
  efe_rel_sq: float,
  wec_rel_sq: float,
) -> dict[str, float]:
  """Converts the per-step raw means into accuracy units."""
  cmb_var = CMB_ACCURACY**2
  return {
    TERM_SN: l_sn,
    TERM_SN_COVER: cover_sq / SIGMA_Z**2,
    TERM_SHEAR: shear_fraction / cmb_var,
    TERM_EXPAND: contraction_fraction / cmb_var,
    TERM_HOMOGENEITY: gradient_sq / cmb_var,
    TERM_EFE: efe_rel_sq / SIGMA_EFE**2,
    TERM_WEC: wec_rel_sq / SIGMA_WEC**2,
  }


def selection_score(terms: dict[str, float]) -> float:
  """The selection score: the sum of the terms."""
  return float(sum(terms.values()))
