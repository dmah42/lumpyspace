"""
Accuracy-normalised selection score for choosing the best checkpoint.

Each term is a mean squared deviation in units of how precisely that quantity
is known, so a term of order 1 means "as good as the accuracy allows", the way
chi^2/N is for the supernovae. The score is their sum. It decides which step
is kept; the training loss does not see it. See Task 4.13 in
docs/technical_design.md.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from src.core.kinematics import comoving_kinematics
from src.training.loss import EfeDiagnostics

# Redshift scatter from peculiar velocities (~300 km/s).
SIGMA_Z = 1e-3
# CMB temperature anisotropy level, which bounds sigma/H, contraction/H and
# metric perturbations at last scattering.
CMB_ACCURACY = 1e-5
# The field equations and the weak energy condition must hold to 1% of the
# local matter term: an order of magnitude below the ~10% density effects that
# inhomogeneous alternatives to Lambda predict, so an effect of that size
# cannot be confused with an unbalanced equation.
SIGMA_EFE = 1e-2
SIGMA_WEC = 1e-2

# Small-denominator floor, matching get_efe_loss. Flat static space has
# H = 0 and kappa rho = 0, and with this floor it scores zero on every
# physics term.
_DENOM_FLOOR = 1e-12

METRIC_SCORE = "score"

# Per-step raw means returned by the training step.
RAW_COVER_SQ = "cover_sq"
RAW_SHEAR_RATIO_SQ = "shear_ratio_sq"
RAW_CONTRACTION_RATIO_SQ = "contraction_ratio_sq"
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
  CMB-boundary quantities at one point, in the dimensionless forms the CMB
  constrains: (sigma/H)^2, sum_i (max(0, -H_i)/H)^2, and sum (d_k g_mu_nu)^2
  over the three spatial derivatives (a squared perturbation amplitude per
  unit code length, ~1 Gpc).
  """
  g = metric_fn(coords)
  dg = jax.jacfwd(metric_fn)(coords)
  kinematics = comoving_kinematics(g, dg)
  h_sq = jnp.maximum(kinematics.h_mean**2, _DENOM_FLOOR)
  shear = kinematics.sigma_sq / h_sq
  contraction = jnp.sum(jnp.maximum(0.0, -kinematics.h_dir) ** 2) / h_sq
  gradient = jnp.sum(jnp.square(dg[:, :, 1:4]))
  return shear, contraction, gradient


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
  shear_ratio_sq: float,
  contraction_ratio_sq: float,
  gradient_sq: float,
  efe_rel_sq: float,
  wec_rel_sq: float,
) -> dict[str, float]:
  """Converts the per-step raw means into accuracy units."""
  cmb_var = CMB_ACCURACY**2
  return {
    TERM_SN: l_sn,
    TERM_SN_COVER: cover_sq / SIGMA_Z**2,
    TERM_SHEAR: shear_ratio_sq / cmb_var,
    TERM_EXPAND: contraction_ratio_sq / cmb_var,
    TERM_HOMOGENEITY: gradient_sq / cmb_var,
    TERM_EFE: efe_rel_sq / SIGMA_EFE**2,
    TERM_WEC: wec_rel_sq / SIGMA_WEC**2,
  }


def selection_score(terms: dict[str, float]) -> float:
  """The selection score: the sum of the terms."""
  return float(sum(terms.values()))
