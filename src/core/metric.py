"""
Core metric tensor wrapper for the PINN.
"""

import math
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from src.core.kinematics import comoving_kinematics, comoving_spatial_metric

# The observer today. The spatial scale is normalised so that det h = 1 here
# (Task 4.15).
OBSERVER_COORDS = jnp.array([1.0, 0.0, 0.0, 0.0])

# Absolute matter densities today, in kg/m^3.
# Baryons: Big Bang nucleosynthesis fixes the baryon-to-photon ratio from
# deuterium abundances, and the measured CMB temperature fixes the photon
# number density, so the baryon density follows directly.
RHO_BARYON_KG_M3 = 4.17e-28
# Total matter, baryons plus dark matter.
RHO_MATTER_KG_M3 = 2.63e-27

# kappa rho = 8 pi G rho / c^2, in code units of 1 / L_UNIT^2 with
# L_UNIT = 1 Gpc (see geodesics.py).
_G_SI = 6.67430e-11  # m^3 kg^-1 s^-2
_C_SI = 2.99792458e8  # m/s
_GPC_M = 3.085677581e25  # m
_KAPPA_PER_KG_M3 = 8.0 * math.pi * _G_SI / _C_SI**2 * _GPC_M**2

KAPPA_RHO_BARYON = _KAPPA_PER_KG_M3 * RHO_BARYON_KG_M3
KAPPA_RHO_MATTER = _KAPPA_PER_KG_M3 * RHO_MATTER_KG_M3

# kappa_rho_0 = KAPPA_RHO_BARYON * exp(kappa_rho_0_raw), so the raw parameter
# spans [0, KAPPA_RHO_RAW_MAX] between the baryon floor and total matter.
KAPPA_RHO_RAW_MAX = math.log(KAPPA_RHO_MATTER / KAPPA_RHO_BARYON)


class MetricNN(eqx.Module):
  """
  PINN that outputs the 10 components of a symmetric 4D metric tensor.
  Input: (t, x, y, z)
  Output: 10 scalars (upper triangle of g_mu_nu)
  """

  OMEGA_0 = 10.0
  EPSILON = 1e-4

  mlp: eqx.nn.MLP
  spatial_weight_net: eqx.nn.MLP
  kappa_rho_0_raw: jnp.ndarray

  def __init__(self, key: jax.Array):
    # 4 inputs (t, x, y, z)
    # 10 outputs for the symmetric metric components
    self.mlp = eqx.nn.MLP(
      in_size=4,
      out_size=10,
      width_size=64,
      depth=4,
      activation=jnp.sin,  # SiREN-like activation
      key=key,
    )

    # Trainable absolute matter density, in log space above the baryon floor.
    # Starts at the floor: baryons only, rising if the field equations need
    # more matter.
    self.kappa_rho_0_raw = jnp.array([0.0])

    # Initialize the final layer to be NEAR Minkowski.
    # We add tiny random noise to avoid the 'perfect' zero-derivative
    # trap that causes NaN gradients in the solver.
    k1, k2, k_spatial = jax.random.split(key, 3)

    # 4 inputs (t, x, y, z), 1 output (w_4d)
    self.spatial_weight_net = eqx.nn.MLP(
      in_size=4,
      out_size=1,
      width_size=16,
      depth=2,
      activation=jnp.tanh,
      key=k_spatial,
    )

    def seed_last_layer(layer, l_key):
      if isinstance(layer, eqx.nn.Linear):
        # Initialize weights with tiny numerical noise to avoid zero-derivatives
        noise_w = jax.random.normal(l_key, layer.weight.shape) * self.EPSILON
        layer = eqx.tree_at(lambda _layer: _layer.weight, layer, noise_w)

        # Add tiny numerical noise to biases
        noise = jax.random.normal(l_key, (10,)) * self.EPSILON
        layer = eqx.tree_at(lambda _layer: _layer.bias, layer, noise)
        return layer
      return layer

    def apply_siren_omega_0(layer, omega_0: float):
      """
      Applies the SIREN omega_0 frequency multiplier to spatial dimensions.
      """
      if isinstance(layer, eqx.nn.Linear):
        # layer.weight is (out_features, in_features) where in_features is 4
        # We ONLY want high frequency in space (x, y, z), not time (t).
        # Time is column 0. Space is columns 1, 2, 3.
        w_scaled = layer.weight.at[:, 1:].set(layer.weight[:, 1:] * omega_0)

        # Scale biases to ensure the high-frequency phase shifts correctly
        # eqx.nn.Linear allows use_bias=False; the SIREN phase shift is not
        # meaningful without one, so fail loudly rather than silently skip it.
        assert layer.bias is not None, "SIREN first layer requires a bias"
        b_scaled = layer.bias * omega_0

        layer = eqx.tree_at(lambda _layer: _layer.weight, layer, w_scaled)
        layer = eqx.tree_at(lambda _layer: _layer.bias, layer, b_scaled)
        return layer
      return layer

    last_idx = len(self.mlp.layers) - 1
    new_layers = list(self.mlp.layers)

    # Apply SIREN high-frequency initialization to the first layer
    new_layers[0] = apply_siren_omega_0(new_layers[0], omega_0=self.OMEGA_0)

    # Apply near-zero initialization to final layer to start near-Minkowski
    new_layers[last_idx] = seed_last_layer(new_layers[last_idx], k2)

    self.mlp = eqx.tree_at(lambda m: m.layers, self.mlp, tuple(new_layers))

  def get_spatial_weight(self, coords: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluates the learned spatial weight at the given 4D coordinates.
    """
    return self.spatial_weight_net(coords)

  def get_cosmology_today(self) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns (kappa_rho_0, omega_m) today computed in a single pass.
    det h = 1 at the observer, so kappa_rho_0 is the observer's density today.
    """
    g = self(OBSERVER_COORDS)
    dg = jax.jacfwd(lambda c: self(c))(OBSERVER_COORDS)
    h_mean_today = comoving_kinematics(g, dg).h_mean

    kappa_rho_0 = KAPPA_RHO_BARYON * jnp.exp(self.kappa_rho_0_raw)
    # Omega_m is reported, not trained: the matter density as a fraction of
    # the critical density 3 H^2 / kappa at the model's own expansion rate.
    omega_m = kappa_rho_0 / jnp.maximum(3.0 * h_mean_today**2, 1e-12)

    return kappa_rho_0, omega_m

  def __call__(self, coords: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the metric components at given coordinates, with the spatial
    scale normalised to det h = 1 at the observer today. The normalisation
    depends on the network alone, so it is computed once under vmap and
    jacfwd over coordinates.
    """
    return normalise_observer_scale(
      self._network_metric(coords), self._network_metric(OBSERVER_COORDS)
    )

  def _network_metric(self, coords: jnp.ndarray) -> jnp.ndarray:
    """
    The network's metric before scale normalisation.
    Ensures Lorentzian signature (-, +, +, +) via exponentiation.
    """
    # Clip coordinates to prevent singularities in random metrics during early
    # training if the ray-tracer wanders outside safe domain.
    safe_coords = jnp.clip(coords, -6.0, 6.0)
    out = self.mlp(safe_coords)

    # Enforce Lorentzian signature as per technical design 4.1
    # g00 must be negative, g_ii must be positive
    g00 = -jnp.exp(out[0])
    g11 = jnp.exp(out[1])
    g22 = jnp.exp(out[2])
    g33 = jnp.exp(out[3])

    # Off-diagonal components
    g01 = out[4]
    g02 = out[5]
    g03 = out[6]
    g12 = out[7]
    g13 = out[8]
    g23 = out[9]

    # Construct the symmetric matrix
    metric = jnp.array(
      [
        [g00, g01, g02, g03],
        [g01, g11, g12, g13],
        [g02, g12, g22, g23],
        [g03, g13, g23, g33],
      ]
    )

    return metric


def normalise_observer_scale(g: jnp.ndarray, g_obs: jnp.ndarray) -> jnp.ndarray:
  """
  Fixes the overall spatial scale by the convention a_0 = 1: det h = 1 at the
  observer, where g_obs is the metric. Returns S g S with
  S = diag(1, s, s, s) and s = det(h_obs)^(-1/6), which takes h to s^2 h.
  One coordinate unit is then one unit of proper length at the observer, and
  a uniform rescaling of the spatial lengths leaves the result unchanged.
  """
  s = jnp.linalg.det(comoving_spatial_metric(g_obs)) ** (-1.0 / 6.0)
  scale = jnp.array([1.0, s, s, s])
  return g * jnp.outer(scale, scale)


def get_metric_fn(model: MetricNN) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Returns a function g(coords) -> (4,4) array."""
  return lambda coords: model(coords)


def clip_matter_density(model: MetricNN) -> MetricNN:
  """
  Projected gradient descent step for the matter density: clips it to
  [KAPPA_RHO_BARYON, KAPPA_RHO_MATTER], at least the baryons and at most
  baryons plus dark matter.
  """
  return eqx.tree_at(
    lambda m: m.kappa_rho_0_raw,
    model,
    jnp.clip(model.kappa_rho_0_raw, 0.0, KAPPA_RHO_RAW_MAX),
  )
