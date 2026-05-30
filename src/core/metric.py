"""
Core metric tensor wrapper for the PINN.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp


class MetricNN(eqx.Module):
  """
  PINN that outputs the 10 components of a symmetric 4D metric tensor.
  Input: (t, x, y, z)
  Output: 10 scalars (upper triangle of g_mu_nu)
  """

  mlp: eqx.nn.MLP
  omega_m_raw: jnp.ndarray

  def __init__(self, key: jax.random.PRNGKey):
    # 4 inputs (t, x, y, z)
    # 10 outputs for the symmetric metric components
    self.mlp = eqx.nn.MLP(
      in_size=4,
      out_size=10,
      width_size=64,
      depth=4,
      activation=jnp.sin,  # SiREN-like activation as per design
      key=key,
    )

    # Initialize trainable matter density parameter directly.
    # Start exactly at 0.3 (dark matter ceiling).
    self.omega_m_raw = jnp.array([0.3])

    # Initialize the final layer to be NEAR Minkowski.
    # We add tiny random noise to avoid the 'perfect' zero-derivative
    # trap that causes NaN gradients in the solver.
    k1, k2 = jax.random.split(key)

    def seed_layer(layer, l_key):
      if isinstance(layer, eqx.nn.Linear):
        # Initialize weights with tiny numerical noise to avoid zero-derivatives
        noise_w = jax.random.normal(l_key, layer.weight.shape) * 1e-4
        layer = eqx.tree_at(lambda _layer: _layer.weight, layer, noise_w)

        # Add tiny numerical noise to biases
        noise = jax.random.normal(l_key, (10,)) * 1e-4
        layer = eqx.tree_at(lambda _layer: _layer.bias, layer, noise)
        return layer
      return layer

    last_idx = len(self.mlp.layers) - 1
    new_layers = list(self.mlp.layers)
    new_layers[last_idx] = seed_layer(new_layers[last_idx], k2)
    self.mlp = eqx.tree_at(lambda m: m.layers, self.mlp, tuple(new_layers))

  def get_cosmology_today(self) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns (kappa_rho_0, omega_m) today computed in a single pass.
    """
    coords_today = jnp.array([1.0, 0.0, 0.0, 0.0])
    g = self(coords_today)
    g_spatial = g[1:4, 1:4]
    g_spatial_inv = jnp.linalg.inv(g_spatial)

    dg_dt = jax.jacfwd(lambda c: self(c))(coords_today)[:, :, 0]
    dg_spatial_dt = dg_dt[1:4, 1:4]

    h_tensor = 0.5 * jnp.matmul(g_spatial_inv, dg_spatial_dt)
    u0 = jnp.sqrt(jnp.abs(g[0, 0]) + 1e-9)
    h_tensor_phys = h_tensor / u0
    h_mean_today = jnp.trace(h_tensor_phys) / 3.0

    # Use the parameter directly as omega_m (PGD will enforce bounds)
    omega_m = self.omega_m_raw
    kappa_rho_0 = omega_m * 3.0 * (h_mean_today**2)

    return kappa_rho_0, omega_m

  def __call__(self, coords: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the metric components at given coordinates.
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


def get_metric_fn(model: MetricNN) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Returns a function g(coords) -> (4,4) array."""
  return lambda coords: model(coords)
