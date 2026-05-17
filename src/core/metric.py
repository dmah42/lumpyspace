"""
Core metric tensor wrapper for the PINN.
"""

from typing import Callable

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

    # Initialize the final layer to be NEAR Minkowski.
    # We add tiny random noise to avoid the 'perfect' zero-derivative
    # trap that causes NaN gradients in the solver.
    k1, k2 = jax.random.split(key)

    def seed_layer(layer, l_key):
      if isinstance(layer, eqx.nn.Linear):
        # Zero the weights
        layer = eqx.tree_at(
          lambda _layer: _layer.weight, layer, jnp.zeros_like(layer.weight)
        )
        # Add tiny numerical noise to biases
        noise = jax.random.normal(l_key, (10,)) * 1e-5
        layer = eqx.tree_at(lambda _layer: _layer.bias, layer, noise)
        return layer
      return layer

    last_idx = len(self.mlp.layers) - 1
    new_layers = list(self.mlp.layers)
    new_layers[last_idx] = seed_layer(new_layers[last_idx], k2)
    self.mlp = eqx.tree_at(lambda m: m.layers, self.mlp, tuple(new_layers))

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
