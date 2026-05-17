"""
Unit test for gradient integrity and numerical finiteness.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from src.core.metric import MetricNN
from src.physics.geodesics import get_luminosity_distance


def test_distance_gradient_finiteness():
  """Verifies that backpropagation through the geodesic pipeline is finite."""
  key = jax.random.PRNGKey(42)
  model = MetricNN(key)
  z_target = 2.26

  def loss_fn(m):
    dl = get_luminosity_distance(m, z_target)
    return dl

  # Compute value and gradient
  val, grads = eqx.filter_value_and_grad(loss_fn)(model)

  # Check forward pass
  assert jnp.isfinite(val)

  # Check gradients
  flat_grads, _ = jax.flatten_util.ravel_pytree(grads)
  is_finite = jnp.all(jnp.isfinite(flat_grads))

  if not is_finite:
    # If failure, print indices of non-finite gradients for debugging
    print("\n--- Non-finite Gradients Detected ---")
    non_finite_indices = jnp.where(~jnp.isfinite(flat_grads))[0]
    print(f"Non-finite indices: {non_finite_indices}")
    print(f"Sample non-finite values: {flat_grads[non_finite_indices][:10]}")

  assert is_finite, "Gradients are non-finite (NaN or Inf)"
