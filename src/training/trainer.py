"""
Training loop for the PINN.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from src.training.loss import get_efe_loss


def train_control_model(
  model, num_steps=1000, learning_rate=1e-4, lam=0.7, key=None
):
  """
  Executes the training loop for the FLRW control baseline.
  Minimizes EFE residuals at random collocation points.
  """
  optimizer = optax.adam(learning_rate)
  opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

  @eqx.filter_jit
  def step(model, opt_state, coords):
    def loss_fn(model):
      # Vectorize EFE loss over batch of coordinates
      v_efe_loss = jax.vmap(lambda c: get_efe_loss(model, c, lam=lam))
      return jnp.mean(v_efe_loss(coords))

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

  # Training Loop
  for i in range(num_steps):
    key, subkey = jax.random.split(key)
    # Sample 128 random points in 4D spacetime (t, x, y, z)
    # Range: [-1, 1] for normalization
    coords = jax.random.uniform(subkey, (128, 4), minval=-1.0, maxval=1.0)

    model, opt_state, loss = step(model, opt_state, coords)

    if i % 100 == 0:
      print(f"Step {i}, Loss: {loss:.6e}")

  return model
