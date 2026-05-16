"""
Training loop for the PINN.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from src.training.loss import get_data_loss, get_efe_loss


def train_control_model(
  model,
  data,  # (redshifts, target_mu)
  num_steps=1000,
  learning_rate=1e-4,
  lam=0.7,
  key=None,
):
  """
  Executes the training loop for the FLRW control baseline.
  Minimizes combined EFE and Data residuals.
  """
  optimizer = optax.adam(learning_rate)
  opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

  redshifts, target_mu = data

  @eqx.filter_jit
  def step(model, opt_state, coords, redshifts, target_mu):
    def loss_fn(model):
      # 1. Physics Loss
      v_efe_loss = jax.vmap(lambda c: get_efe_loss(model, c, lam=lam))
      l_phys = jnp.mean(v_efe_loss(coords))

      # 2. Data Loss
      l_data = get_data_loss(model, redshifts, target_mu)

      return l_phys + 10.0 * l_data  # Weight data loss

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

  # Training Loop
  for i in range(num_steps):
    key, subkey = jax.random.split(key)
    coords = jax.random.uniform(subkey, (128, 4), minval=-1.0, maxval=1.0)

    model, opt_state, loss = step(
      model, opt_state, coords, redshifts, target_mu
    )

    if i % 100 == 0:
      print(f"Step {i}, Loss: {loss:.6e}")

  return model
