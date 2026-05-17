"""
Training loop for the PINN.
"""

from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from src.training.loss import get_data_loss, get_efe_loss


def train_control_model(
  model: eqx.Module,
  data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
  num_steps: int = 1000,
  learning_rate: float = 1e-4,
  lam: float = 0.7,
  key: Optional[jax.Array] = None,
) -> eqx.Module:
  """
  Executes the training loop for the FLRW control baseline.
  Minimizes combined EFE and Data residuals.
  """
  # Use gradient clipping to stabilize training
  optimizer = optax.chain(
    optax.clip_by_global_norm(1.0), optax.adam(learning_rate)
  )
  opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

  redshifts, target_mu, mu_err = data

  @eqx.filter_jit
  def step(
    model: eqx.Module,
    opt_state: optax.OptState,
    coords: jnp.ndarray,
    redshifts: jnp.ndarray,
    target_mu: jnp.ndarray,
    mu_err: jnp.ndarray,
  ) -> Tuple[eqx.Module, optax.OptState, jnp.ndarray]:
    def loss_fn(model):
      # 1. Physics Loss
      v_efe_loss = jax.vmap(lambda c: get_efe_loss(model, c, lam=lam))
      l_phys = jnp.mean(v_efe_loss(coords))

      # 2. Data Loss
      l_data = get_data_loss(model, redshifts, target_mu, mu_err)

      return l_phys + 10.0 * l_data  # Weight data loss

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

  # Training Loop
  for i in range(num_steps):
    key, subkey = jax.random.split(key)
    # Sample points in the normalized training domain [-1, 1]
    coords = jax.random.uniform(subkey, (128, 4), minval=-1.0, maxval=1.0)

    model, opt_state, loss = step(
      model, opt_state, coords, redshifts, target_mu, mu_err
    )

    if jnp.isnan(loss):
      print(f"CRITICAL: NaN loss detected at step {i}. Terminating training.")
      break

    if i % 10 == 0:  # More frequent reporting for short tests
      print(f"Step {i}, Loss: {loss:.6e}")

  return model
