"""
Training library for the PINN, including early stopping and telemetry.
"""

import csv
import os
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from src.training.loss import get_data_loss, get_efe_loss


@contextmanager
def _get_log_writer(log_path: Optional[str]):
  """Helper to handle optional CSV logging."""
  if log_path is None:
    yield None
  else:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, mode="w", newline="") as log_file:
      log_writer = csv.DictWriter(
        log_file, fieldnames=["step", "loss", "l_phys", "l_data"]
      )
      log_writer.writeheader()
      yield log_writer, log_file


def train_model(
  model: eqx.Module,
  data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
  max_steps: int = 10000,
  learning_rate: float = 1e-4,
  lam: float = 0.7,
  target_loss: float = 1e-6,
  patience: int = 500,
  log_path: Optional[str] = "logs/training_metrics.csv",
  key: Optional[jax.Array] = None,
) -> eqx.Module:
  """
  Executes the training loop for the PINN.
  Minimizes combined EFE and Data residuals with early stopping and telemetry.
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
  ) -> Tuple[eqx.Module, optax.OptState, jnp.ndarray, Dict[str, jnp.ndarray]]:
    def loss_fn(model):
      # 1. Physics Loss (EFE residuals)
      v_efe_loss = jax.vmap(lambda c: get_efe_loss(model, c, lam=lam))
      l_phys = jnp.mean(v_efe_loss(coords))

      # 2. Data Loss (Supernova chi-squared)
      l_data = get_data_loss(model, redshifts, target_mu, mu_err)

      total_loss = l_phys + 10.0 * l_data
      return total_loss, {
        "loss": total_loss,
        "l_phys": l_phys,
        "l_data": l_data,
      }

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
      model
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, metrics

  # Training State for early stopping
  best_loss = float("inf")
  patience_counter = 0

  if log_path:
    print(f"Starting training. Logging to {log_path}...")
  else:
    print("Starting training (logging disabled)...")

  with _get_log_writer(log_path) as writer_context:
    for i in range(max_steps):
      key, subkey = jax.random.split(key)
      # Sample points in the normalized training domain [-1, 1]
      coords = jax.random.uniform(subkey, (128, 4), minval=-1.0, maxval=1.0)

      model, opt_state, _, metrics = step(
        model, opt_state, coords, redshifts, target_mu, mu_err
      )

      current_loss = float(metrics["loss"])

      # Logging & Telemetry
      if writer_context and i % 10 == 0:
        log_writer, log_file = writer_context
        log_writer.writerow(
          {
            "step": i,
            "loss": f"{current_loss:.6e}",
            "l_phys": f"{float(metrics['l_phys']):.6e}",
            "l_data": f"{float(metrics['l_data']):.6e}",
          }
        )
        log_file.flush()

      if i % 100 == 0:
        print(f"Step {i}, Loss: {current_loss:.6e}")

      # Early Stopping & Safety Checks
      if jnp.isnan(current_loss):
        print(f"CRITICAL: NaN loss detected at step {i}.")
        break

      if current_loss < target_loss:
        print(
          f"Target loss reached at step {i}. Final loss: {current_loss:.6e}"
        )
        break

      if current_loss < best_loss:
        best_loss = current_loss
        patience_counter = 0
      else:
        patience_counter += 1

      if patience_counter >= patience:
        print(f"Early stopping triggered at step {i} (patience reached).")
        break

  return model
