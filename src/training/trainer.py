"""
Training library for the PINN, including early stopping and telemetry.
"""

import csv
import os
from contextlib import contextmanager

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from src.training.loss import get_bao_loss, get_data_loss, get_efe_loss


@contextmanager
def _get_log_writer(log_path: str | None, resume: bool = False):
  """Helper to handle optional CSV logging."""
  if log_path is None:
    yield None
  else:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.exists(log_path) and os.path.getsize(log_path) > 0
    mode = "a" if (resume and file_exists) else "w"
    with open(log_path, mode=mode, newline="") as log_file:
      log_writer = csv.DictWriter(
        log_file,
        fieldnames=["step", "loss", "l_phys", "l_sn", "l_bao", "kappa_rho_0"],
      )
      if not (resume and file_exists):
        log_writer.writeheader()
      yield log_writer, log_file


def train_model(
  model: eqx.Module,
  data: tuple[
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
  ],
  max_steps: int = 10000,
  learning_rate: float = 1e-4,
  target_loss: float = 1e-6,
  patience: int = 500,
  kick_period: int = 1000,
  peak_learning_rate: float = 1e-3,
  log_path: str | None = "logs/training_metrics.csv",
  checkpoint_path: str | None = None,
  key: jax.Array | None = None,
  w_efe: float = 1.0,
  w_sn: float = 10.0,
  w_bao: float = 0.5,
  resume: bool = False,
) -> eqx.Module:
  """
  Executes the training loop for the PINN.
  Minimizes combined EFE and Data residuals with early stopping and telemetry.
  """

  # Periodic learning rate schedule to continuously kick the model out
  # of local minima. Decays rapidly, then stays flat at baseline.
  def lr_schedule(step):
    cycle_length = kick_period
    decay_length = cycle_length // 5  # Decay over the first 20% of the cycle

    cycle_step = step % cycle_length
    progress = jnp.minimum(cycle_step / decay_length, 1.0)
    cosine_val = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))

    return learning_rate + (peak_learning_rate - learning_rate) * cosine_val

  # Use gradient clipping to stabilize training
  optimizer = optax.chain(
    optax.clip_by_global_norm(1.0), optax.adam(lr_schedule)
  )
  opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

  sn_data, bao_data = data
  sn_z, sn_mu, sn_err = sn_data
  bao_z, bao_dm, bao_dh, bao_cov = bao_data

  @eqx.filter_jit
  def step(
    model: eqx.Module,
    opt_state: optax.OptState,
    coords: jnp.ndarray,
    sn_z: jnp.ndarray,
    sn_mu: jnp.ndarray,
    sn_err: jnp.ndarray,
    bao_z: jnp.ndarray,
    bao_dm: jnp.ndarray,
    bao_dh: jnp.ndarray,
    bao_cov: jnp.ndarray,
  ) -> tuple[eqx.Module, optax.OptState, jnp.ndarray, dict[str, jnp.ndarray]]:
    def loss_fn(model):
      # 1. Physics Loss (EFE residuals)
      v_efe_loss = jax.vmap(lambda c: get_efe_loss(model, c))
      l_phys = jnp.mean(v_efe_loss(coords))

      # 2. Data Loss (Supernova chi-squared)
      l_sn = get_data_loss(model, sn_z, sn_mu, sn_err)

      # 3. BAO Loss (3D Chi-squared)
      l_bao = get_bao_loss(model, bao_z, bao_dm, bao_dh, bao_cov)

      total_loss = w_efe * l_phys + w_sn * l_sn + w_bao * l_bao
      return total_loss, {
        "loss": total_loss,
        "l_phys": l_phys,
        "l_sn": l_sn,
        "l_bao": l_bao,
        "kappa_rho_0": model.kappa_rho_0[0],
      }

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
      model
    )

    # Scale the gradient of the trainable matter density parameter to overcome
    # the suppression caused by the dimensionality of the MLP weights during
    # global norm clipping.
    boosted_kappa_grad = grads.kappa_rho_0 * 80.0
    grads = eqx.tree_at(lambda m: m.kappa_rho_0, grads, boosted_kappa_grad)

    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, metrics

  # Training State for early stopping
  best_loss = float("inf")
  patience_counter = 0

  start_step = 0
  if resume and log_path:
    if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
      try:
        with open(log_path) as f:
          reader = list(csv.DictReader(f))
          if reader:
            start_step = int(reader[-1]["step"]) + 10
      except Exception as e:
        print(f"Warning: Could not parse last step from log: {e}")

  if log_path:
    if resume and start_step > 0:
      print(f"Resuming log at step {start_step}. Appending to {log_path}...")
    else:
      print(f"Starting training. Logging to {log_path}...")
  else:
    print("Starting training (logging disabled)...")

  with _get_log_writer(log_path, resume=resume) as writer_context:
    for i in range(max_steps):
      key, subkey = jax.random.split(key)
      # We sample coordinates across the physical domain [-4.0, 1.0]
      # to ensure the metric is constrained well beyond the supernova data.
      t_coords = jax.random.uniform(subkey, (128, 1), minval=-4.0, maxval=1.0)
      spatial_coords = jax.random.uniform(
        subkey, (128, 3), minval=-1.0, maxval=1.0
      )
      coords = jnp.concatenate([t_coords, spatial_coords], axis=1)

      model, opt_state, _, metrics = step(
        model,
        opt_state,
        coords,
        sn_z,
        sn_mu,
        sn_err,
        bao_z,
        bao_dm,
        bao_dh,
        bao_cov,
      )

      current_loss = float(metrics["loss"])

      # Logging & Telemetry
      if writer_context and i % 10 == 0:
        log_writer, log_file = writer_context
        log_writer.writerow(
          {
            "step": start_step + i,
            "loss": f"{current_loss:.6e}",
            "l_phys": f"{float(metrics['l_phys']):.6e}",
            "l_sn": f"{float(metrics['l_sn']):.6e}",
            "l_bao": f"{float(metrics['l_bao']):.6e}",
            "kappa_rho_0": f"{float(metrics['kappa_rho_0']):.6e}",
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
        if checkpoint_path:
          print(
            f"Saving checkpoint with loss {current_loss:.6e} at step {i}..."
          )
          os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
          eqx.tree_serialise_leaves(checkpoint_path, model)
      else:
        patience_counter += 1

      if patience_counter >= patience:
        print(f"Early stopping triggered at step {i} (patience reached).")
        break

  return model
