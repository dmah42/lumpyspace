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

from src.training.loss import (
  apply_spatial_weight,
  get_bao_loss,
  get_data_loss,
  get_efe_loss,
)


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
        fieldnames=[
          "step",
          "loss",
          "l_phys",
          "l_wec",
          "l_sn",
          "l_bao",
          "omega_m",
          "lambda_wec",
          "mean_w_spatial",
        ],
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
  kick_period: int = 200,
  peak_learning_rate: float = 1e-3,
  log_path: str | None = "logs/training_metrics.csv",
  checkpoint_path: str | None = None,
  key: jax.Array | None = None,
  w_efe: float = 1.0,
  w_sn: float = 10.0,
  w_bao: float = 0.5,
  w_wec: float = 10.0,
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

  # Use gradient clipping to stabilize training for the MLP,
  # but allow Omega_m to learn independently to avoid being suppressed.
  def label_fn(tree):
    labels = jax.tree_util.tree_map(lambda _: "mlp", tree)
    return eqx.tree_at(
      lambda m: m.omega_m_raw, labels, "omega", is_leaf=lambda x: x is None
    )

  optimizers = {
    "mlp": optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule)),
    "omega": optax.adam(lambda s: lr_schedule(s) * 100.0),
  }

  optimizer = optax.multi_transform(optimizers, label_fn)
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
    lambda_wec: jnp.ndarray,
    w_wec: jnp.ndarray,
  ) -> tuple[eqx.Module, optax.OptState, jnp.ndarray, dict[str, jnp.ndarray]]:
    def loss_fn(model):
      # Compute matter density and Omega_m today in a single pass to avoid
      # redundant AD evaluations.
      kappa_rho_0_today, omega_m_today = model.get_cosmology_today()

      # 1. Physics Loss (EFE residuals & WEC penalties)
      v_efe_loss = jax.vmap(
        lambda c: get_efe_loss(model, c, kappa_rho_0_today[0])
      )
      l_phys_raw_vals, l_wec_raw_vals, w_4d_vals = v_efe_loss(coords)

      # Compute point-wise Augmented Lagrangian penalty for WEC
      al_penalty_raw_vals = (
        lambda_wec_val * l_wec_raw_vals + 0.5 * w_wec_val * (l_wec_raw_vals**2)
      )

      # Total raw physics error per point
      total_phys_raw_vals = w_efe * l_phys_raw_vals + al_penalty_raw_vals

      # Apply the 4D Spatial Curriculum point-wise
      weighted_phys_vals = apply_spatial_weight(total_phys_raw_vals, w_4d_vals)
      total_weighted_phys = jnp.mean(weighted_phys_vals)

      # Means for logging and lambda update
      l_phys = jnp.mean(l_phys_raw_vals)
      l_wec = jnp.mean(l_wec_raw_vals)
      mean_w_spatial = jnp.mean(w_4d_vals)

      # 2. Data Loss (Supernova chi-squared)
      l_sn = get_data_loss(model, sn_z, sn_mu, sn_err)

      # 3. BAO Loss (3D Chi-squared)
      l_bao = jax.lax.cond(
        w_bao > 0.0,
        lambda _: get_bao_loss(model, bao_z, bao_dm, bao_dh, bao_cov),
        lambda _: jnp.array(0.0),
        operand=None,
      )

      # Total Loss seamlessly integrates the spatially weighted physical errors
      # (EFE + AL WEC) with the global data losses.
      total_loss = total_weighted_phys + w_sn * l_sn + w_bao * l_bao
      return total_loss, {
        "loss": total_loss,
        "l_phys": l_phys,
        "l_wec": l_wec,
        "l_sn": l_sn,
        "l_bao": l_bao,
        "omega_m": omega_m_today[0],
        "mean_w_spatial": mean_w_spatial,
      }

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
      model
    )

    updates, opt_state = optimizer.update(
      grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)

    # Projected Gradient Descent: enforce physical bounds [0.05, 0.3]
    model = eqx.tree_at(
      lambda m: m.omega_m_raw, model, jnp.clip(model.omega_m_raw, 0.05, 0.3)
    )
    return model, opt_state, loss, metrics

  # Training State for early stopping
  best_loss = float("inf")
  patience_counter = 0

  start_step = 0
  start_lambda_wec = 0.0
  if resume and log_path:
    if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
      try:
        with open(log_path) as f:
          reader = list(csv.DictReader(f))
          if reader:
            start_step = int(reader[-1]["step"]) + 10
            start_lambda_wec = float(reader[-1]["lambda_wec"])
      except Exception as e:
        print(f"Warning: Could not parse last step from log: {e}")

  if log_path:
    if resume and start_step > 0:
      print(f"Resuming log at step {start_step}. Appending to {log_path}...")
    else:
      print(f"Starting training. Logging to {log_path}...")
  else:
    print("Starting training (logging disabled)...")

  # Initialize Augmented Lagrangian multipliers and scaling parameter
  lambda_wec_val = jnp.array(start_lambda_wec)
  w_wec_val = jnp.array(w_wec)

  with _get_log_writer(log_path, resume=resume) as writer_context:
    for i in range(max_steps):
      key, subkey = jax.random.split(key)
      # We sample coordinates across the physical domain [-4.0, 1.0]
      # to ensure the metric is constrained well beyond the supernova data.
      k1, k2, k3 = jax.random.split(subkey, 3)
      # 250/3.5 ~ 71.4 points per unit redshift
      t_active = jax.random.uniform(k1, (250, 1), minval=-2.5, maxval=1.0)
      # 30/0.7 ~ 40 points per unit redshift
      t_inactive = jax.random.uniform(k2, (30, 1), minval=-3.2, maxval=-2.5)
      t_coords = jnp.concatenate([t_active, t_inactive], axis=0)

      spatial_coords = jax.random.uniform(k3, (280, 3), minval=-1.0, maxval=1.0)
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
        lambda_wec_val,
        w_wec_val,
      )

      current_loss = float(metrics["loss"])
      current_step = start_step + i

      # Update Augmented Lagrangian multiplier:
      # lambda <- lambda + mu * violation
      l_wec_val = metrics["l_wec"]
      lambda_wec_val = lambda_wec_val + w_wec_val * l_wec_val

      # Logging & Telemetry
      if writer_context and i % 10 == 0:
        log_writer, log_file = writer_context
        log_writer.writerow(
          {
            "step": current_step,
            "loss": f"{current_loss:.6e}",
            "l_phys": f"{float(metrics['l_phys']):.6e}",
            "l_wec": f"{float(metrics['l_wec']):.6e}",
            "l_sn": f"{float(metrics['l_sn']):.6e}",
            "l_bao": f"{float(metrics['l_bao']):.6e}",
            "omega_m": f"{float(metrics['omega_m']):.6e}",
            "lambda_wec": f"{float(lambda_wec_val):.6e}",
            "mean_w_spatial": f"{float(metrics['mean_w_spatial']):.6e}",
          }
        )
        log_file.flush()

      if i % 100 == 0:
        print(
          f"Step {current_step}, Loss: {current_loss:.6e} | "
          f"L_EFE: {float(metrics['l_phys']):.3e} | "
          f"L_SN: {float(metrics['l_sn']):.3e} | "
          f"L_WEC: {float(metrics['l_wec']):.3e}"
        )

      # Early Stopping & Safety Checks
      if jnp.isnan(current_loss):
        print(f"CRITICAL: NaN loss detected at step {current_step}.")
        break

      if current_loss < target_loss:
        print(
          f"Target loss reached at step {current_step}. "
          f"Final loss: {current_loss:.6e}"
        )
        break

      if current_loss < best_loss:
        best_loss = current_loss
        patience_counter = 0
        if checkpoint_path:
          print(
            f"Saving checkpoint with loss {current_loss:.6e} at step "
            f"{current_step}..."
          )
          os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
          eqx.tree_serialise_leaves(checkpoint_path, model)
      else:
        patience_counter += 1

      if patience_counter >= patience:
        print(
          f"Early stopping triggered at step {current_step} (patience reached)."
        )
        break

  return model
