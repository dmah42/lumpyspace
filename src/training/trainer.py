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

from src.training.checkpoint import TrainingState, load_meta, save_meta
from src.training.loss import (
  apply_spatial_weight,
  get_bao_loss,
  get_data_loss,
  get_efe_loss,
)
from src.training.scheduler import AdaptivePenaltyState

START_W_WEC = 1.0


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
        ],
      )
      if not (resume and file_exists):
        log_writer.writeheader()
      yield log_writer, log_file


def save_checkpoint(
  path: str,
  model: eqx.Module,
  step: int,
  best_loss: float,
  lambda_wec: float,
  wec_scheduler: AdaptivePenaltyState,
) -> None:
  """Helper to serialize model leaves and meta state atomically."""
  os.makedirs(os.path.dirname(path), exist_ok=True)
  eqx.tree_serialise_leaves(path, model)
  state: TrainingState = {
    "step": step,
    "best_loss": best_loss,
    "lambda_wec": lambda_wec,
    "w_penalty": wec_scheduler.w_penalty,
    "ema_violation": wec_scheduler.ema_violation,
    "last_check_violation": wec_scheduler.last_check_violation,
  }
  save_meta(path + ".meta", state)


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
  adaptive_check_interval: int = 500,
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
    key: jax.Array,
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
    # We sample coordinates across the physical domain [-4.0, 1.0]
    # to ensure the metric is constrained well beyond the supernova data.
    k1, k2, k3 = jax.random.split(key, 3)

    # Span = 3.5. 800 points -> ~228 points per unit redshift
    t_active = jax.random.uniform(k1, (800, 1), minval=-2.5, maxval=1.0)

    # Span = 1.49. 200 points -> ~134 points per unit redshift
    t_inactive = jax.random.uniform(k2, (200, 1), minval=-3.99, maxval=-2.5)

    t_coords = jnp.concatenate([t_active, t_inactive], axis=0)
    spatial_coords = jax.random.uniform(k3, (1000, 3), minval=-1.0, maxval=1.0)
    coords = jnp.concatenate([t_coords, spatial_coords], axis=1)

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
      al_penalty_raw_vals = lambda_wec * l_wec_raw_vals + 0.5 * w_wec * (
        l_wec_raw_vals**2
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
  start_w_wec = START_W_WEC
  start_ema = 1.0
  start_last_check = float("inf")

  if resume and checkpoint_path:
    latest_path = checkpoint_path.replace(".eqx", "_latest.eqx")
    meta_path = latest_path + ".meta"
    state = load_meta(meta_path)

    start_step = state["step"]
    best_loss = state["best_loss"]
    start_lambda_wec = state["lambda_wec"]
    start_w_wec = state["w_penalty"]
    start_ema = state["ema_violation"]
    start_last_check = state["last_check_violation"]

  if log_path:
    if resume and start_step > 0:
      print(f"Resuming log at step {start_step}. Appending to {log_path}...")
    else:
      print(f"Starting training. Logging to {log_path}...")
  else:
    print("Starting training (logging disabled)...")

  # Initialize Augmented Lagrangian multipliers and scaling parameter
  lambda_wec_val = jnp.array(start_lambda_wec)
  w_wec_val = jnp.array(start_w_wec)
  wec_scheduler = AdaptivePenaltyState(
    w_penalty=start_w_wec,
    ema_violation=start_ema,
    last_check_violation=start_last_check,
  )

  with _get_log_writer(log_path, resume=resume) as writer_context:
    for i in range(max_steps):
      key, subkey = jax.random.split(key)

      model, opt_state, _, metrics = step(
        model,
        opt_state,
        subkey,
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

      # Update EMA for constraint violation and check for penalty bump
      l_wec_val_scalar = float(metrics["l_wec"])
      bumped = wec_scheduler.update(
        l_wec_val_scalar,
        i,
        check_interval=adaptive_check_interval,
      )

      if bumped:
        print(
          f"Adaptive Penalty: w_wec bumped to {wec_scheduler.w_penalty:.1f} "
          f"(EMA l_wec: {wec_scheduler.ema_violation:.6e})"
        )

      w_wec_val = jnp.array(wec_scheduler.w_penalty)

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
          }
        )
        log_file.flush()

      if i % 100 == 0:
        print(
          f"step {current_step}, loss: {current_loss:.6e} | "
          f"l_efe: {float(metrics['l_phys']):.3e} | "
          f"l_sn: {float(metrics['l_sn']):.3e} | "
          f"l_wec: {float(metrics['l_wec']):.3e}"
        )

        # Continually save the latest state so resumption never loses progress
        if checkpoint_path:
          latest_path = checkpoint_path.replace(".eqx", "_latest.eqx")
          save_checkpoint(
            latest_path,
            model,
            current_step,
            best_loss,
            float(lambda_wec_val),
            wec_scheduler,
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
            f"Saving best model with loss {current_loss:.6e} at step "
            f"{current_step}..."
          )
          save_checkpoint(
            checkpoint_path,
            model,
            current_step,
            best_loss,
            float(lambda_wec_val),
            wec_scheduler,
          )

          # Also update the latest checkpoint, because this is now the most
          # recent state we have safely serialized to disk!
          latest_path = checkpoint_path.replace(".eqx", "_latest.eqx")
          save_checkpoint(
            latest_path,
            model,
            current_step,
            best_loss,
            float(lambda_wec_val),
            wec_scheduler,
          )
      else:
        patience_counter += 1

      if patience_counter >= patience:
        print(
          f"Early stopping triggered at step {current_step} (patience reached)."
        )
        break

  return model
