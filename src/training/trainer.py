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
from src.training.constraints import ConstraintManager
from src.training.loss import (
  CONSTRAINT_METRICS,
  METRIC_BAO,
  METRIC_EXPAND,
  METRIC_LOSS,
  METRIC_MEAN_W_SPATIAL,
  METRIC_OMEGA_M,
  METRIC_PHYS,
  METRIC_SHEAR,
  METRIC_SN,
  METRIC_SPATIAL,
  METRIC_WEC,
  apply_spatial_weight,
  get_bao_loss,
  get_cmb_loss,
  get_data_loss,
  get_efe_loss,
)
from src.training.scheduler import create_geometric_sgdr_schedule

START_W_PENALTY = 1.0


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
          METRIC_LOSS,
          METRIC_PHYS,
          METRIC_WEC,
          METRIC_EXPAND,
          METRIC_SHEAR,
          METRIC_SPATIAL,
          METRIC_SN,
          METRIC_BAO,
          METRIC_OMEGA_M,
        ],
      )
      if not (resume and file_exists):
        log_writer.writeheader()
      yield log_writer, log_file


def save_checkpoint(
  path: str,
  model: eqx.Module,
  opt_state: optax.OptState,
  step: int,
  best_loss: float,
  constraints: dict[str, ConstraintManager],
) -> None:
  """Helper to serialize model leaves, opt_state, and meta state atomically."""
  os.makedirs(os.path.dirname(path), exist_ok=True)
  eqx.tree_serialise_leaves(path, model)
  eqx.tree_serialise_leaves(path + ".opt", opt_state)
  state: TrainingState = {
    "step": step,
    "best_loss": best_loss,
    METRIC_WEC: constraints[METRIC_WEC].to_dict(),
    METRIC_EXPAND: constraints[METRIC_EXPAND].to_dict(),
    METRIC_SHEAR: constraints[METRIC_SHEAR].to_dict(),
    METRIC_SPATIAL: constraints[METRIC_SPATIAL].to_dict(),
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
  kick_period_0: int = 500,
  kick_period_mult: float = 1.5,
  peak_learning_rate: float = 1e-3,
  log_path: str | None = "logs/training_metrics.csv",
  checkpoint_path: str | None = None,
  key: jax.Array | None = None,
  w_efe: float = 1.0,
  w_sn: float = 10.0,
  w_bao: float = 0.5,
  adaptive_check_interval: int = 1000,
  resume: bool = False,
) -> eqx.Module:
  """
  Executes the training loop for the PINN.
  Minimizes combined EFE and Data residuals with early stopping and telemetry.
  """

  lr_schedule = create_geometric_sgdr_schedule(
    learning_rate=learning_rate,
    peak_learning_rate=peak_learning_rate,
    kick_period_0=kick_period_0,
    kick_period_mult=kick_period_mult,
  )

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
    lambdas: dict[str, jnp.ndarray],
    w_penalties: dict[str, jnp.ndarray],
  ) -> tuple[eqx.Module, optax.OptState, jnp.ndarray, dict[str, jnp.ndarray]]:
    # We sample coordinates across the physical domain [-4.0, 1.0] in four
    # spans: Supernova data ([-2.5, 1.0]), inactive range ([-3.99, -2.5]),
    # spatial coordinates for the spatial weights ([-1.0, 1.0]), and the CMB
    # constraints ([-4.0, -3.99]).
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Span = 3.5. 800 points -> ~228 points per unit redshift
    t_active = jax.random.uniform(k1, (800, 1), minval=-2.5, maxval=1.0)

    # Span = 1.49. 200 points -> ~134 points per unit redshift
    t_inactive = jax.random.uniform(k2, (200, 1), minval=-3.99, maxval=-2.5)

    t_coords = jnp.concatenate([t_active, t_inactive], axis=0)
    spatial_coords = jax.random.uniform(k3, (1000, 3), minval=-1.0, maxval=1.0)
    coords = jnp.concatenate([t_coords, spatial_coords], axis=1)

    # Sample narrow CMB boundary slice for deep past priors
    t_cmb = jax.random.uniform(k4, (50, 1), minval=-4.0, maxval=-3.99)
    spatial_cmb = jax.random.uniform(k4, (50, 3), minval=-1.0, maxval=1.0)
    coords_cmb = jnp.concatenate([t_cmb, spatial_cmb], axis=1)

    def get_al_penalty(loss_val, lam, w):
      """Computes the Augmented Lagrangian penalty."""
      return lam * loss_val + 0.5 * w * (loss_val**2)

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
      al_penalty_raw_vals = get_al_penalty(
        l_wec_raw_vals, lambdas["l_wec"], w_penalties["l_wec"]
      )

      # Total raw physics error per point
      total_phys_raw_vals = w_efe * l_phys_raw_vals + al_penalty_raw_vals

      # Apply the 4D Spatial Curriculum point-wise
      weighted_phys_vals = apply_spatial_weight(total_phys_raw_vals, w_4d_vals)
      total_weighted_phys = jnp.mean(weighted_phys_vals)

      # 1.5 CMB Boundary Penalties (Deep Past)
      v_cmb_loss = jax.vmap(lambda c: get_cmb_loss(model, c))
      l_expand_raw, l_shear_raw, l_spatial_raw = v_cmb_loss(coords_cmb)

      l_expand = jnp.mean(l_expand_raw)
      l_shear = jnp.mean(l_shear_raw)
      l_spatial_cmb = jnp.mean(l_spatial_raw)

      cmb_penalty = (
        get_al_penalty(l_expand, lambdas["l_expand"], w_penalties["l_expand"])
        + get_al_penalty(l_shear, lambdas["l_shear"], w_penalties["l_shear"])
        + get_al_penalty(
          l_spatial_cmb, lambdas["l_spatial"], w_penalties["l_spatial"]
        )
      )

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
      # (EFE + AL WEC) with the global data losses and CMB constraints.
      total_loss = (
        total_weighted_phys + cmb_penalty + w_sn * l_sn + w_bao * l_bao
      )
      return total_loss, {
        METRIC_LOSS: total_loss,
        METRIC_PHYS: l_phys,
        METRIC_WEC: l_wec,
        METRIC_EXPAND: l_expand,
        METRIC_SHEAR: l_shear,
        METRIC_SPATIAL: l_spatial_cmb,
        METRIC_SN: l_sn,
        METRIC_BAO: l_bao,
        METRIC_OMEGA_M: omega_m_today[0],
        METRIC_MEAN_W_SPATIAL: mean_w_spatial,
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
  start_step = 0
  patience_counter = 0

  constraints = {
    METRIC_WEC: ConstraintManager(),
    METRIC_EXPAND: ConstraintManager(),
    METRIC_SHEAR: ConstraintManager(),
    METRIC_SPATIAL: ConstraintManager(),
  }

  if resume and checkpoint_path:
    latest_path = checkpoint_path.replace(".eqx", "_latest.eqx")
    meta_path = latest_path + ".meta"
    opt_path = latest_path + ".opt"
    state = load_meta(meta_path)

    start_step = state["step"]
    best_loss = state["best_loss"]

    constraints[METRIC_WEC] = ConstraintManager.from_dict(state["l_wec"])
    constraints[METRIC_EXPAND] = ConstraintManager.from_dict(state["l_expand"])
    constraints[METRIC_SHEAR] = ConstraintManager.from_dict(state["l_shear"])
    constraints[METRIC_SPATIAL] = ConstraintManager.from_dict(
      state["l_spatial"]
    )

    if os.path.exists(opt_path):
      opt_state = eqx.tree_deserialise_leaves(opt_path, opt_state)

  with _get_log_writer(log_path, resume=resume) as writer_context:
    for i in range(max_steps):
      key, subkey = jax.random.split(key)

      lambdas = {}
      w_penalties = {}
      for k, m in constraints.items():
        lam, w = m.get_arrays()
        lambdas[k] = jnp.array(lam)
        w_penalties[k] = jnp.array(w)

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
        lambdas,
        w_penalties,
      )

      current_loss = float(metrics[METRIC_LOSS])
      current_step = start_step + i

      # Update Constraints
      for k in CONSTRAINT_METRICS:
        manager = constraints[k]
        val = float(metrics[k])
        bumped = manager.update(val, current_step, adaptive_check_interval)
        if bumped:
          w_val = manager.scheduler.w_penalty
          print(f"Adaptive Penalty: {k} bumped to {w_val:.1f}")

      # Logging & Telemetry
      if writer_context and i % 10 == 0:
        log_writer, log_file = writer_context
        log_writer.writerow(
          {
            "step": current_step,
            METRIC_LOSS: f"{current_loss:.6e}",
            METRIC_PHYS: f"{float(metrics[METRIC_PHYS]):.6e}",
            METRIC_WEC: f"{float(metrics[METRIC_WEC]):.6e}",
            METRIC_EXPAND: f"{float(metrics[METRIC_EXPAND]):.6e}",
            METRIC_SHEAR: f"{float(metrics[METRIC_SHEAR]):.6e}",
            METRIC_SPATIAL: f"{float(metrics[METRIC_SPATIAL]):.6e}",
            METRIC_SN: f"{float(metrics[METRIC_SN]):.6e}",
            METRIC_BAO: f"{float(metrics[METRIC_BAO]):.6e}",
            METRIC_OMEGA_M: f"{float(metrics[METRIC_OMEGA_M]):.6e}",
          }
        )
        log_file.flush()

      if i % 100 == 0:
        print(
          f"step {current_step}, loss: {current_loss:.6e} | "
          f"{METRIC_PHYS}: {float(metrics[METRIC_PHYS]):.3e} | "
          f"{METRIC_SN}: {float(metrics[METRIC_SN]):.3e} | "
          f"{METRIC_WEC}: {float(metrics[METRIC_WEC]):.3e}"
        )

        # Continually save the latest state so resumption never loses progress
        if checkpoint_path:
          latest_path = checkpoint_path.replace(".eqx", "_latest.eqx")
          save_checkpoint(
            latest_path,
            model,
            opt_state,
            current_step,
            best_loss,
            constraints,
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
            opt_state,
            current_step,
            best_loss,
            constraints,
          )

          # Also update the latest checkpoint, because this is now the most
          # recent state we have safely serialized to disk!
          latest_path = checkpoint_path.replace(".eqx", "_latest.eqx")
          save_checkpoint(
            latest_path,
            model,
            opt_state,
            current_step,
            best_loss,
            constraints,
          )
      else:
        patience_counter += 1

      if patience_counter >= patience:
        print(
          f"Early stopping triggered at step {current_step} (patience reached)."
        )
        break

  return model
