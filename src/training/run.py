"""
Execution entry point for production training on Pantheon+ data.
"""

import os

import equinox as eqx
import jax

from src.core.metric import MetricNN
from src.training.data import load_bao_data, load_pantheon_plus
from src.training.trainer import train_model


def run_training(
  max_steps: int = 2000,
  learning_rate: float = 1e-5,
  checkpoint_path: str = "checkpoints/pinn_metric.eqx",
  log_path: str = "logs/training_metrics.csv",
  w_efe: float = 1.0,
  w_sn: float = 5.0,
  w_bao: float = 1.0,
) -> None:
  """
  Orchestrates the full training pipeline on the real dataset.
  """
  # 1. Setup PRNG
  key = jax.random.PRNGKey(42)
  model_key, train_key = jax.random.split(key)

  # 2. Load Real Data
  print("Loading Pantheon+ Supernova dataset...")
  try:
    sn_data = load_pantheon_plus()
  except FileNotFoundError:
    print("Error: Pantheon+ data not found. Run scripts/download_pantheon.py.")
    return

  print("Loading SDSS DR12 BAO dataset...")
  try:
    bao_data = load_bao_data()
  except FileNotFoundError:
    print("Error: BAO data not found. Run scripts/generate_bao_data.py.")
    return

  combined_data = (sn_data, bao_data)

  # 3. Initialize Model
  print("Initializing 4D Metric PINN...")
  model = MetricNN(model_key)
  resume = os.path.exists(checkpoint_path)
  if resume:
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    model = eqx.tree_deserialise_leaves(checkpoint_path, model)
  else:
    print("No checkpoint found. Starting from scratch.")

  # 4. Run Training
  train_model(
    model,
    combined_data,
    max_steps=max_steps,
    learning_rate=learning_rate,
    log_path=log_path,
    kick_period=200,
    peak_learning_rate=1.5e-4,
    checkpoint_path=checkpoint_path,
    key=train_key,
    w_efe=w_efe,
    w_sn=w_sn,
    w_bao=w_bao,
    resume=resume,
  )

  # 5. Training Complete
  print(f"Training session concluded. Best model is saved at {checkpoint_path}")


if __name__ == "__main__":
  run_training()
