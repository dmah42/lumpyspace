"""
Execution entry point for production training on Pantheon+ data.
"""

import os
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from src.core.metric import MetricNN
from src.training.data import load_pantheon_plus
from src.training.trainer import train_model


def run_training(
  max_steps: int = 10000,
  learning_rate: float = 1e-4,
  target_loss: float = 1e-6,
  patience: int = 500,
  checkpoint_path: str = "checkpoints/pinn_metric.eqx",
  log_path: str = "logs/training_metrics.csv",
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
    # load_pantheon_plus returns (z, mu, mu_err)
    data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] = load_pantheon_plus()
  except FileNotFoundError:
    print("Error: Pantheon+ data not found. Run scripts/download_pantheon.py.")
    return

  # 3. Initialize Model
  print("Initializing 4D Metric PINN...")
  model = MetricNN(model_key)

  # 4. Run Training
  trained_model = train_model(
    model,
    data,
    max_steps=max_steps,
    learning_rate=learning_rate,
    target_loss=target_loss,
    patience=patience,
    log_path=log_path,
    key=train_key,
  )

  # 5. Save Model Checkpoint
  os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
  eqx.tree_serialise_leaves(checkpoint_path, trained_model)
  print(f"Training session concluded. Model saved to {checkpoint_path}")


if __name__ == "__main__":
  run_training()
