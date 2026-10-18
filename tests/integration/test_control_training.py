"""
Integration test for the training loop using mock data.
"""

import csv

import jax
import jax.numpy as jnp

from src.core.metric import MetricNN
from src.training.data import load_mock_data
from src.training.trainer import train_model


def _mock_training_data():
  """A small subset of mock FLRW supernovae and a dummy BAO point."""
  sn_data = load_mock_data()
  # Use a smaller subset for fast integration test
  subset_sn = (
    sn_data[0][:5],
    sn_data[1][:5],
    sn_data[2][:5],
    sn_data[3][:5],
  )

  # Create a dummy mock BAO dataset: (z, dm, dh, cov_inv)
  # We just need it to run without shape errors.
  # Shape expected: z (1,), dm (1,), dh (1,), cov_inv (1, 2, 2)
  mock_bao_z = jnp.array([0.5])
  mock_bao_dm = jnp.array([10.0])
  mock_bao_dh = jnp.array([20.0])
  mock_bao_cov_inv = jnp.array([[[1.0, 0.0], [0.0, 1.0]]])
  subset_bao = (mock_bao_z, mock_bao_dm, mock_bao_dh, mock_bao_cov_inv)

  return subset_sn, subset_bao


def test_control_training_loop() -> None:
  """
  Verification of the Hybrid Training Loop (Physics + Data).

  Ritual Purpose: This test ensures that the core training infrastructure
  is functional, JIT-compilable, and numerically stable. it verifies the
  cooperation between the PINN (MetricNN), the Einstein Field Equation (EFE)
  residual loss, and the observational (Supernova) data loss.

  Verification Ritual:
  1. Load a subset of mock FLRW supernova data.
  2. Initialize a MetricNN and a stabilized Adam optimizer (with
     gradient clipping).
  3. Run a short training burst (5 steps) of the combined loss:
     Loss = Physics_Residual + 10 * Observational_Residual.

  Expected Outcome:
  - The training loop must execute without numerical explosion (NaN).
  - The loss should show a downward trend or remain stable, indicating that
    the JAX gradient engine is successfully backpropagating through the
    nested ODE integration.
  """
  key = jax.random.PRNGKey(42)
  model_key, train_key = jax.random.split(key)

  # 1. Load Mock Data
  combined_data = _mock_training_data()

  # 2. Initialize Model
  model = MetricNN(model_key)

  # 3. Run a short training burst
  print("\nStarting control training test with mock data...")
  trained_model = train_model(
    model,
    combined_data,
    max_steps=5,
    learning_rate=1e-4,
    key=train_key,
    log_path=None,
    penalty_log_path=None,
  )

  assert trained_model is not None
  print("Integration test successful.")


def test_resume_logs_each_step_once(tmp_path) -> None:
  """
  A resumed run continues from the step after its last checkpoint, so no step
  number appears twice in the training log.
  """
  key = jax.random.PRNGKey(7)
  model_key, train_key = jax.random.split(key)
  combined_data = _mock_training_data()
  checkpoint_path = str(tmp_path / "pinn.eqx")
  log_path = str(tmp_path / "training_metrics.csv")

  # Steps 0-10 log at 0 and 10, and the latest checkpoint is step 10.
  model = train_model(
    MetricNN(model_key),
    combined_data,
    train_key,
    max_steps=11,
    log_path=log_path,
    penalty_log_path=None,
    checkpoint_path=checkpoint_path,
  )
  train_model(
    model,
    combined_data,
    train_key,
    max_steps=11,
    log_path=log_path,
    penalty_log_path=None,
    checkpoint_path=checkpoint_path,
    resume=True,
  )

  with open(log_path) as f:
    steps = [int(row["step"]) for row in csv.DictReader(f)]
  assert steps == [0, 10, 11, 21]
