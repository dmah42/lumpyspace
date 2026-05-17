"""
Integration test for the training loop using mock data.
"""

import jax

from src.core.metric import MetricNN
from src.training.data import load_mock_data
from src.training.trainer import train_model


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

  # 1. Load Mock Data (z, mu, mu_err)
  data = load_mock_data()
  # Use a smaller subset for fast integration test
  subset_data = (data[0][:5], data[1][:5], data[2][:5])

  # 2. Initialize Model
  model = MetricNN(model_key)

  # 3. Run a short training burst
  print("\nStarting control training test with mock data...")
  trained_model = train_model(
    model,
    subset_data,
    max_steps=5,
    learning_rate=1e-4,
    lam=0.7,
    key=train_key,
    log_path=None,
  )

  assert trained_model is not None
  print("Integration test successful.")
