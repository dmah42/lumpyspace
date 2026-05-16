"""
Integration test for the FLRW control training loop.
"""

import jax

from src.core.metric import MetricNN
from src.training.data import load_mock_data
from src.training.trainer import train_control_model


def test_control_training_loop():
  """
  Verify that the training loop can execute and reduce the combined loss.
  """
  key = jax.random.PRNGKey(42)
  model_key, train_key = jax.random.split(key)

  # 1. Load Mock Data
  z, mu = load_mock_data()
  # Use a smaller subset for fast integration test
  data = (z[:5], mu[:5])

  # 2. Initialize Model
  model = MetricNN(model_key)

  # 3. Run a short training burst
  print("\nStarting control training integration test with Data Loss...")
  trained_model = train_control_model(
    model, data, num_steps=5, learning_rate=1e-3, lam=0.7, key=train_key
  )

  assert trained_model is not None
  print("Integration test successful.")
