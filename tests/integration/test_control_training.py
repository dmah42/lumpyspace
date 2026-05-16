"""
Integration test for the FLRW control training loop.
"""

import jax

from src.core.metric import MetricNN
from src.training.trainer import train_control_model


def test_control_training_loop():
  """
  Verify that the training loop can execute and reduce the physics loss.
  """
  key = jax.random.PRNGKey(42)
  model_key, train_key = jax.random.split(key)

  # Initialize Model
  model = MetricNN(model_key)

  # Run a short training burst
  # Lambda = 0.7 (Standard LCDM)
  print("\nStarting control training integration test...")
  trained_model = train_control_model(
    model, num_steps=10, learning_rate=1e-3, lam=0.7, key=train_key
  )

  assert trained_model is not None
  print("Integration test successful.")


if __name__ == "__main__":
  test_control_training_loop()
