"""
Data loaders for the PINN.
"""

import jax.numpy as jnp
import pandas as pd


def load_mock_data(file_path="data/mock_flrw.csv"):
  """Loads mock FLRW data for calibration."""
  df = pd.read_csv(file_path)
  z = jnp.array(df["z"].values)
  mu = jnp.array(df["mu"].values)
  return z, mu
