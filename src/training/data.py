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


def load_pantheon_plus(file_path="data/pantheon_plus.dat"):
  """
  Loads the Pantheon+ supernova dataset.
  Columns: zCMB, MU_SH0ES, MU_SH0ES_ERR_DIAG
  """
  # Space-separated file
  df = pd.read_csv(file_path, sep=" ")

  # Extract relevant columns
  z = jnp.array(df["zCMB"].values)
  mu = jnp.array(df["MU_SH0ES"].values)
  err = jnp.array(df["MU_SH0ES_ERR_DIAG"].values)

  # Filter out negative redshifts or invalid data if any
  mask = z > 0
  return z[mask], mu[mask], err[mask]


def normalize_coordinates(z, z_max=2.5):
  """
  Normalizes redshift to training coordinate t in [-1, 1].
  Following Task 4.1: Map z in [0, 2] to t in [-1, 1].
  Actually, we use z_max for safety.
  t = 1 - 2 * (z / z_max)
  So z=0 -> t=1, z=z_max -> t=-1.
  """
  t = 1.0 - 2.0 * (z / z_max)
  return t
