"""
Data loaders for the PINN.
"""

from typing import Tuple

import jax.numpy as jnp
import pandas as pd


def load_mock_data(
  file_path: str = "data/mock_flrw.csv",
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Loads mock FLRW data for calibration."""
  df = pd.read_csv(file_path)
  z = jnp.array(df["z"].values)
  mu = jnp.array(df["mu"].values)
  # Default error to 1.0 for mock data
  return z, mu, jnp.ones_like(z)


def load_pantheon_plus(
  file_path: str = "data/pantheon_plus.dat",
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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


def normalize_coordinates(z: jnp.ndarray) -> jnp.ndarray:
  """
  Normalizes redshift to training coordinate t.
  Scale: z=0 -> t=1.0, z=1.0 -> t=0.0
  Equation: t = 1.0 - z
  """
  return 1.0 - z
