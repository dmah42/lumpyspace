"""
Data loaders for the PINN.
"""

import jax.numpy as jnp
import pandas as pd


def load_mock_data(
  file_path: str = "data/mock_flrw.csv",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Loads mock FLRW data for calibration."""
  df = pd.read_csv(file_path)
  z = jnp.array(df["z"].values)
  mu = jnp.array(df["mu"].values)
  # Default error to 1.0 for mock data
  return z, mu, jnp.ones_like(z)


def load_pantheon_plus(
  file_path: str = "data/pantheon_plus.dat",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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


def load_bao_data(
  file_path: str = "data/bao_consensus.csv",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """
  Loads the BAO consensus dataset.

  Returns:
    z: jnp.ndarray of effective redshifts for the BAO measurements.
    dm_obs: jnp.ndarray of observed Transverse Comoving Distances (D_M / r_d).
    dh_obs: jnp.ndarray of observed Radial Hubble Distances (D_H / r_d).
    cov_inv: jnp.ndarray of shape (N, 2, 2) containing the inverted covariance
             matrices for the (D_M, D_H) pairs at each redshift.
  """
  df = pd.read_csv(file_path)

  z = jnp.array(df["z"].values)
  dm_obs = jnp.array(df["DM_over_rd"].values)
  dm_err = jnp.array(df["DM_err"].values)
  dh_obs = jnp.array(df["DH_over_rd"].values)
  dh_err = jnp.array(df["DH_err"].values)
  r_corr = jnp.array(df["r_corr"].values)

  # Build covariance matrices for each data point
  # C = [[sigma_DM^2, r * sigma_DM * sigma_DH],
  #      [r * sigma_DM * sigma_DH, sigma_DH^2]]
  covs = []
  for i in range(len(z)):
    cov = jnp.array(
      [
        [dm_err[i] ** 2, r_corr[i] * dm_err[i] * dh_err[i]],
        [r_corr[i] * dm_err[i] * dh_err[i], dh_err[i] ** 2],
      ]
    )
    covs.append(cov)

  covs = jnp.stack(covs)

  # Compute inverses
  cov_invs = jnp.linalg.inv(covs)

  return z, dm_obs, dh_obs, cov_invs


def normalize_coordinates(z: jnp.ndarray) -> jnp.ndarray:
  """
  Normalizes redshift to training coordinate t.
  Scale: z=0 -> t=1.0, z=1.0 -> t=0.0
  Equation: t = 1.0 - z
  """
  a = 1 / (1 + z)
  return (a * 5.0) - 4.0
  # return 1.0 - z
