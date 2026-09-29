"""
Data loaders for the PINN.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM


def load_mock_data(
  num_samples: int = 100,
  z_max: float = 2.0,
  h0: float = 70.0,
  om0: float = 0.3,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """
  Generates synthetic FLRW supernova data directly in memory.

  Sources are spread evenly over the sky via a Fibonacci sphere. The mock
  cosmology is isotropic, so the true distance is direction-independent by
  construction - which makes this a stress test of the directional
  ray-tracer: every direction must return the same dL(z).
  """
  cosmo = FlatLambdaCDM(H0=h0, Om0=om0)
  z = np.linspace(0.01, z_max, num_samples)
  mu = cosmo.distmod(z).value
  z_jnp = jnp.array(z)
  return (
    z_jnp,
    jnp.array(mu),
    jnp.ones_like(z_jnp),
    fibonacci_sphere(num_samples),
  )


def fibonacci_sphere(n: int) -> jnp.ndarray:
  """Deterministic, near-uniform unit vectors over the sphere."""
  i = np.arange(n)
  phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
  theta = np.pi * (1.0 + 5.0**0.5) * i
  return jnp.array(
    np.stack(
      [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)],
      axis=-1,
    )
  )


def radec_to_unit_vector(
  ra_deg: np.ndarray, dec_deg: np.ndarray
) -> jnp.ndarray:
  """Converts equatorial sky coordinates to Cartesian unit vectors."""
  ra = np.deg2rad(ra_deg)
  dec = np.deg2rad(dec_deg)
  return jnp.array(
    np.stack(
      [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)],
      axis=-1,
    )
  )


def load_pantheon_plus(
  file_path: str = "data/pantheon_plus.dat",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """
  Loads the Pantheon+ supernova dataset.
  Columns: zCMB, MU_SH0ES, MU_SH0ES_ERR_DIAG, RA, DEC

  Returns (z, mu, err, direction) where direction is a unit 3-vector per
  supernova. Without it every source would be traced along the same fixed
  ray, leaving the model's anisotropic structure unconstrained in every
  direction but one.
  """
  # Space-separated file
  df = pd.read_csv(file_path, sep=" ")

  # Extract relevant columns
  z = jnp.array(df["zCMB"].values)
  mu = jnp.array(df["MU_SH0ES"].values)
  err = jnp.array(df["MU_SH0ES_ERR_DIAG"].values)
  direction = radec_to_unit_vector(df["RA"].values, df["DEC"].values)

  # Filter out negative redshifts or invalid data if any
  mask = z > 0
  return z[mask], mu[mask], err[mask], direction[mask]


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

  cov_array = jnp.stack(covs)

  # Compute inverses
  cov_invs = jnp.linalg.inv(cov_array)

  return z, dm_obs, dh_obs, cov_invs
