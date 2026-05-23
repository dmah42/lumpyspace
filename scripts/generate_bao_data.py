"""
Generates the standard SDSS DR12 consensus Baryon Acoustic Oscillation (BAO)
dataset.
Reference: Alam et al. 2017 (BOSS DR12).
"""

import os

import pandas as pd


def generate_bao_consensus():
  """
  Creates a CSV containing the standard 3-bin SDSS BAO consensus data.
  The observables are D_M / r_d and D_H / r_d.
  """
  print("Generating SDSS DR12 BAO Consensus dataset...")

  # Data from Alam et al. 2017
  # Redshifts (z_eff)
  z = [0.38, 0.51, 0.61]

  # Transverse Comoving Distance / Sound Horizon (D_M / r_d)
  dm_over_rd = [15.12, 19.78, 22.92]
  dm_err = [0.22, 0.27, 0.36]

  # Radial Hubble Distance / Sound Horizon (D_H / r_d)
  dh_over_rd = [25.00, 22.33, 20.75]
  dh_err = [0.76, 0.58, 0.73]

  # Correlation coefficients between D_M and D_H
  r_corr = [0.43, 0.45, 0.53]

  data = {
    "z": z,
    "DM_over_rd": dm_over_rd,
    "DM_err": dm_err,
    "DH_over_rd": dh_over_rd,
    "DH_err": dh_err,
    "r_corr": r_corr,
  }

  df = pd.DataFrame(data)

  # Ensure data directory exists
  os.makedirs("data", exist_ok=True)
  output_path = "data/bao_consensus.csv"

  df.to_csv(output_path, index=False)
  print(f"Successfully wrote 3 BAO data points to {output_path}")


if __name__ == "__main__":
  generate_bao_consensus()
