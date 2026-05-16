"""
Generate mock FLRW luminosity distance data for PINN calibration.
Uses Astropy's FlatLambdaCDM cosmology.
"""

import os

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM


def generate_mock_data():
  # Parameters from Technical Design
  h0 = 70.0
  om0 = 0.3
  z_max = 2.0
  num_samples = 100

  print(f"Generating mock FLRW data: H0={h0}, Om0={om0}, z_max={z_max}")

  # Initialize Cosmology
  cosmo = FlatLambdaCDM(H0=h0, Om0=om0)

  # Generate redshifts
  z = np.linspace(0.01, z_max, num_samples)

  # Calculate Luminosity Distance in Mpc
  dl = cosmo.luminosity_distance(z).value

  # Calculate Distance Modulus
  mu = cosmo.distmod(z).value

  # Create DataFrame
  df = pd.DataFrame({"z": z, "dL_mpc": dl, "mu": mu})

  # Ensure data directory exists
  os.makedirs("data", exist_ok=True)

  # Save to CSV
  output_path = "data/mock_flrw.csv"
  df.to_csv(output_path, index=False)
  print(f"Successfully saved mock data to {output_path}")


if __name__ == "__main__":
  generate_mock_data()
