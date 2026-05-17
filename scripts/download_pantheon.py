import os
import urllib.request


def download_pantheon_plus():
  url = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat"
  output_dir = "data"
  output_path = os.path.join(output_dir, "pantheon_plus.dat")

  print(f"Downloading Pantheon+ dataset from: {url}")

  os.makedirs(output_dir, exist_ok=True)

  try:
    with urllib.request.urlopen(url) as response, open(output_path, "wb") as f:
      f.write(response.read())
    print(f"Data saved to: {output_path}")
  except Exception as e:
    print(f"Error downloading data: {e}")


if __name__ == "__main__":
  download_pantheon_plus()
