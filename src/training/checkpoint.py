import json
import os
from typing import TypedDict


class TrainingState(TypedDict):
  """Strictly typed dictionary for resuming training state."""

  step: int
  best_loss: float
  lambda_wec: float
  w_penalty: float
  ema_violation: float
  last_check_violation: float


def save_meta(meta_path: str, state: TrainingState) -> None:
  """
  Serializes the training state dictionary to a JSON .meta sidecar file.
  """
  with open(meta_path, "w") as meta_f:
    json.dump(state, meta_f, indent=2)


def load_meta(meta_path: str) -> TrainingState:
  """
  Reads the JSON .meta sidecar file and strictly validates the presence
  of all required keys for training resumption. No fallbacks.

  Raises:
    FileNotFoundError: If the .meta file does not exist.
    ValueError: If the JSON is invalid or missing required keys.
  """
  if not os.path.exists(meta_path):
    raise FileNotFoundError(f"Cannot resume: Meta file missing at {meta_path}.")

  try:
    with open(meta_path) as meta_f:
      data = json.load(meta_f)
  except json.JSONDecodeError as e:
    raise ValueError("Incompatible .meta JSON file") from e

  required_keys = {
    "step",
    "best_loss",
    "lambda_wec",
    "w_penalty",
    "ema_violation",
    "last_check_violation",
  }

  missing_keys = required_keys - set(data.keys())
  if missing_keys:
    raise ValueError(
      f"Invalid .meta file. Missing required keys: {missing_keys}"
    )

  return {
    "step": int(data["step"]),
    "best_loss": float(data["best_loss"]),
    "lambda_wec": float(data["lambda_wec"]),
    "w_penalty": float(data["w_penalty"]),
    "ema_violation": float(data["ema_violation"]),
    "last_check_violation": float(data["last_check_violation"]),
  }
