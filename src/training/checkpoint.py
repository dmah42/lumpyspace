import dataclasses
import json
import os

from src.training.constraints import ConstraintState
from src.training.loss import (
  METRIC_EXPAND,
  METRIC_SHEAR,
  METRIC_SPATIAL,
  METRIC_WEC,
)


@dataclasses.dataclass
class TrainingState:
  """Training state required to resume a run."""

  step: int
  best_loss: float
  best_feasible_sn: float
  l_wec: ConstraintState
  l_expand: ConstraintState
  l_shear: ConstraintState
  l_spatial: ConstraintState


def save_meta(meta_path: str, state: TrainingState) -> None:
  """
  Serializes the training state to a JSON .meta sidecar file.
  """
  with open(meta_path, "w") as meta_f:
    json.dump(dataclasses.asdict(state), meta_f, indent=2)


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
    "best_feasible_sn",
    METRIC_WEC,
    METRIC_EXPAND,
    METRIC_SHEAR,
    METRIC_SPATIAL,
  }

  missing_keys = required_keys - set(data.keys())
  if missing_keys:
    raise ValueError(
      f"Invalid .meta file. Missing required keys: {missing_keys}"
    )

  def parse_constraint(c_data: dict) -> ConstraintState:
    req_c_keys = {
      "lambda_val",
      "w_penalty",
      "ema_violation",
      "last_check_violation",
    }
    missing_c_keys = req_c_keys - set(c_data.keys())
    if missing_c_keys:
      raise ValueError(f"Missing constraint keys: {missing_c_keys}")
    return ConstraintState(
      lambda_val=float(c_data["lambda_val"]),
      w_penalty=float(c_data["w_penalty"]),
      ema_violation=float(c_data["ema_violation"]),
      last_check_violation=float(c_data["last_check_violation"]),
    )

  return TrainingState(
    step=int(data["step"]),
    best_loss=float(data["best_loss"]),
    best_feasible_sn=float(data["best_feasible_sn"]),
    l_wec=parse_constraint(data[METRIC_WEC]),
    l_expand=parse_constraint(data[METRIC_EXPAND]),
    l_shear=parse_constraint(data[METRIC_SHEAR]),
    l_spatial=parse_constraint(data[METRIC_SPATIAL]),
  )
