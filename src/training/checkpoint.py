import dataclasses
import json
import os

from src.training.constraints import ConstraintState
from src.training.selection import SCORE_TERMS


@dataclasses.dataclass
class ConstraintStates:
  """Augmented Lagrangian state for each constraint."""

  wec: ConstraintState
  expand: ConstraintState
  shear: ConstraintState
  spatial: ConstraintState


@dataclasses.dataclass
class TrainingState:
  """
  State of one checkpoint: what a run needs to resume from it, and the scores
  of the step it was taken at.
  """

  step: int
  l_sn: float
  score: float
  score_terms: dict[str, float]
  constraints: ConstraintStates


def save_meta(meta_path: str, state: TrainingState) -> None:
  """
  Serializes the training state to a JSON .meta sidecar file.
  """
  with open(meta_path, "w") as meta_f:
    json.dump(dataclasses.asdict(state), meta_f, indent=2)


def _require(data: dict, keys: set[str], what: str) -> None:
  missing = keys - set(data.keys())
  if missing:
    raise ValueError(f"Invalid .meta file. Missing {what} keys: {missing}")


def _parse_constraint(c_data: dict) -> ConstraintState:
  _require(
    c_data,
    {"lambda_val", "w_penalty", "ema_violation", "last_check_violation"},
    "constraint",
  )
  return ConstraintState(
    lambda_val=float(c_data["lambda_val"]),
    w_penalty=float(c_data["w_penalty"]),
    ema_violation=float(c_data["ema_violation"]),
    last_check_violation=float(c_data["last_check_violation"]),
  )


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

  _require(
    data, {"step", "l_sn", "score", "score_terms", "constraints"}, "required"
  )
  _require(data["score_terms"], set(SCORE_TERMS), "score term")
  _require(
    data["constraints"], {"wec", "expand", "shear", "spatial"}, "constraint"
  )

  constraints = data["constraints"]
  return TrainingState(
    step=int(data["step"]),
    l_sn=float(data["l_sn"]),
    score=float(data["score"]),
    score_terms={k: float(data["score_terms"][k]) for k in SCORE_TERMS},
    constraints=ConstraintStates(
      wec=_parse_constraint(constraints["wec"]),
      expand=_parse_constraint(constraints["expand"]),
      shear=_parse_constraint(constraints["shear"]),
      spatial=_parse_constraint(constraints["spatial"]),
    ),
  )


def load_best(meta_path: str) -> tuple[int | None, float]:
  """
  Returns (step, score) from the best checkpoint's .meta file, or (None, inf)
  when no best checkpoint has been saved yet.
  """
  if not os.path.exists(meta_path):
    return None, float("inf")
  state = load_meta(meta_path)
  return state.step, state.score
