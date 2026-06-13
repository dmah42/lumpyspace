import json
import os

import pytest

from src.training.checkpoint import (
  ConstraintState,
  TrainingState,
  load_meta,
  save_meta,
)
from src.training.loss import (
  METRIC_EXPAND,
  METRIC_SHEAR,
  METRIC_SPATIAL,
  METRIC_WEC,
)


def test_save_and_load_meta(tmp_path):
  meta_path = str(tmp_path / "model.eqx.meta")
  constraint_state: ConstraintState = {
    "lambda_val": 0.1,
    "w_penalty": 2.0,
    "ema_violation": 0.05,
    "last_check_violation": 0.06,
  }

  state: TrainingState = {
    "step": 100,
    "best_loss": 0.001,
    METRIC_WEC: constraint_state,
    METRIC_EXPAND: constraint_state,
    METRIC_SHEAR: constraint_state,
    METRIC_SPATIAL: constraint_state,
  }

  save_meta(meta_path, state)

  assert os.path.exists(meta_path)

  loaded_state = load_meta(meta_path)
  assert loaded_state == state


def test_load_meta_missing_file():
  with pytest.raises(
    FileNotFoundError, match="Cannot resume: Meta file missing"
  ):
    load_meta("/fake/path/model.eqx.meta")


def test_load_meta_corrupted_json(tmp_path):
  meta_path = str(tmp_path / "corrupt.meta")
  with open(meta_path, "w") as f:
    f.write("{ invalid json")

  with pytest.raises(ValueError, match="Incompatible .meta JSON file"):
    load_meta(meta_path)


def test_load_meta_missing_keys(tmp_path):
  meta_path = str(tmp_path / "missing.meta")
  with open(meta_path, "w") as f:
    json.dump({"step": 100, "best_loss": 0.001}, f)  # Missing other keys

  with pytest.raises(ValueError, match="Missing required keys"):
    load_meta(meta_path)
