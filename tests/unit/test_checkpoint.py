import dataclasses
import json
import os

import pytest

from src.training.checkpoint import (
  TrainingState,
  load_best_feasible,
  load_meta,
  save_meta,
)
from src.training.constraints import ConstraintState


def _state(step: int, best_feasible_sn: float) -> TrainingState:
  constraint_state = ConstraintState(
    lambda_val=0.1,
    w_penalty=2.0,
    ema_violation=0.05,
    last_check_violation=0.06,
  )
  return TrainingState(
    step=step,
    best_feasible_sn=best_feasible_sn,
    l_wec=constraint_state,
    l_expand=constraint_state,
    l_shear=constraint_state,
    l_spatial=constraint_state,
  )


def test_save_and_load_meta(tmp_path):
  meta_path = str(tmp_path / "model.eqx.meta")
  state = _state(step=100, best_feasible_sn=0.5)

  save_meta(meta_path, state)

  assert os.path.exists(meta_path)

  loaded_state = load_meta(meta_path)
  assert loaded_state == state


def test_load_meta_ignores_unknown_keys(tmp_path):
  # Meta files written before best_loss was removed still load.
  meta_path = str(tmp_path / "model.eqx.meta")
  state = _state(step=100, best_feasible_sn=0.5)
  data = dataclasses.asdict(state)
  data["best_loss"] = 5.5
  with open(meta_path, "w") as f:
    json.dump(data, f)

  assert load_meta(meta_path) == state


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
    json.dump({"step": 100, "best_feasible_sn": 0.5}, f)  # Missing other keys

  with pytest.raises(ValueError, match="Missing required keys"):
    load_meta(meta_path)


def test_load_best_feasible_reads_best_checkpoint_meta(tmp_path):
  meta_path = str(tmp_path / "model.eqx.meta")
  save_meta(meta_path, _state(step=3282, best_feasible_sn=0.4585))

  assert load_best_feasible(meta_path) == (3282, 0.4585)


def test_load_best_feasible_before_first_feasible_step(tmp_path):
  meta_path = str(tmp_path / "model.eqx.meta")

  assert load_best_feasible(meta_path) == (None, float("inf"))
