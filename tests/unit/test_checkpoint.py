import dataclasses
import json
import os

import pytest

from src.training.checkpoint import (
  ConstraintStates,
  TrainingState,
  load_best,
  load_meta,
  save_meta,
)
from src.training.constraints import ConstraintState
from src.training.selection import SCORE_TERMS


def _state(step: int, score: float) -> TrainingState:
  constraint_state = ConstraintState(
    lambda_val=0.1,
    w_penalty=2.0,
    ema_violation=0.05,
    last_check_violation=0.06,
  )
  return TrainingState(
    step=step,
    l_sn=0.45,
    score=score,
    score_terms={name: float(i) for i, name in enumerate(SCORE_TERMS)},
    constraints=ConstraintStates(
      wec=constraint_state,
      expand=constraint_state,
      shear=constraint_state,
      spatial=constraint_state,
    ),
  )


def _write(path, data) -> str:
  with open(path, "w") as f:
    json.dump(data, f)
  return str(path)


def test_save_and_load_meta(tmp_path):
  meta_path = str(tmp_path / "model.eqx.meta")
  state = _state(step=100, score=12.5)

  save_meta(meta_path, state)

  assert os.path.exists(meta_path)
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
  meta_path = _write(tmp_path / "missing.meta", {"step": 100, "score": 1.0})

  with pytest.raises(ValueError, match="Missing required keys"):
    load_meta(meta_path)


def test_load_meta_missing_score_term(tmp_path):
  data = dataclasses.asdict(_state(step=100, score=12.5))
  del data["score_terms"][SCORE_TERMS[0]]
  meta_path = _write(tmp_path / "missing_term.meta", data)

  with pytest.raises(ValueError, match="Missing score term keys"):
    load_meta(meta_path)


def test_load_meta_missing_constraint(tmp_path):
  data = dataclasses.asdict(_state(step=100, score=12.5))
  del data["constraints"]["shear"]
  meta_path = _write(tmp_path / "missing_constraint.meta", data)

  with pytest.raises(ValueError, match="Missing constraint keys"):
    load_meta(meta_path)


def test_load_best_reads_best_checkpoint_meta(tmp_path):
  meta_path = str(tmp_path / "model.eqx.meta")
  save_meta(meta_path, _state(step=3282, score=7.25))

  assert load_best(meta_path) == (3282, 7.25)


def test_load_best_before_first_best_step(tmp_path):
  meta_path = str(tmp_path / "model.eqx.meta")

  assert load_best(meta_path) == (None, float("inf"))
