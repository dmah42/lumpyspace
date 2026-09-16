import jax.numpy as jnp

from src.training.constraints import ConstraintManager
from src.training.scheduler import (
  AdaptivePenaltyState,
  PenaltyAction,
  create_geometric_sgdr_schedule,
)


def test_adaptive_penalty_state_bumps_when_stalled():
  state = AdaptivePenaltyState(w_penalty=1.0)

  # Step 0
  action = state.update(10.0, step=0, check_interval=5)
  assert action == PenaltyAction.STEADY
  assert state.ema_violation == 10.0
  assert state.last_check_violation == float("inf")

  # Step 1-4: Stall at 10.0
  for i in range(1, 5):
    action = state.update(10.0, step=i, check_interval=5)
    assert action == PenaltyAction.STEADY

  # Step 5: Check interval hit, first interval never bumps (last_check is inf)
  action = state.update(10.0, step=5, check_interval=5)
  assert action == PenaltyAction.STEADY
  assert state.w_penalty == 1.0
  assert state.last_check_violation == 10.0

  # Step 6-9: Stall at 10.0
  for i in range(6, 10):
    action = state.update(10.0, step=i, check_interval=5)
    assert action == PenaltyAction.STEADY

  # Step 10: Second interval, now last_check=10.0, so it bumps!
  action = state.update(10.0, step=10, check_interval=5)
  assert action == PenaltyAction.BUMPED
  assert state.w_penalty == 5.0
  assert state.last_check_violation == 10.0


def test_adaptive_penalty_state_does_not_bump_when_improving():
  state = AdaptivePenaltyState(w_penalty=1.0)
  state.update(10.0, step=0, check_interval=5)

  # Dramatically improve violation to 1.0
  for i in range(1, 5):
    state.update(1.0, step=i, check_interval=5)

  action = state.update(1.0, step=5, check_interval=5)
  assert action == PenaltyAction.STEADY
  assert state.w_penalty == 1.0


def test_adaptive_penalty_state_decays_when_well_satisfied():
  # Start with an elevated penalty weight
  state = AdaptivePenaltyState(w_penalty=10.0)

  # Step 0 to 20: Establish baseline at violation ~10.0
  for i in range(21):
    state.update(10.0, step=i, check_interval=20)
  assert state.last_check_violation == 10.0
  assert state.w_penalty == 10.0

  # Steps 21-40: Violation drops dramatically to 0.1 (< 0.5 * 10.0)
  for i in range(21, 40):
    state.update(0.1, step=i, check_interval=20)

  action = state.update(0.1, step=40, check_interval=20)
  assert action == PenaltyAction.DECAYED
  assert state.w_penalty == 5.0  # 10.0 / GAMMA_DOWN (2.0)


def test_adaptive_penalty_state_neutral_zone():
  state = AdaptivePenaltyState(w_penalty=10.0)

  # Step 0 to 20: Baseline at 10.0
  for i in range(21):
    state.update(10.0, step=i, check_interval=20)
  assert state.last_check_violation == 10.0

  # Steps 21-40: Moderate improvement to 8.0 (80% of last check).
  # EMA reaches ~8.24, which is < 9.5 (no bump) and > 5.0 (no decay).
  for i in range(21, 40):
    state.update(8.0, step=i, check_interval=20)

  action = state.update(8.0, step=40, check_interval=20)
  assert action == PenaltyAction.STEADY
  assert state.w_penalty == 10.0


def test_adaptive_penalty_state_respects_min_w():
  # Start at minimum w_penalty = 1.0
  state = AdaptivePenaltyState(w_penalty=1.0)

  # Baseline at 10.0
  for i in range(21):
    state.update(10.0, step=i, check_interval=20)

  # Violation drops to nearly 0
  for i in range(21, 40):
    state.update(0.01, step=i, check_interval=20)

  action = state.update(0.01, step=40, check_interval=20)
  assert action == PenaltyAction.DECAYED
  assert state.w_penalty == 1.0  # Clamped at MIN_W


def test_geometric_sgdr_schedule():
  schedule = create_geometric_sgdr_schedule(
    learning_rate=1e-4,
    peak_learning_rate=1e-3,
    kick_period_0=100,
    kick_period_mult=2.0,
    max_impulse_steps=100,
  )

  # Cycle 0: Length 100 (steps 0 to 99)
  # Decay length = min(20, 100) = 20
  # At step 0, progress = 0.0, cosine_val = 1.0 -> lr = 1e-3
  assert jnp.isclose(schedule(0), 1e-3)

  # At step 10, progress = 0.5, cosine_val = 0.5 -> lr = 5.5e-4
  assert jnp.isclose(schedule(10), 1e-4 + (1e-3 - 1e-4) * 0.5)

  # At step 20, progress = 1.0, cosine_val = 0.0 -> lr = 1e-4
  assert jnp.isclose(schedule(20), 1e-4)

  # At step 99, progress = 1.0, cosine_val = 0.0 -> lr = 1e-4
  assert jnp.isclose(schedule(99), 1e-4)

  # Cycle 1: Length 200 (steps 100 to 299)
  # Decay length = min(40, 100) = 40
  # At step 100 (start of new cycle), progress = 0.0 -> lr = 1e-3
  assert jnp.isclose(schedule(100), 1e-3)

  # At step 140, progress = 1.0 -> lr = 1e-4
  assert jnp.isclose(schedule(140), 1e-4)

  # Cycle 2: Length 400 (steps 300 to 699)
  # Decay length = min(80, 100) = 80
  # At step 300 (start of new cycle), progress = 0.0 -> lr = 1e-3
  assert jnp.isclose(schedule(300), 1e-3)


def test_geometric_sgdr_schedule_impulse_cap():
  # Cap impulse at 30 steps
  schedule = create_geometric_sgdr_schedule(
    learning_rate=1e-4,
    peak_learning_rate=1e-3,
    kick_period_0=100,
    kick_period_mult=2.0,
    max_impulse_steps=30,
  )

  # Cycle 1: Length 200 (steps 100 to 299)
  # 200 / 5 = 40, but capped at 30!
  # At step 100: start of kick
  assert jnp.isclose(schedule(100), 1e-3)

  # At step 115: midpoint of 30-step decay
  assert jnp.isclose(schedule(115), 1e-4 + (1e-3 - 1e-4) * 0.5)

  # At step 130: decay complete
  assert jnp.isclose(schedule(130), 1e-4)

  # At step 140: stays flat at baseline (would still be decaying without cap)
  assert jnp.isclose(schedule(140), 1e-4)


def test_constraint_manager_action_and_lambda():
  manager = ConstraintManager(lambda_val=0.0)

  # Step 0
  action = manager.update(10.0, step=0, check_interval=5)
  assert action == PenaltyAction.STEADY
  assert manager.lambda_val == 0.0

  # Steps 1 to 4: Stall at 10.0
  for i in range(1, 5):
    action = manager.update(10.0, step=i, check_interval=5)
    assert action == PenaltyAction.STEADY

  # Step 5: Check interval hit (first interval establishes baseline)
  action = manager.update(10.0, step=5, check_interval=5)
  assert action == PenaltyAction.STEADY
  # lambda accumulated: 0.0 + 1.0 * 10.0 = 10.0
  assert manager.lambda_val == 10.0
  assert manager.scheduler.w_penalty == 1.0

  # Steps 6 to 9: Still stalled
  for i in range(6, 10):
    manager.update(10.0, step=i, check_interval=5)

  # Step 10: Stalled violation triggers BUMPED
  action = manager.update(10.0, step=10, check_interval=5)
  assert action == PenaltyAction.BUMPED
  assert manager.scheduler.w_penalty == 5.0
  # lambda accumulated: 10.0 + 5.0 * 10.0 = 60.0
  assert manager.lambda_val == 60.0

  # Serialization round-trip
  data = manager.to_dict()
  restored = ConstraintManager.from_dict(data)
  assert restored.lambda_val == 60.0
  assert restored.scheduler.w_penalty == 5.0
  assert restored.scheduler.ema_violation == manager.scheduler.ema_violation
  assert (
    restored.scheduler.last_check_violation
    == manager.scheduler.last_check_violation
  )
