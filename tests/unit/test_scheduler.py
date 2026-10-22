import jax.numpy as jnp

from src.training.constraints import ConstraintManager
from src.training.scheduler import (
  GAMMA,
  GAMMA_DOWN,
  MAX_W,
  START_W,
  AdaptivePenaltyState,
  PenaltyAction,
  create_geometric_sgdr_schedule,
  full_cycle_without_improvement,
  sgdr_cycle_index,
)


def test_adaptive_penalty_state_starts_at_start_w():
  assert AdaptivePenaltyState().w_penalty == START_W
  assert AdaptivePenaltyState().w_min == START_W


def test_constraint_manager_starting_at_sets_weight_and_floor():
  manager = ConstraintManager.starting_at(10.0)
  assert manager.scheduler.w_penalty == 10.0
  assert manager.scheduler.w_min == 10.0
  assert manager.lambda_val == 0.0


def test_adaptive_penalty_state_bumps_when_stalled():
  state = AdaptivePenaltyState()

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
  assert state.w_penalty == START_W
  assert state.last_check_violation == 10.0

  # Step 6-9: Stall at 10.0
  for i in range(6, 10):
    action = state.update(10.0, step=i, check_interval=5)
    assert action == PenaltyAction.STEADY

  # Step 10: Second interval, now last_check=10.0, so it bumps!
  action = state.update(10.0, step=10, check_interval=5)
  assert action == PenaltyAction.BUMPED
  assert state.w_penalty == START_W * GAMMA
  assert state.last_check_violation == 10.0


def test_adaptive_penalty_state_does_not_bump_when_improving():
  state = AdaptivePenaltyState()
  state.update(10.0, step=0, check_interval=5)

  # Dramatically improve violation to 1.0
  for i in range(1, 5):
    state.update(1.0, step=i, check_interval=5)

  action = state.update(1.0, step=5, check_interval=5)
  assert action == PenaltyAction.STEADY
  assert state.w_penalty == START_W


def test_adaptive_penalty_state_decays_when_well_satisfied():
  # Start with an elevated penalty weight
  elevated = START_W * 10
  state = AdaptivePenaltyState(w_penalty=elevated)

  # Step 0 to 20: Establish baseline at violation ~10.0
  for i in range(21):
    state.update(10.0, step=i, check_interval=20)
  assert state.last_check_violation == 10.0
  assert state.w_penalty == elevated

  # Steps 21-40: Violation drops dramatically to 0.1 (< 0.5 * 10.0)
  for i in range(21, 40):
    state.update(0.1, step=i, check_interval=20)

  action = state.update(0.1, step=40, check_interval=20)
  assert action == PenaltyAction.DECAYED
  assert state.w_penalty == elevated / GAMMA_DOWN


def test_adaptive_penalty_state_neutral_zone():
  state = AdaptivePenaltyState()

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
  assert state.w_penalty == START_W


def test_adaptive_penalty_state_respects_min_w():
  # Decay returns the weight to its starting pressure and never below it.
  state = AdaptivePenaltyState(w_penalty=10.0, w_min=10.0)

  # Baseline at 10.0
  for i in range(21):
    state.update(10.0, step=i, check_interval=20)

  # Violation drops to nearly 0
  for i in range(21, 40):
    state.update(0.01, step=i, check_interval=20)

  action = state.update(0.01, step=40, check_interval=20)
  assert action == PenaltyAction.DECAYED
  assert state.w_penalty == 10.0


def test_adaptive_penalty_state_respects_max_w():
  # Three stalled checks take START_W past MAX_W, where it is held.
  state = AdaptivePenaltyState()
  actions = [state.update(10.0, step=i, check_interval=5) for i in range(0, 26)]

  assert actions.count(PenaltyAction.BUMPED) == 4
  assert START_W * GAMMA**2 < MAX_W < START_W * GAMMA**3
  assert state.w_penalty == MAX_W


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


def test_sgdr_cycle_index_integer_boundaries():
  # T_0 = 100, M = 2: cycles start at 0, 100, 300, 700.
  for step, expected in [
    (0, 0),
    (99, 0),
    (100, 1),
    (299, 1),
    (300, 2),
    (699, 2),
    (700, 3),
  ]:
    assert sgdr_cycle_index(step, 100, 2.0) == expected


def test_sgdr_cycle_index_fractional_boundaries():
  # T_0 = 500, M = 1.5: cycles start at 0, 500, 1250, 2375, 4062.5, 6593.75.
  for step, expected in [
    (499, 0),
    (500, 1),
    (1249, 1),
    (1250, 2),
    (2374, 2),
    (2375, 3),
    (4062, 3),
    (4063, 4),
    (6593, 4),
    (6594, 5),
  ]:
    assert sgdr_cycle_index(step, 500, 1.5) == expected


def test_sgdr_cycle_index_matches_schedule_restarts():
  # The learning rate rises only at a restart, so the steps where the
  # schedule increases must be exactly the steps where the cycle index
  # increments. Uses the production kick parameters, whose cycle boundaries
  # are fractional from the fourth restart onward.
  schedule = create_geometric_sgdr_schedule(
    learning_rate=1e-4,
    peak_learning_rate=1e-3,
    kick_period_0=500,
    kick_period_mult=1.5,
    max_impulse_steps=100,
  )
  steps = jnp.arange(11000)
  lr = schedule(steps)
  schedule_restarts = {int(s) for s in steps[1:][lr[1:] > lr[:-1]]}
  index_restarts = {
    s
    for s in range(1, 11000)
    if sgdr_cycle_index(s, 500, 1.5) > sgdr_cycle_index(s - 1, 500, 1.5)
  }
  assert index_restarts == {500, 1250, 2375, 4063, 6594, 10391}
  assert schedule_restarts == index_restarts


def test_full_cycle_without_improvement():
  # T_0 = 500, M = 1.5. Improvement at 3002 is in cycle 3 (2375-4062).
  # Cycle 4 (4063-6593) is the full cycle that must pass without improvement.
  assert not full_cycle_without_improvement(3002, 3002, 500, 1.5)
  assert not full_cycle_without_improvement(4062, 3002, 500, 1.5)
  assert not full_cycle_without_improvement(6593, 3002, 500, 1.5)
  assert full_cycle_without_improvement(6594, 3002, 500, 1.5)


def test_full_cycle_without_improvement_late_in_cycle():
  # An improvement just before a restart still earns the whole next cycle.
  assert not full_cycle_without_improvement(4062, 4062, 500, 1.5)
  assert not full_cycle_without_improvement(6593, 4062, 500, 1.5)
  assert full_cycle_without_improvement(6594, 4062, 500, 1.5)


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
  # lambda accumulated: 0.0 + START_W * 10.0
  assert manager.lambda_val == START_W * 10.0
  assert manager.scheduler.w_penalty == START_W

  # Steps 6 to 9: Still stalled
  for i in range(6, 10):
    manager.update(10.0, step=i, check_interval=5)

  # Step 10: Stalled violation triggers BUMPED
  action = manager.update(10.0, step=10, check_interval=5)
  assert action == PenaltyAction.BUMPED
  bumped = START_W * GAMMA
  assert manager.scheduler.w_penalty == bumped
  # lambda accumulated: START_W * 10.0 + bumped * 10.0
  expected_lambda = START_W * 10.0 + bumped * 10.0
  assert manager.lambda_val == expected_lambda

  # Serialization round-trip
  restored = ConstraintManager.from_state(manager.to_state(), w_min=START_W)
  assert restored.lambda_val == expected_lambda
  assert restored.scheduler.w_penalty == bumped
  assert restored.scheduler.w_min == START_W
  assert restored.scheduler.ema_violation == manager.scheduler.ema_violation
  assert (
    restored.scheduler.last_check_violation
    == manager.scheduler.last_check_violation
  )
