import jax.numpy as jnp

from src.training.scheduler import (
  AdaptivePenaltyState,
  create_geometric_sgdr_schedule,
)


def test_adaptive_penalty_state_bumps_when_stalled():
  state = AdaptivePenaltyState(w_penalty=1.0)

  # Step 0
  bumped = state.update(10.0, step=0, check_interval=5)
  assert not bumped
  assert state.ema_violation == 10.0
  assert state.last_check_violation == float("inf")

  # Step 1-4: Stall at 10.0
  for i in range(1, 5):
    bumped = state.update(10.0, step=i, check_interval=5)
    assert not bumped

  # Step 5: Check interval hit, first interval never bumps (last_check is inf)
  bumped = state.update(10.0, step=5, check_interval=5)
  assert not bumped
  assert state.w_penalty == 1.0
  assert state.last_check_violation == 10.0

  # Step 6-9: Stall at 10.0
  for i in range(6, 10):
    bumped = state.update(10.0, step=i, check_interval=5)
    assert not bumped

  # Step 10: Second interval, now last_check=10.0, so it bumps!
  bumped = state.update(10.0, step=10, check_interval=5)
  assert bumped
  assert state.w_penalty == 5.0
  assert state.last_check_violation == 10.0


def test_adaptive_penalty_state_does_not_bump_when_improving():
  state = AdaptivePenaltyState(w_penalty=1.0)
  state.update(10.0, step=0, check_interval=5)

  # Dramatically improve violation to 1.0
  for i in range(1, 5):
    state.update(1.0, step=i, check_interval=5)

  bumped = state.update(1.0, step=5, check_interval=5)
  assert not bumped
  assert state.w_penalty == 1.0


def test_geometric_sgdr_schedule():
  schedule = create_geometric_sgdr_schedule(
    learning_rate=1e-4,
    peak_learning_rate=1e-3,
    kick_period_0=100,
    kick_period_mult=2.0,
  )

  # Cycle 0: Length 100 (steps 0 to 99)
  # Decay length = 20
  # At step 0, progress = 0.0, cosine_val = 1.0 -> lr = 1e-3
  assert jnp.isclose(schedule(0), 1e-3)

  # At step 10, progress = 0.5, cosine_val = 0.5 -> lr = 5.5e-4
  assert jnp.isclose(schedule(10), 1e-4 + (1e-3 - 1e-4) * 0.5)

  # At step 20, progress = 1.0, cosine_val = 0.0 -> lr = 1e-4
  assert jnp.isclose(schedule(20), 1e-4)

  # At step 99, progress = 1.0, cosine_val = 0.0 -> lr = 1e-4
  assert jnp.isclose(schedule(99), 1e-4)

  # Cycle 1: Length 200 (steps 100 to 299)
  # Decay length = 40
  # At step 100 (start of new cycle), progress = 0.0 -> lr = 1e-3
  assert jnp.isclose(schedule(100), 1e-3)

  # At step 140, progress = 1.0 -> lr = 1e-4
  assert jnp.isclose(schedule(140), 1e-4)

  # Cycle 2: Length 400 (steps 300 to 699)
  # At step 300 (start of new cycle), progress = 0.0 -> lr = 1e-3
  assert jnp.isclose(schedule(300), 1e-3)
