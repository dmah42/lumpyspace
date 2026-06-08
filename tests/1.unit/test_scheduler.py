from src.training.scheduler import AdaptivePenaltyState


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
  assert state.w_penalty == 10.0
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
