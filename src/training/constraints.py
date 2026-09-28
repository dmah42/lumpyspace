import dataclasses

from src.training.scheduler import AdaptivePenaltyState, PenaltyAction


@dataclasses.dataclass
class ConstraintState:
  """Serialized form of a ConstraintManager, as stored in a .meta file."""

  lambda_val: float
  w_penalty: float
  ema_violation: float
  last_check_violation: float


@dataclasses.dataclass
class ConstraintManager:
  """
  Unifies the tracking of an Augmented Lagrangian multiplier (lambda)
  and its corresponding Adaptive Penalty scaler (w_penalty).
  """

  lambda_val: float = 0.0
  scheduler: AdaptivePenaltyState = dataclasses.field(
    default_factory=AdaptivePenaltyState
  )

  def update(
    self, current_violation: float, step: int, check_interval: int = 500
  ) -> PenaltyAction:
    """
    Updates the internal scheduler and steps the lambda multiplier.
    Returns PenaltyAction indicating the scheduler action.
    """
    action = self.scheduler.update(current_violation, step, check_interval)
    if step > 0 and step % check_interval == 0:
      self.lambda_val += self.scheduler.w_penalty * current_violation
    return action

  def get_arrays(self) -> tuple[float, float]:
    """
    Returns the (lambda, w_penalty) as floats for JIT-compatible arrays.
    """
    return float(self.lambda_val), float(self.scheduler.w_penalty)

  def to_state(self) -> ConstraintState:
    """Serializes state for checkpointing."""
    return ConstraintState(
      lambda_val=self.lambda_val,
      w_penalty=self.scheduler.w_penalty,
      ema_violation=self.scheduler.ema_violation,
      last_check_violation=self.scheduler.last_check_violation,
    )

  @classmethod
  def from_state(cls, state: ConstraintState) -> "ConstraintManager":
    """Deserializes state from a checkpoint record."""
    scheduler = AdaptivePenaltyState(
      w_penalty=state.w_penalty,
      ema_violation=state.ema_violation,
      last_check_violation=state.last_check_violation,
    )
    return cls(lambda_val=state.lambda_val, scheduler=scheduler)
