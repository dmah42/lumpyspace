import dataclasses

from src.training.scheduler import AdaptivePenaltyState


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
  ) -> bool:
    """
    Updates the internal scheduler and steps the lambda multiplier.
    Returns True if the penalty weight was bumped.
    """
    bumped = self.scheduler.update(current_violation, step, check_interval)
    self.lambda_val += self.scheduler.w_penalty * current_violation
    return bumped

  def get_arrays(self) -> tuple[float, float]:
    """
    Returns the (lambda, w_penalty) as floats for JIT-compatible arrays.
    """
    return float(self.lambda_val), float(self.scheduler.w_penalty)

  def to_dict(self) -> dict:
    """Serializes state for checkpointing."""
    return {
      "lambda_val": self.lambda_val,
      "w_penalty": self.scheduler.w_penalty,
      "ema_violation": self.scheduler.ema_violation,
      "last_check_violation": self.scheduler.last_check_violation,
    }

  @classmethod
  def from_dict(cls, data: dict) -> "ConstraintManager":
    """Deserializes state from a checkpoint dictionary."""
    scheduler = AdaptivePenaltyState(
      w_penalty=data["w_penalty"],
      ema_violation=data["ema_violation"],
      last_check_violation=data["last_check_violation"],
    )
    return cls(lambda_val=data["lambda_val"], scheduler=scheduler)
