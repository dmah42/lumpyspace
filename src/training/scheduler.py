"""
Schedulers for dynamically updating hyperparameter penalties during training.
"""

import dataclasses

TAU = 0.9
GAMMA = 10.0
MAX_W = 1e5


@dataclasses.dataclass
class AdaptivePenaltyState:
  """
  Manages state and update logic for an Augmented Lagrangian penalty weight.
  It tracks an Exponential Moving Average (EMA) of the constraint violation,
  and scales up the penalty weight if the violation stalls.
  """

  w_penalty: float = 1.0
  ema_violation: float = 1.0
  last_check_violation: float = float("inf")

  def update(
    self,
    current_violation: float,
    step: int,
    check_interval: int = 500,
  ) -> bool:
    """
    Updates the internal state with the latest violation.
    Returns True if the penalty weight was bumped.
    """
    self._update_ema(current_violation, step)

    bumped = False
    if step > 0 and step % check_interval == 0:
      if self.ema_violation > TAU * self.last_check_violation:
        self.w_penalty = min(self.w_penalty * GAMMA, MAX_W)
        bumped = True
      self.last_check_violation = self.ema_violation

    return bumped

  def _update_ema(self, current_violation: float, step: int) -> None:
    """
    Updates the Exponential Moving Average (EMA) of the violation.
    """
    if step == 0:
      self.ema_violation = current_violation
    else:
      self.ema_violation = 0.9 * self.ema_violation + 0.1 * current_violation
