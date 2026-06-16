"""
Schedulers for dynamically updating hyperparameter penalties during training.
"""

import dataclasses
from collections.abc import Callable

import jax.numpy as jnp

# The fraction of the previous violation that the current violation must drop
# below to avoid a penalty bump. E.g., 0.95 means the violation must improve by
# at least 5% every check interval.
TAU = 0.95

# The multiplicative factor applied to the penalty weight when the violation
# stalls.
GAMMA = 5.0

# MAX_W: The absolute maximum value the penalty weight is allowed to reach.
MAX_W = 1e6


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


def create_geometric_sgdr_schedule(
  learning_rate: float,
  peak_learning_rate: float,
  kick_period_0: int,
  kick_period_mult: float,
) -> Callable:
  """
  Creates a periodic learning rate schedule to continuously kick the model out
  of local minima. Decays rapidly, then stays flat at baseline.

  Uses geometric progression for the cycle lengths: T_i = T_0 * M^i
  """

  def schedule(step):
    t_0 = float(kick_period_0)
    m = float(kick_period_mult)

    n = jnp.floor(jnp.log(1.0 + step * (m - 1.0) / t_0) / jnp.log(m))
    cycle_start = t_0 * (m**n - 1.0) / (m - 1.0)
    cycle_length = t_0 * (m**n)
    cycle_step = step - cycle_start

    decay_length = cycle_length / 5.0  # Decay over the first 20% of the cycle

    progress = jnp.minimum(cycle_step / decay_length, 1.0)
    cosine_val = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))

    return learning_rate + (peak_learning_rate - learning_rate) * cosine_val

  return schedule
