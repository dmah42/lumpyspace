"""
Schedulers for dynamically updating hyperparameter penalties during training.
"""

import dataclasses
import enum
from collections.abc import Callable

import jax.numpy as jnp

# The fraction of the previous violation that the current violation must drop
# below to avoid a penalty bump. E.g., 0.95 means the violation must improve by
# at least 5% every check interval.
TAU = 0.95

# The multiplicative factor applied to the penalty weight when the violation
# stalls.
GAMMA = 5.0

# The decay factor applied to the penalty weight when the constraint is
# well-satisfied.
GAMMA_DOWN = 2.0

# The constraint violation must drop below this fraction of the last check
# violation to trigger penalty relaxation.
DECAY_TAU = 0.5

# MIN_W: The absolute minimum value the penalty weight is allowed to reach.
MIN_W = 1.0

# MAX_W: The absolute maximum value the penalty weight is allowed to reach.
MAX_W = 1e6


class PenaltyAction(enum.Enum):
  """Possible outcomes of an adaptive penalty check."""

  STEADY = "steady"
  BUMPED = "bumped"
  DECAYED = "decayed"


@dataclasses.dataclass
class AdaptivePenaltyState:
  """
  Manages state and update logic for an Augmented Lagrangian penalty weight.
  It tracks an Exponential Moving Average (EMA) of the constraint violation,
  scales up the penalty weight if the violation stalls, and relaxes it when
  the constraint is well-satisfied.
  """

  w_penalty: float = 1.0
  ema_violation: float = 1.0
  last_check_violation: float = float("inf")

  def update(
    self,
    current_violation: float,
    step: int,
    check_interval: int = 500,
  ) -> PenaltyAction:
    """
    Updates the internal state with the latest violation.
    Returns PenaltyAction indicating whether the penalty weight was bumped,
    decayed, or stayed steady.
    """
    self._update_ema(current_violation, step)

    action = PenaltyAction.STEADY
    if step > 0 and step % check_interval == 0:
      if self.last_check_violation < float("inf"):
        if self.ema_violation > TAU * self.last_check_violation:
          self.w_penalty = min(self.w_penalty * GAMMA, MAX_W)
          action = PenaltyAction.BUMPED
        elif self.ema_violation < DECAY_TAU * self.last_check_violation:
          self.w_penalty = max(self.w_penalty / GAMMA_DOWN, MIN_W)
          action = PenaltyAction.DECAYED
      self.last_check_violation = self.ema_violation

    return action

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
  max_impulse_steps: int,
) -> Callable:
  """
  Creates a periodic learning rate schedule to continuously kick the model out
  of local minima. Decays rapidly, then stays flat at baseline.

  Uses geometric progression for the cycle lengths: T_i = T_0 * M^i.
  Caps the cosine decay duration at max_impulse_steps to prevent extended
  high-temperature burns in late geometric cycles.
  """

  def schedule(step):
    t_0 = float(kick_period_0)
    m = float(kick_period_mult)

    n = jnp.floor(jnp.log(1.0 + step * (m - 1.0) / t_0) / jnp.log(m))
    cycle_start = t_0 * (m**n - 1.0) / (m - 1.0)
    cycle_length = t_0 * (m**n)
    cycle_step = step - cycle_start

    # Cap the impulse duration to prevent extended high-LR burns in late
    # geometric cycles (Loshchilov & Hutter 2017; WSD paradigm).
    decay_length = jnp.minimum(cycle_length / 5.0, float(max_impulse_steps))

    progress = jnp.minimum(cycle_step / decay_length, 1.0)
    cosine_val = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))

    return learning_rate + (peak_learning_rate - learning_rate) * cosine_val

  return schedule
