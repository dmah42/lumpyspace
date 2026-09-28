import jax
import jax.numpy as jnp

from src.core.metric import MetricNN
from src.training.loss import get_cmb_loss


def test_cmb_loss_outputs():
  key = jax.random.PRNGKey(42)
  model = MetricNN(key)
  coords = jnp.array([-3.995, 0.5, -0.2, 0.8])

  l_expand, l_shear, l_spatial = get_cmb_loss(model, coords)

  assert l_expand.shape == ()
  assert l_shear.shape == ()
  assert l_spatial.shape == ()

  assert not jnp.isnan(l_expand)
  assert not jnp.isnan(l_shear)
  assert not jnp.isnan(l_spatial)

  # Ensure they are non-negative since they are penalties
  assert l_expand >= 0.0
  assert l_shear >= 0.0
  assert l_spatial >= 0.0
