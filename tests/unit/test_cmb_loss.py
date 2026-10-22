import equinox as eqx
import jax
import jax.numpy as jnp

from src.core.metric import MetricNN
from src.training.loss import CMB_ACCURACY, cmb_fractions, get_cmb_loss


def _flrw(h):
  def metric(coords):
    a_sq = jnp.exp(2.0 * h * coords[0])
    return jnp.diag(jnp.array([-1.0, a_sq, a_sq, a_sq]))

  return metric


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


def test_cmb_loss_is_bounded_at_full_violation():
  # Contracting FLRW: every directional rate is contraction, so the
  # expansion constraint is fully violated and scores 1 less its tolerance.
  l_expand, l_shear, l_spatial = get_cmb_loss(
    _flrw(-0.3), jnp.array([-3.995, 0.0, 0.0, 0.0])
  )

  assert jnp.allclose(l_expand, 1.0 - CMB_ACCURACY**2, atol=1e-6)
  assert l_shear == 0.0
  assert l_spatial == 0.0


def test_cmb_loss_expanding_flrw_and_minkowski_meet_every_condition():
  def minkowski(coords):
    return jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))

  for metric in (_flrw(0.3), minkowski):
    losses = get_cmb_loss(metric, jnp.array([-3.995, 0.2, -0.4, 0.6]))
    for loss in losses:
      assert loss == 0.0


def test_cmb_fractions_gradient_is_the_perturbation_amplitude():
  eps = 0.01

  def metric(coords):
    return jnp.diag(jnp.array([-1.0, 1.0 + eps * coords[1], 1.0, 1.0]))

  _, _, gradient = cmb_fractions(metric, jnp.array([0.0, 0.3, 0.0, 0.0]))

  assert jnp.allclose(gradient, eps, rtol=1e-5)


def test_cmb_fractions_stay_bounded_near_static_geometry():
  # Rates far below H_FLOOR: a nearly static, sheared metric. The raw
  # (sigma/H)^2 is unbounded here; the fractions stay in [0, 1].
  p = jnp.array([1e-7, -2e-7, 1e-7])

  def metric(coords):
    return jnp.diag(
      jnp.concatenate([jnp.array([-1.0]), jnp.exp(2 * p * coords[0])])
    )

  shear, contraction, _ = cmb_fractions(metric, jnp.array([0.0, 0, 0, 0]))

  assert 0.0 <= shear < 1e-4
  assert 0.0 <= contraction < 1e-4


def test_cmb_loss_gradients_are_finite_at_initialisation():
  model = MetricNN(jax.random.PRNGKey(0))
  coords = jax.random.uniform(jax.random.PRNGKey(1), (8, 4), minval=-1.0)
  coords = coords.at[:, 0].set(-3.995)

  def total(m):
    losses = jax.vmap(lambda c: get_cmb_loss(m, c))(coords)
    return sum(jnp.mean(loss) for loss in losses)

  grads = eqx.filter_grad(total)(model)
  leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))

  assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
