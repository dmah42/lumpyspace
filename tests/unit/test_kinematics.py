import jax
import jax.numpy as jnp

from src.core.kinematics import comoving_kinematics, comoving_spatial_metric


def _kinematics(metric_fn, coords):
  return comoving_kinematics(metric_fn(coords), jax.jacfwd(metric_fn)(coords))


def _bell_metric(accel):
  # Minkowski space in coordinates that move rigidly with velocity
  # v(t) = accel * t along x: x = X + accel * t^2 / 2.
  def metric(coords):
    v = accel * coords[0]
    return jnp.array(
      [
        [-(1.0 - v**2), v, 0.0, 0.0],
        [v, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
      ]
    )

  return metric


def test_flrw_expands_isotropically():
  # a(t) = exp(h t): every axis expands at h and there is no shear.
  h = 0.3

  def metric(coords):
    a_sq = jnp.exp(2.0 * h * coords[0])
    return jnp.diag(jnp.array([-1.0, a_sq, a_sq, a_sq]))

  kinematics = _kinematics(metric, jnp.array([0.5, 0.1, -0.2, 0.3]))

  assert jnp.allclose(kinematics.h_dir, h, atol=1e-5)
  assert jnp.allclose(kinematics.h_mean, h, atol=1e-5)
  assert jnp.allclose(kinematics.sigma_sq, 0.0, atol=1e-8)


def test_kasner_matches_analytic_rates():
  # Kasner: a_i = t^p_i, so H_i = p_i / t and
  # sigma^2 = 1/2 sum_i (p_i - 1/3)^2 / t^2.
  p = jnp.array([-1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])
  t = 2.0

  def metric(coords):
    return jnp.diag(jnp.concatenate([jnp.array([-1.0]), coords[0] ** (2 * p)]))

  kinematics = _kinematics(metric, jnp.array([t, 0.0, 0.0, 0.0]))

  assert jnp.allclose(kinematics.h_dir, p / t, atol=1e-5)
  assert jnp.allclose(kinematics.h_mean, 1.0 / (3.0 * t), atol=1e-5)
  assert jnp.allclose(kinematics.sigma_sq, 1.0 / 12.0, atol=1e-5)


def test_shift_terms_carry_bells_spaceship_stretch():
  # Observers accelerated together in the lab frame drift apart in their own
  # rest frame (Bell's spaceship). g_xx is constant here, so the expansion
  # along x comes entirely from the shift terms: theta = gamma^3 a^2 t.
  accel, t = 1.0, 0.5
  v = accel * t
  gamma = 1.0 / jnp.sqrt(1.0 - v**2)
  theta = gamma**3 * accel**2 * t

  kinematics = _kinematics(_bell_metric(accel), jnp.array([t, 0.0, 0.0, 0.0]))

  assert jnp.allclose(kinematics.h_dir, jnp.array([theta, 0.0, 0.0]), atol=1e-5)
  assert jnp.allclose(kinematics.h_mean, theta / 3.0, atol=1e-5)
  assert jnp.allclose(kinematics.sigma_sq, theta**2 / 3.0, atol=1e-5)


def test_comoving_spatial_metric():
  g = jnp.diag(jnp.array([-2.0, 1.5, 2.0, 3.0]))
  assert jnp.allclose(comoving_spatial_metric(g), g[1:4, 1:4])

  # In the rest frame of observers moving at v, rulers along x read gamma^2.
  v = 0.5
  h = comoving_spatial_metric(_bell_metric(1.0)(jnp.array([v, 0.0, 0.0, 0.0])))
  assert jnp.allclose(h[0, 0], 1.0 / (1.0 - v**2), atol=1e-6)
  assert jnp.allclose(h[1:, 1:], jnp.eye(2))
