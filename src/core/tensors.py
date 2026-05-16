"""
Automated tensor algebra for PINN physics losses.
Computes Christoffel symbols, Riemann, and Ricci tensors using JAX AD.
"""

from typing import Callable

import jax
import jax.numpy as jnp


def get_christoffel_symbols(
  metric_fn: Callable[[jnp.ndarray], jnp.ndarray], coords: jnp.ndarray
):
  """
  Computes Christoffel symbols Gamma^alpha_{mu nu}.
  Gamma^alpha_{mu nu} = 0.5 * g^alpha_sigma * (d_mu g_{ns} + d_nu g_{ms}
                        - d_sigma g_{mu nu})
  """
  g = metric_fn(coords)
  g_inv = jnp.linalg.inv(g)

  # Jacobian: dg[i, j, k] = d_k g_{ij}
  dg = jax.jacfwd(metric_fn)(coords)

  # Gamma^a_{bc} = 0.5 * g^{ad} (d_b g_{cd} + d_c g_{bd} - d_d g_{bc})
  # indices: a=alpha, b=mu, c=nu, d=sigma
  # d_b g_{cd} -> dg[c, d, b]
  # d_c g_{bd} -> dg[b, d, c]
  # d_d g_{bc} -> dg[b, c, d]

  gamma = 0.5 * (
    jnp.einsum("ad,cdb->abc", g_inv, dg)
    + jnp.einsum("ad,bdc->abc", g_inv, dg)
    - jnp.einsum("ad,bcd->abc", g_inv, dg)
  )
  return gamma


def get_ricci_tensor(
  metric_fn: Callable[[jnp.ndarray], jnp.ndarray], coords: jnp.ndarray
):
  """
  Computes the Ricci tensor R_{mu nu}.
  R_{mn} = d_r G^r_mn - d_n G^r_mr + G^r_mn G^s_rs - G^s_mr G^r_ns
  """

  def gamma_fn(c):
    return get_christoffel_symbols(metric_fn, c)

  gamma = gamma_fn(coords)
  # d_gamma Gamma^alpha_{mu nu} -> shape (4, 4, 4, 4)
  # Index structure: dgamma[alpha, mu, nu, gamma] = d_gamma Gamma^alpha_{mu nu}
  dgamma = jax.jacfwd(gamma_fn)(coords)

  # R_{mu nu}
  # Term 1: d_rho Gamma^rho_{mu nu}
  term1 = jnp.trace(dgamma, axis1=0, axis2=3)  # Sums over rho

  # Term 2: d_nu Gamma^rho_{mu rho}
  # dgamma[rho, mu, rho, nu] = d_nu Gamma^rho_{mu rho}
  term2 = jnp.trace(dgamma, axis1=0, axis2=2)  # Result is [mu, nu]

  # Term 3: Gamma^rho_{mu nu} Gamma^sigma_{rho sigma}
  # gamma_contracted_sigma[rho] = Gamma^sigma_{rho sigma}
  gamma_contracted_sigma = jnp.trace(gamma, axis1=0, axis2=2)
  term3 = jnp.einsum("rmn,r->mn", gamma, gamma_contracted_sigma)

  # Term 4: Gamma^sigma_{mu rho} Gamma^rho_{nu sigma}
  term4 = jnp.einsum("smr,rns->mn", gamma, gamma)

  return term1 - term2 + term3 - term4


def get_ricci_scalar(
  metric_fn: Callable[[jnp.ndarray], jnp.ndarray], coords: jnp.ndarray
):
  """Computes Ricci Scalar R = g^{mu nu} R_{mu nu}."""
  g = metric_fn(coords)
  g_inv = jnp.linalg.inv(g)
  ricci_tensor = get_ricci_tensor(metric_fn, coords)
  return jnp.einsum("mn,mn->", g_inv, ricci_tensor)
