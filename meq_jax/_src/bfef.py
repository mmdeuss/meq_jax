# Copyright 2026 The meq_jax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAX implementation of the bfef functions."""

import jax.numpy as jnp
import jaxtyping as jt

# pylint: disable=invalid-name


def f_(
    FxA: jt.Float[jt.Array, ''],
    FBA: jt.Float[jt.Array, ''],
) -> tuple[jt.Float[jt.Array, '4'], jt.Float[jt.Array, '4']]:
  """Computes the 4 elementary physical basis functions for a grid point.

  Args:
      FxA: The value of (Fx - FA) at a point.
      FBA: The value of (FB - FA).

  Returns:
      A tuple (g, Ig) of the 4-element elementary basis functions.
  """
  g1 = FxA - FBA
  g2 = g1 * FxA
  g = jnp.array([1.0, g1, g2, g2 * (g1 + FxA)])

  Ig1 = 0.5 * g1**2
  Ig = jnp.array([g1, Ig1, Ig1 * (2.0 / 3.0 * g1 + FBA), 0.5 * g2**2])
  return g, Ig


def f(
    FxA: jt.Float[jt.Array, ''],
    FBA: jt.Float[jt.Array, ''],
    nP: int,
    nT: int,
) -> tuple[jt.Float[jt.Array, '... nP+nT'], jt.Float[jt.Array, '... nP+nT']]:
  """Computes the 4 elementary physical basis functions for a grid point.

  Dispatches the result based on the number of P' and TT' basis functions.

  Args:
      FxA: The value of (Fx - FA) at a point.
      FBA: The value of (FB - FA).
      nP: The number of 'P' type basis functions.
      nT: The number of 'T' type basis functions.

  Returns:
      A tuple (g, Ig) of the 4-element elementary basis functions.
  """
  g, Ig = f_(FxA, FBA)
  return dispatch(g, Ig, nP, nT)


def fag_(
    FBA: jt.Float[jt.Array, ''],
) -> tuple[jt.Float[jt.Array, '4'], jt.Float[jt.Array, '4']]:
  """Conversion factors from normalized to physical basis functions.

  Args:
      FBA: The value of (FB - FA).

  Returns:
      A tuple (alphapg, alphag) of the conversion factors.
  """
  alphapg = jnp.pow(FBA, jnp.arange(4))
  return alphapg, FBA * alphapg


def fag(
    FBA: jt.Float[jt.Array, ''],
    nP: int,
    nT: int,
) -> tuple[jt.Float[jt.Array, 'nP+nT'], jt.Float[jt.Array, 'nP+nT']]:
  """Conversion factors from normalized to physical basis functions.

  Dispatches the result based on the number of P' and TT' basis functions.

  Args:
      FBA: The value of (FB - FA).
      nP: The number of 'P' type basis functions.
      nT: The number of 'T' type basis functions.

  Returns:
      A tuple (alphapg, alphag) of the conversion factors.
  """
  alphapg, alphag = fag_(FBA)
  return dispatch(alphapg, alphag, nP, nT)


def fA_(
    FBA: jt.Float[jt.Array, ''],
) -> tuple[jt.Float[jt.Array, '4'], jt.Float[jt.Array, '4']]:
  """Computes the 4 elementary physical basis functions at the magnetic axis.

  Args:
      FBA: The value of (FB - FA).

  Returns:
      A tuple (g, Ig) of the 4-element elementary basis functions at the axis.
  """
  g = jnp.array([1.0, -FBA, 0.0, 0.0])
  Ig = jnp.array([-FBA, 0.5 * FBA**2, (1.0 / 6.0) * FBA**3, 0.0])
  return g, Ig


def fA(
    FBA: jt.Float[jt.Array, ''],
    nP: int,
    nT: int,
) -> tuple[jt.Float[jt.Array, 'nP+nT'], jt.Float[jt.Array, 'nP+nT']]:
  """Computes the 4 elementary physical basis functions at the magnetic axis.

  Dispatches the result based on the number of P' and TT' basis functions.

  Args:
      FBA: The value of (FB - FA).
      nP: The number of 'P' type basis functions.
      nT: The number of 'T' type basis functions.

  Returns:
      A tuple (g, Ig) of the 4-element elementary basis functions at the axis.
  """
  g, Ig = fA_(FBA)
  return dispatch(g, Ig, nP, nT)


def fN_(
    FN: jt.Float[jt.Array, ''],
) -> tuple[jt.Float[jt.Array, '4'], jt.Float[jt.Array, '4']]:
  """Computes normalized basis functions.

  Args:
      FN: The value of the normalized basis function.

  Returns:
      A tuple (g, Ig) of the 4-element normalized basis functions.
  """
  g = jnp.array(
      [1.0, FN - 1.0, (FN - 1.0) * FN, (FN - 1.0) * FN * (2 * FN - 1.0)]
  )
  Ig = jnp.array([
      FN - 1.0,
      0.5 * (FN - 1.0) ** 2,
      (FN - 1.0) ** 2 * (2.0 * FN + 1.0) / 6.0,
      0.5 * (FN * (FN - 1.0)) ** 2,
  ])
  return g, Ig


def fN(
    FN: jt.Float[jt.Array, ''],
    nP: int,
    nT: int,
) -> tuple[jt.Float[jt.Array, 'nP+nT'], jt.Float[jt.Array, 'nP+nT']]:
  """Computes normalized basis functions.

  Dispatches the result based on the number of P' and TT' basis functions.

  Args:
      FN: The value of the normalized basis function.
      nP: The number of 'P' type basis functions.
      nT: The number of 'T' type basis functions.

  Returns:
      A tuple (g, Ig) of the 4-element normalized basis functions.
  """
  g, Ig = fN_(FN)
  return dispatch(g, Ig, nP, nT)


def fPg(
    nP: int,
    nT: int,
) -> tuple[jt.Float[jt.Array, 'nP+nT'], jt.Float[jt.Array, 'nP+nT']]:
  """Assigns each function to P' or TT'.

  Args:
      nP: The number of 'P' type basis functions.
      nT: The number of 'T' type basis functions.

  Returns:
      A tuple (fPg, fTg) of the basis function types.
  """
  fpg = jnp.concatenate([jnp.ones(nP), jnp.zeros(nT)])
  ftg = jnp.concatenate([jnp.zeros(nP), jnp.ones(nT)])
  return fpg, ftg


def dispatch(
    g_in: jt.Float[jt.Array, '... 4'],
    Ig_in: jt.Float[jt.Array, '... 4'],
    nP: int,
    nT: int,
) -> tuple[jt.Float[jt.Array, '... nP+nT'], jt.Float[jt.Array, '... nP+nT']]:
  """Selects and concatenates elementary basis functions.

  This function mimics the `bfef_dispatch` logic in C. It constructs the
  final basis function set by taking specific slices from the 4-element
  elementary basis function arrays.

  Args:
      g_in: The elementary 'g' basis functions
      Ig_in: The elementary 'Ig' basis functions
      nP: The number of 'P' type basis functions to select.
      nT: The number of 'T' type basis functions to select.

  Returns:
      A tuple (g_out, Ig_out) with the final basis functions, where each
      array has shape (..., nP + nT).
  """
  # Select nP elements for the P' part
  g_p_part = g_in[..., :nP]
  Ig_p_part = Ig_in[..., :nP]

  # Select nT elements for the TT' part
  g_t_part = g_in[..., :nT]
  Ig_t_part = Ig_in[..., :nT]

  # Concatenate the parts to form the final set
  g_out = jnp.concatenate([g_p_part, g_t_part], axis=-1)
  Ig_out = jnp.concatenate([Ig_p_part, Ig_t_part], axis=-1)

  return g_out, Ig_out


# pylint: enable=invalid-name
