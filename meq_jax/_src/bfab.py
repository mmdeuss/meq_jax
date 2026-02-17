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

from meq_jax._src import bfef

# pylint: disable=invalid-name


def f(
    FxA: jt.Float[jt.Array, ''],
    FBA: jt.Float[jt.Array, ''],
    nP: int,
    nT: int,
) -> tuple[jt.Float[jt.Array, 'nP+nT'], jt.Float[jt.Array, 'nP+nT']]:
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
  g, Ig = bfef.f_(FxA, FBA)
  return dispatch(g, Ig, nP, nT)


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
  alphapg, alphag = bfef.fag_(FBA)
  return dispatch(alphapg, alphag, nP, nT)


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
  g, Ig = bfef.fA_(FBA)
  return dispatch(g, Ig, nP, nT)


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
  g, Ig = bfef.fN_(FN)
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
  return bfef.fPg(nP, nT)


def dispatch(
    g_in: jt.Float[jt.Array, '... 4'],
    Ig_in: jt.Float[jt.Array, '... 4'],
    nP: int,
    nT: int,
) -> tuple[jt.Float[jt.Array, '... nP+nT'], jt.Float[jt.Array, '... nP+nT']]:
  """Selects and concatenates elementary basis functions.

  This function mimics the `bfab_dispatch` logic in C. It constructs the
  final basis function set by taking specific slices from the 4-element
  elementary basis function arrays. It skips the first element (index 0)
  of the elementary set.

  Args:
      g_in: The elementary 'g' basis functions
      Ig_in: The elementary 'Ig' basis functions
      nP: The number of 'P' type basis functions to select.
      nT: The number of 'T' type basis functions to select.

  Returns:
      A tuple (g_out, Ig_out) with the final basis functions, where each
      array has shape (..., nP + nT).
  """
  # Select nP elements for the P' part, starting from index 1
  g_p_part = g_in[..., 1 : 1 + nP]
  Ig_p_part = Ig_in[..., 1 : 1 + nP]

  # Select nT elements for the TT' part, also starting from index 1
  g_t_part = g_in[..., 1 : 1 + nT]
  Ig_t_part = Ig_in[..., 1 : 1 + nT]

  # Concatenate the parts to form the final set
  g_out = jnp.concatenate([g_p_part, g_t_part], axis=-1)
  Ig_out = jnp.concatenate([Ig_p_part, Ig_t_part], axis=-1)

  return g_out, Ig_out


# pylint: enable=invalid-name
