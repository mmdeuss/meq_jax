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

"""JAX implementation of the locQ.m file."""

import functools

import jax
import jax.numpy as jnp
import jaxtyping as jt


# pylint: disable=invalid-name


def _locQ_single_iq(
    aQ: jt.Float[jt.Array, 'nQ'],
    iqQ: jt.Float[jt.Array, 'nQ'],
    iq: jt.Float[jt.Array, ''],
    naR: int,
    NF: float,
) -> jt.Float[jt.Array, 'naR']:
  """Helper function to process a single iq value."""
  nQ = aQ.shape[0]

  iqQ_rev = iqQ[::-1]
  aQ_rev = aQ[::-1]

  def while_cond(state):
    kQ_rev, nrq, _ = state
    return (kQ_rev < nQ - 1) & (nrq < naR)

  def while_body(state):
    kQ_rev, nrq, aR_single = state

    iqQ_kQ = iqQ_rev[kQ_rev]
    iqQ_kQ_minus_1 = iqQ_rev[kQ_rev + 1]
    aQ_kQ = aQ_rev[kQ_rev]
    aQ_kQ_minus_1 = aQ_rev[kQ_rev + 1]

    is_between = (iq - iqQ_kQ) * (iq - iqQ_kQ_minus_1) < 0
    is_equal_kQ = iq == iqQ_kQ
    is_equal_last = (kQ_rev == nQ - 2) & (iq == iqQ_kQ_minus_1)

    denom = iqQ_kQ - iqQ_kQ_minus_1
    safe_denom = jnp.where(denom == 0, 1.0, denom)
    interp_val = aQ_kQ + (iq - iqQ_kQ) * (aQ_kQ - aQ_kQ_minus_1) / safe_denom

    def update_state_fn(val_to_set, current_st):
      nrq_, aR_single_ = current_st
      return nrq_ + 1, aR_single_.at[nrq_].set(val_to_set)

    # Nested conds to implement if/elif/else
    new_nrq, new_aR_single = jax.lax.cond(
        is_between,
        functools.partial(update_state_fn, interp_val),
        functools.partial(
            jax.lax.cond,
            is_equal_kQ,
            functools.partial(update_state_fn, aQ_kQ),
            functools.partial(
                jax.lax.cond,
                is_equal_last,
                functools.partial(update_state_fn, aQ_kQ_minus_1),
                lambda x: x,
            ),
        ),
        (nrq, aR_single),
    )

    return kQ_rev + 1, new_nrq, new_aR_single

  return jax.lax.while_loop(
      while_cond, while_body, (0, 0, jnp.full((naR,), NF))
  )[2]


def locQ(
    aQ: jt.Float[jt.Array, 'nO nQ'],
    iqQ: jt.Float[jt.Array, 'nQ'],
    iqR: jt.Float[jt.Array, 'nR'],
    naR: int = 1,
    NF: float = jnp.nan,
) -> jt.Float[jt.Array, 'nO nR naR']:
  """Location of q surfaces.

  This function is a JAX implementation of the locQ function.

  NOTE: The output axes currently match the matlab implementation, but this
  might not be the best shape for downstream calculations. It can easily be
  changed by modifying the vmap out axes.

  Args:
      aQ: x data (can be a matrix of size [nO,nQ] when QQ has nQ elements)
      iqQ: y data
      iqR: Requested y values (will be of size [nR] )
      naR: (optional) Number of maximum R values to be found (per QR) for
        nonmonotonic profiles. (default=1: outermost value found)
      NF: (optional) AR = NF signals the q value was not found

  Returns:
      AR: Location of the flux surfaces where QR=QQ(AQ==AR)
  """
  if naR < 1:
    raise ValueError('invalid naR < 1')

  if aQ.shape[1] != iqQ.shape[0]:
    raise ValueError('aQ must have iqQ columns')

  def inner_vmap(
      iqR_: jt.Float[jt.Array, ''], aQ_: jt.Float[jt.Array, 'n0 nQ']
  ) -> jt.Float[jt.Array, 'n0 naR']:
    return jax.vmap(lambda aq: _locQ_single_iq(aq, iqQ, iqR_, naR, NF))(aQ_)

  return jax.vmap(inner_vmap, in_axes=(0, None), out_axes=1)(iqR, aQ)


# pylint: enable=invalid-name
