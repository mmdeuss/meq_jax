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

"""minQmex JAX implementation.

This overrides the implementation in
  third_party/meq/mexc/minQmex.c
whose pure Matlab implementation is in
  third_party/meq/mexm/minQmexm.m
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import jaxtyping as jt

# pylint: disable=invalid-name


def minQ(
    aQ: jt.Float[jt.Array, 'nQ'],
    qQ: jt.Float[jt.Array, 'nQ'],
    sq: jt.Float[jt.Array, ''],
    n: int = 1,
    NF: float = jnp.nan,
) -> Tuple[jt.Float[jt.Array, 'n'], jt.Float[jt.Array, 'n']]:
  """Find at most N local extrema values of the profile(AQ,QQ)

  using local quadratic interpolation. Count N starting outboard
  Also finds extrema at edge of domain.

  Args:
    aQ: x data
    qQ: y data
    sQ: SQ=1: minima, SQ=-1: maxima
    n: default=1
    NF: return value in case less than N extrema were found
  """
  ismin = sq > 0
  nQ = qQ.shape[0]
  assert nQ >= 2, 'nQ must be at least 2'

  # Initial state
  aqmin_init = jnp.full((n,), NF, dtype=aQ.dtype)
  qmin_init = jnp.full((n,), NF, dtype=qQ.dtype)
  imin_init = 0
  done_init = jnp.array(False)

  # Outer point
  def outer_point_fn(state):
    """Logic to execute if the outer point is an extremum."""
    _, aqmin, qmin, _ = state
    # Add the last point
    aqmin = aqmin.at[0].set(aQ[nQ - 1])
    qmin = qmin.at[0].set(qQ[nQ - 1])
    # Increment index and check if we are done
    imin = 1
    done = jnp.array(n == 1)
    return imin, aqmin, qmin, done

  # Condition for the outer point check
  # qQ[nQ - 1] in Python maps to qQ[nQ] in MATLAB
  cond_1 = jnp.logical_and(ismin, qQ[nQ - 1] - qQ[nQ - 2] < 0)
  cond_2 = jnp.logical_and(~ismin, qQ[nQ - 1] - qQ[nQ - 2] > 0)
  outer_cond = jnp.logical_or(cond_1, cond_2)

  state_after_outer = jax.lax.cond(
      outer_cond,
      outer_point_fn,
      lambda state: state,
      (imin_init, aqmin_init, qmin_init, done_init),
  )

  # Inner points
  # We replace the `for` loop with `jax.lax.fori_loop`.
  # The loop variable `i` will go from 0 to nQ-3.
  # We map this to the MATLAB index `kQ` which goes from nQ-2 down to 1.

  def loop_body(i, state):
    """Main loop."""
    _, _, _, done = state

    def processing_fn(state_inner):
      """Main logic: find extrema if not done."""

      # Map loop index `i` to MATLAB's `kQ`
      kQ = (nQ - 1) - i
      # Difference between MATLAB and Python indices
      kQ = kQ - 1

      # Condition for extremum (derivative sign change)
      cond_1 = jnp.logical_and(
          ismin,
          jnp.logical_and(qQ[kQ + 1] - qQ[kQ] > 0, qQ[kQ] - qQ[kQ - 1] <= 0),
      )
      cond_2 = jnp.logical_and(
          ~ismin,
          jnp.logical_and(qQ[kQ + 1] - qQ[kQ] < 0, qQ[kQ] - qQ[kQ - 1] >= 0),
      )
      cond_is_extremum = jnp.logical_or(cond_1, cond_2)

      def quadratic_fit_fn(state_fit):
        """Calculates and adds the extremum via quadratic fit."""
        imin_f, aqmin_f, qmin_f, _ = state_fit

        # Quadratic fit logic
        x32 = aQ[kQ + 1] - aQ[kQ]
        x12 = aQ[kQ - 1] - aQ[kQ]
        y32 = qQ[kQ + 1] - qQ[kQ]
        y12 = qQ[kQ - 1] - qQ[kQ]

        # Denominator (inverse determinant)
        iD = 1.0 / (x32 * x12 * (x32 - x12))

        # Coefficients (from explicit 2x2 inverse)
        p1 = x12 * y32 - x32 * y12
        p2 = x32 * x32 * y12 - x12 * x12 * y32

        # Location of extremum
        xmin = -p2 / (2.0 * p1)
        ymin = xmin * p2 * iD * 0.5

        # Add the new point
        aqmin_f = aqmin_f.at[imin_f].set(aQ[kQ] + xmin)
        qmin_f = qmin_f.at[imin_f].set(qQ[kQ] + ymin)

        # Increment index
        imin_f = imin_f + 1

        # Check for early exit
        done_f = imin_f >= n

        return imin_f, aqmin_f, qmin_f, done_f

      return jax.lax.cond(
          cond_is_extremum, quadratic_fit_fn, lambda state: state, state_inner
      )

    return jax.lax.cond(
        done,
        lambda state: state,  # If done, do nothing
        processing_fn,  # If not done, run the main logic
        state,
    )

  state_after_loop = jax.lax.fori_loop(0, nQ - 2, loop_body, state_after_outer)
  _, _, _, done = state_after_loop

  # If still counting, then first point is a minimum
  def first_point_fn(state):
    """Add the first point if we are not yet done."""
    imin_f, aqmin_f, qmin_f, done_f = state
    aqmin_f = aqmin_f.at[imin_f].set(aQ[0])
    qmin_f = qmin_f.at[imin_f].set(qQ[0])
    # No need to update imin or done, we are at the end.
    return imin_f, aqmin_f, qmin_f, done_f

  _, aqmin_final, qmin_final, _ = jax.lax.cond(
      done,
      lambda state: state,
      first_point_fn,
      state_after_loop,
  )
  return aqmin_final, qmin_final


# pylint: enable=invalid-name
