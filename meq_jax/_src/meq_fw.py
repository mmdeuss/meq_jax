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

"""Flux functions."""

import jax
import jax.numpy as jnp

# pylint: disable=invalid-name


# TODO(ncasagrande): Give better (pythonic) names.
def meq_fw(
    FB: jnp.ndarray,
    FX: jnp.ndarray,
    lX: jnp.ndarray,
    nX: int,
    nFW: int,
    dimw: int,
) -> jnp.ndarray:
  """Determines flux values to be tracked by a gap algorithm.

  Args:
    FB: Flux at the limiter boundary.
    FX: Fluxes at all X points. Shape doesn't really matter as long as one
      dimension is 1, and thereforce can be flattened.
    lX: Whether the limiter is diverted.
    nX: Number of X points (since FX may be larger than that).
    nFW: Number of flux values to be tracked.
    dimw: Maximum number of X points.

  Returns:
    First the boundary value, then X points ordered by flux.
  """
  if nFW <= 0:
    return jnp.zeros(0, dtype=FB.dtype)

  def _nX_neg(FX, FB):  # pylint: disable=unused-argument
    # Limiter with no X-points: track the boundary flux everywhere.
    return jnp.full((nFW,), FB, dtype=FB.dtype)

  def _nX_pos(FX, FB):
    # Sort the relevant subset of FX based on its distance to FB.
    # This is only computed if there are X-points to consider.
    # Flatten the input FX to a 1D array.

    # Replace Matlab: fx_subset = FX.flatten()[:nX] where nX is dynamic
    # to guarantee fixed shape.
    # Create a 1D array with a fixed upper bound size
    FX_extended = jnp.full((dimw,), fill_value=-jnp.inf, dtype=FB.dtype)

    # Create a mask to select the first nX entries of FX.
    FX_flattened_shape = FX.flatten().shape[0]
    indices = jnp.arange(FX_flattened_shape)
    FX_flattened_cropped = jnp.where(indices < nX, FX.flatten(), -jnp.inf)

    # Update the 1D array with the first nX entries of FX.
    FX_extended = jax.lax.dynamic_update_slice(
        FX_extended, FX_flattened_cropped, (0,)
    )

    sort_indices = jnp.argsort(jnp.abs(FX_extended - FB))
    sFX = jnp.take(
        # FX,
        FX_extended,
        sort_indices,
        indices_are_sorted=True,
        unique_indices=True,
    )

    def _lX_true(sFX, FB):  # pylint: disable=unused-argument
      nFWX = jnp.minimum(nX, nFW)
      # Fill FW with the first `nFWX` sorted flux values from sFX.
      # Pad any remaining elements with the last value used (sFX[nFWX-1]).
      # This is done efficiently by clipping the gather indices.
      gather_indices = jnp.minimum(jnp.arange(nFW), nFWX - 1)
      return jnp.take(sFX, gather_indices, 0)

    def _lX_false(sFX, FB):
      nFWX = jnp.minimum(nX, nFW - 1)
      # The source of values is the boundary flux followed by the sorted
      # X-point fluxes.
      source_values = jnp.concatenate((jnp.array([FB]), sFX))
      # Select FW[0] = source[0] (i.e., FB),
      # FW[1..nFWX] = source[1..nFWX] (i.e., sFX[0..nFWX-1]),
      # and pad the rest with the last value, source_values[nFWX].
      # This is done by clipping the gather indices at `nFWX`.
      gather_indices = jnp.minimum(jnp.arange(nFW), nFWX)
      return jnp.take(source_values, gather_indices, 0)

    return jax.lax.cond(
        lX,
        _lX_true,
        _lX_false,
        sFX,
        FB,
    )

  return jax.lax.cond(nX <= 0, _nX_neg, _nX_pos, FX, FB)


# pylint: enable=invalid-name
