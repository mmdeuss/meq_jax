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

"""bbox majic implementation."""

import jax.numpy as jnp
import jaxtyping as jt


def bboxmex(
    lxy: jt.Bool[jt.Array, 'nx ny'],
    y: jt.Float[jt.Array, 'ny'],
    x: jt.Float[jt.Array, 'nx'],
) -> jt.Float[jt.Array, '1 4']:
  """Computes the bounding box of points (x, y) where lxy is a boolean mask.

  Args:
    lxy: 2D boolean array, True for points to include in the bounding box.
    y: 1D array of y-coordinates (associated with rows)
    x: 1D array of x-coordinates (associated with columns)

  Returns:
    A jax array of shape [1, 4] containing the bounding box y1, x1, y2, x2.
    For more details, see meq/bboxmex.m.
  """
  masked_x = jnp.where(lxy, x[:, None], jnp.nan)
  masked_y = jnp.where(lxy, y[None, :], jnp.nan)

  x1 = jnp.nanmin(masked_x)
  x2 = jnp.nanmax(masked_x)
  y1 = jnp.nanmin(masked_y)
  y2 = jnp.nanmax(masked_y)

  # Handle the case where no points are selected (all NaN)
  no_points_selected = jnp.all(jnp.isnan(masked_x))
  x1 = jnp.where(no_points_selected, 0.0, x1)
  x2 = jnp.where(no_points_selected, 0.0, x2)
  y1 = jnp.where(no_points_selected, 0.0, y1)
  y2 = jnp.where(no_points_selected, 0.0, y2)

  return jnp.array([y1, x1, y2, x2])
