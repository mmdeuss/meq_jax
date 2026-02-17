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

"""bavx majic implementation."""

import jax.numpy as jnp
import jaxtyping as jt


# pylint: disable=invalid-name
def bavx(
    Rx: jt.Float[jt.Array, 'nx'],
    Zx: jt.Float[jt.Array, 'nx'],
    vrx: jt.Float[jt.Array, 'nx'],
    vzx: jt.Float[jt.Array, 'nx'],
    A: jt.Float[jt.Array, ''],
    R: jt.Float[jt.Array, 'nr'],
    Z: jt.Float[jt.Array, 'nr'],
    mask: jt.Bool[jt.Array, 'nx'],
) -> jt.Bool[jt.Array, 'nr']:
  """Finds points inside X-point polygon.

  See meq/bavxmex.m for more details.

  Note the existence of the mask argument, which is necessary to make meqpdom
  work with fixed-size arrays. This argument is not present in the MATLAB
  version - the function here is equivalent to calling the MATLAB version with
  arguments Rx[mask], Zx[mask], vrx[mask] and vzx[mask].
  """
  # Create a grid of rpx and zpx.
  rpx = R[:, None] - Rx
  zpx = Z[:, None] - Zx

  # Compute the condition for each point.
  condition = (vrx * rpx + vzx * zpx) < A * jnp.abs(vzx * rpx - vrx * zpx)

  # Check if the condition is true for any point in a row.
  return jnp.logical_not(
      jnp.any(jnp.logical_and(condition, mask), axis=1)).reshape(R.size)


def bavx2(
    Rx: jt.Float[jt.Array, 'nx'],
    Zx: jt.Float[jt.Array, 'nx'],
    vrx: jt.Float[jt.Array, 'nx'],
    vzx: jt.Float[jt.Array, 'nx'],
    A: jt.Float[jt.Array, ''],
    R: jt.Float[jt.Array, 'nr'],
    Z: jt.Float[jt.Array, 'nz'],
    mask: jt.Bool[jt.Array, 'nx'],
    # bavxmex triggers bavx2 based on the existence of this argument.
    selector: None,
) -> jt.Bool[jt.Array, 'nr nz']:
  """Finds points inside X-point polygon (mesh version).

  See meq/bavxmex.m for more details.

  Note the existence of the mask argument, which is necessary to make meqpdom
  work with fixed-size arrays. This argument is not present in the MATLAB
  version - the function here is equivalent to calling the MATLAB version with
  arguments Rx[mask], Zx[mask], vrx[mask] and vzx[mask].
  """
  del selector
  # Create a grid of rpx and zpx
  z_grid, r_grid = jnp.meshgrid(Z, R)
  rpx = r_grid[:, :, None] - Rx
  zpx = z_grid[:, :, None] - Zx

  # Compute the condition for each point.
  condition = (vrx * rpx + vzx * zpx) < A * jnp.abs(vzx * rpx - vrx * zpx)

  # Check if the condition is true for any point in a row.
  return jnp.logical_not(jnp.any(jnp.logical_and(condition, mask), axis=2))
