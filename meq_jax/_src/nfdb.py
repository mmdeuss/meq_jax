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

"""JAX implementation of the NFDB MATLAB function."""

import jax.numpy as jnp


def nfdb(grid: jnp.ndarray) -> jnp.ndarray:
  """Replace the boundary of a grid with the normal diff 4*F(x+dx)-F(x+2dx).

  (Except the corners of the grid).
  For details, see: [J-M. Moret et al. Fus.Eng.Des 2015], Section 4.2.

  Args:
    grid: The grid to process. [nr, nz]

  Returns:
    The processed grid. [2*(nr-2)+2*(nz-2)]
  """
  # 1. Top boundary difference (corresponds to the first C loop)
  # Calculates 4*F[1, 1:-1] - F[2, 1:-1]
  top_boundary = 4.0 * grid[1, 1:-1] - grid[2, 1:-1]

  # 2. Left and right boundary differences (corresponds to the second C loop)
  # Left side: 4*F[1:-1, 1] - F[1:-1, 2]
  left_side = 4.0 * grid[1:-1, 1] - grid[1:-1, 2]
  # Right side: 4*F[1:-1, -2] - F[1:-1, -3]
  right_side = 4.0 * grid[1:-1, -2] - grid[1:-1, -3]

  # The C code interleaves the left and right side results.
  # We can replicate this by stacking them as columns and flattening.
  sides_interleaved = jnp.stack([left_side, right_side], axis=1).flatten()

  # 3. Bottom boundary difference (corresponds to the third C loop)
  # Calculates 4*F[-2, 1:-1] - F[-3, 1:-1]
  bottom_boundary = 4.0 * grid[-2, 1:-1] - grid[-3, 1:-1]

  # 4. Concatenate the results in the same order as the C function
  return jnp.concatenate([top_boundary, sides_interleaved, bottom_boundary])
