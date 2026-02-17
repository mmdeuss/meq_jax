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

"""bint majic implementation."""

import jax.numpy as jnp
import jaxtyping as jt


def bintmex(
    Fx: jt.Float[jt.Array, 'nr nz'],
    k: jt.Integer[jt.Array, 'ni'],
    c: jt.Float[jt.Array, 'ni 4'],
) -> jt.Float[jt.Array, 'ni']:  # Fi.
  """Four point bilinear interpolation.

  See meq/bintmex.m for more details.
  """
  flat = Fx.flatten()
  nz = Fx.shape[1]
  Fi = (
      flat[k] * c[:, 0]
      + flat[k + nz] * c[:, 1]
      + flat[k + nz + 1] * c[:, 2]
      + flat[k + 1] * c[:, 3]
  )
  return Fi.flatten()
