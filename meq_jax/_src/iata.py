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

"""iata majic implementation."""

import jax.numpy as jnp
import jaxtyping as jt


def iatamex(
    A1: jt.Float[jt.Array, 'n m1'],
    A2: jt.Float[jt.Array, 'n m2'],
) -> jt.Float[jt.Array, 'n n']:  # B.
  """Returns the inverse of A1 @ A1.T + A2 @ A2.T

  This does not have any of the special casing of the original MEX
  implementation, but is sufficient to bootstrap a minimal JAX implementation.

  See meq/iatamex.m for more details.
  """
  return jnp.linalg.inv(jnp.dot(A1, A1.T) + jnp.dot(A2, A2.T))
