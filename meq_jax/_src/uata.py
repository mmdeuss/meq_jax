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

"""uata majic implementation."""

import jax.numpy as jnp
import jaxtyping as jt


# TODO(pfau): make this work with jnp.float32 too.
def uatamex(
    a: jt.Float[jt.Array, 'n m'],
) -> jt.Float[jt.Array, 'n*(n+1)/2']:  # B.
  """Returns the elements in the upper triangle of A.T @ A.

  Currently, this computes the entire matrix A.T @ A and then extracts
  the upper triangle, which performs double the compute necessary, defeating
  the purpose of a custom MEX implementation. But for now, this is sufficient
  to bootstrap a minimal JAX implementation, which we can improve later.

  See meq/uatamex.m for more details.
  """
  # We use lower indices rather than upper indices because Matlab uses a
  # different convention for indexing. Since the matrix is symmetric, the upper
  # and lower indices are equivalent.
  return jnp.dot(a, a.T)[jnp.tril_indices(a.shape[0])]
