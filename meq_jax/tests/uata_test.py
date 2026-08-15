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

"""Tests for uata."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from meq_jax._src import uata
from meqpy import octave_utils
import numpy as np

jax.config.update('jax_enable_x64', True)


# pylint: disable=invalid-name
class UataTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('wide', 10, 40),
      ('tall', 30, 8),
      ('square', 12, 12),
      ('n=1', 1, 25),
      ('m=1', 6, 1),
  )
  def test_uatamex_vs_octave(self, n, m):
    """Tests that uatamex and the compiled Octave MEX are equivalent.

    Conventions: the JAX a has shape (n, m), transposed relative to the
    MATLAB a of shape (m, n), so the MATLAB A'*A equals the JAX A @ A.T
    (an n x n symmetric matrix). MATLAB extracts the upper triangle with
    column-major linear indexing; the JAX version extracts the lower
    triangle with row-major jnp.tril_indices, which visits the same
    elements of a symmetric matrix in the same order.
    """
    oct2py = octave_utils.create_meq_oct2py_instance()

    prng = jax.random.PRNGKey(0)
    a = jax.random.normal(prng, (n, m))

    # JAX version
    jit_uatamex = jax.jit(uata.uatamex)
    b_jax = jit_uatamex(a)

    # MATLAB version (compiled MEX)
    b_matlab = oct2py.uatamex(np.asarray(a).T)

    self.assertEqual(b_jax.shape, (n * (n + 1) // 2,))
    np.testing.assert_allclose(
        b_jax, np.asarray(b_matlab).ravel(), rtol=1e-12, atol=1e-12
    )


if __name__ == '__main__':
  absltest.main()
