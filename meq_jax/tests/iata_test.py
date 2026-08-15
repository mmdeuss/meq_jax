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

"""Tests for iata."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import iata
from meqpy import octave_utils
import numpy as np

jax.config.update('jax_enable_x64', True)


# pylint: disable=invalid-name
class IataTest(parameterized.TestCase):
  """Tests iatamex against the compiled Octave MEX.

  Conventions: the JAX A1 has shape (n, m1), transposed relative to the
  MATLAB A1 of shape (m1, n), so the MATLAB inv(A1'*A1 + A2'*A2) equals
  the JAX inv(A1 @ A1.T + A2 @ A2.T) (an n x n symmetric matrix, so no
  transposition is needed on the output).

  The MEX special-cases n = 0..4 with hand-unrolled LDL' decompositions
  (with 0-pivot handling for singular matrices) and uses a LAPACK LDL
  factorization for n > 4, whereas the JAX version always uses
  jnp.linalg.inv (LU). The n = 1..4 branches and the n > 4 branch are all
  covered below on nonsingular inputs. The singular-matrix branches
  (which return pseudo-inverse-like results by zeroing 1/0 pivots) and
  the empty-A2 single-argument form of iatamexm are documented behaviors
  that jnp.linalg.inv does not reproduce, so they are not tested here.
  The m2 = 0 case (empty A2 with the correct number of columns) is
  supported by both and is tested.
  """

  @parameterized.named_parameters(
      ('n=1', 1, 5, 3),
      ('n=2', 2, 7, 4),
      ('n=3', 3, 9, 6),
      ('n=4', 4, 10, 8),
      ('n=7_lapack_branch', 7, 20, 12),
      ('n=12_lapack_branch', 12, 40, 25),
      ('m1=1', 3, 1, 8),
      ('empty_A2', 5, 16, 0),
  )
  def test_iatamex_vs_octave(self, n, m1, m2):
    oct2py = octave_utils.create_meq_oct2py_instance()

    prng = jax.random.PRNGKey(0)
    prng_a1, prng_a2 = jax.random.split(prng)

    # Keep m1 + m2 >= n so A1 @ A1.T + A2 @ A2.T is (generically)
    # well-conditioned; the singular branches of the MEX are not
    # reproduced by the JAX implementation (see class docstring).
    A1 = jax.random.normal(prng_a1, (n, m1))
    A2 = jax.random.normal(prng_a2, (n, m2))

    # JAX version
    jit_iatamex = jax.jit(iata.iatamex)
    B_jax = jit_iatamex(A1, A2)

    # MATLAB version (compiled MEX)
    B_matlab = oct2py.iatamex(np.asarray(A1).T, np.asarray(A2).T)

    # Sanity check against the definition.
    S = jnp.dot(A1, A1.T) + jnp.dot(A2, A2.T)
    np.testing.assert_allclose(
        jnp.dot(B_jax, S), np.eye(n), rtol=1e-10, atol=1e-10
    )

    # The MEX uses an LDL' factorization while jnp.linalg.inv uses LU, so
    # the results are not bitwise identical, but for the well-conditioned
    # inputs used here both are accurate to O(eps * cond) << 1e-12.
    np.testing.assert_allclose(
        B_jax, np.asarray(B_matlab).reshape(n, n), rtol=1e-12, atol=1e-12
    )


if __name__ == '__main__':
  absltest.main()
