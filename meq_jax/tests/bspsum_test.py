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

"""Tests for bspsum against the Octave/genlib bspsum implementation.

bspsum evaluates combinations of B-spline base functions defined by a knot
sequence T, coefficients C and evaluation points X, optionally for the D-th
derivative. The spline order is k = numel(T) - size(C,1).

Conventions: the JAX version takes C transposed ([n_combinations, n_coeffs]
instead of Matlab's [n_coeffs, n_combinations]) and returns the transposed
result ([n_combinations, nX] instead of [nX, n_combinations]). This mirrors
the flcs.py usage: bspsum(taul, L.Ml, s, d, 0, 0) with the JAX (transposed)
L.Ml of shape [2, n_coeffs], where taul are the arclength knots with
boundary multiplicity 4 and interior multiplicity 3.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import bspsum
from meqpy import octave_utils
import numpy as np

# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)


# pylint: disable=invalid-name
def _clamped_knots(breaks, k, interior_multiplicity):
  """Knot sequence with boundary multiplicity k, like L.taul for k=4, m=3."""
  return np.concatenate(
      [np.repeat(breaks[0], k - 1),
       np.repeat(breaks, interior_multiplicity)[
           interior_multiplicity - 1:1 - interior_multiplicity or None],
       np.repeat(breaks[-1], k - 1)])


class BspsumTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._oct = octave_utils.create_meq_oct2py_instance()

  def _compare(self, T, C, X, d):
    """Compares JAX bspsum with the genlib bspsum MEX.

    Args:
      T: knot sequence, shape [nT] (numpy).
      C: coefficients in Matlab layout [n_coeffs, n_combinations] (numpy).
      X: evaluation points, shape [nX] (numpy).
      d: derivative order.
    """
    T = np.asarray(T, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    # JAX version: transposed C, transposed output. d is passed dynamically
    # (traced), matching how flcs.py calls the jitted bspsum.
    V = jax.jit(bspsum.bspsum)(
        jnp.asarray(T), jnp.asarray(C.T), jnp.asarray(X), d, 0, 1)

    V_ = self._oct.bspsum(T.reshape(-1, 1), C, X.reshape(-1, 1), float(d),
                          nout=1)
    V_ = np.asarray(V_).reshape(X.size, C.shape[1])

    # The de Boor recursions are performed in the same order in both
    # implementations, so the results should agree to rounding level.
    np.testing.assert_allclose(V, V_.T, rtol=1e-12, atol=1e-12)
    return V

  @parameterized.named_parameters(
      ('k4_d0', 4, 0), ('k4_d1', 4, 1), ('k4_d2', 4, 2),
      ('k3_d0', 3, 0), ('k3_d1', 3, 1), ('k3_d2', 3, 2),
      ('k2_d0', 2, 0), ('k2_d1', 2, 1),
  )
  def test_random_distinct_knots(self, k, d):
    """Random strictly increasing knots, several combinations at once."""
    prng = jax.random.PRNGKey(k * 10 + d)
    k_t, k_c, k_x = jax.random.split(prng, 3)
    nC, mC = 11, 3
    nT = nC + k
    T = np.sort(np.asarray(jax.random.uniform(k_t, (nT,), maxval=10.0)))
    C = np.asarray(jax.random.normal(k_c, (nC, mC)))
    # X must lie in [T(k), T(end-k+1)] (1-based), i.e. [T[k-1], T[nT-k]].
    X = np.sort(np.asarray(jax.random.uniform(
        k_x, (25,), minval=T[k - 1], maxval=T[nT - k])))
    self._compare(T, C, X, d)

  @parameterized.parameters(0, 1, 2)
  def test_clamped_cubic_taul_structure(self, d):
    """Cubic spline with the same knot structure as L.taul in flcs.

    Boundary knots have multiplicity 4 and interior knots multiplicity 3,
    over the arclength domain [0, 1], with a 2-row coefficient matrix (r, z)
    as in flcs.py bspsum(taul, L.Ml, s, d, 0, 0).
    """
    k = 4
    prng = jax.random.PRNGKey(d)
    k_b, k_c = jax.random.split(prng, 2)
    breaks = np.concatenate(
        [[0.0], np.sort(np.asarray(jax.random.uniform(k_b, (9,)))), [1.0]])
    T = _clamped_knots(breaks, k, 3)
    nC = T.size - k
    C = np.asarray(jax.random.normal(k_c, (nC, 2)))
    # Include the domain endpoints and the exact (repeated) interior knots
    # among the sorted evaluation points.
    X = np.sort(np.concatenate([breaks, np.linspace(0.0, 1.0, 41)]))
    self._compare(T, C, X, d)

  @parameterized.parameters(0, 1, 2)
  def test_scalar_evaluation_point(self, d):
    """A single evaluation point (as used for the X-point arclength sx)."""
    k = 4
    prng = jax.random.PRNGKey(17)
    k_c, k_x = jax.random.split(prng, 2)
    T = _clamped_knots(np.linspace(0.0, 1.0, 8), k, 3)
    nC = T.size - k
    C = np.asarray(jax.random.normal(k_c, (nC, 2)))
    X = np.asarray(jax.random.uniform(k_x, (1,)))
    V = self._compare(T, C, X, d)
    self.assertEqual(V.shape, (2, 1))

  def test_single_combination(self):
    """C with a single column."""
    T = np.arange(-2.0, 13.0)
    C = np.asarray(
        jax.random.normal(jax.random.PRNGKey(23), (12,))).reshape(-1, 1)
    X = np.linspace(1.0, 10.0, 31)
    self._compare(T, C, X, 1)

  def test_doc_example(self):
    """The example from genlib/bspsum.m: bspsum(-2:12, eye(12,15), X)."""
    T = np.arange(-2.0, 13.0)
    C = np.eye(12, 15)
    X = np.linspace(0.0, 10.0, 101)
    self._compare(T, C, X, 0)

  def test_bspsum2_bspsum3_wrappers(self):
    """bspsum2/bspsum3 must agree with the MEX for d=1 and d=0."""
    k = 4
    prng = jax.random.PRNGKey(29)
    k_c, k_x = jax.random.split(prng, 2)
    T = _clamped_knots(np.linspace(0.0, 1.0, 6), k, 3)
    nC = T.size - k
    C = np.asarray(jax.random.normal(k_c, (nC, 3)))
    X = np.sort(np.asarray(jax.random.uniform(k_x, (13,))))

    V2 = jax.jit(bspsum.bspsum2)(
        jnp.asarray(T), jnp.asarray(C.T), jnp.asarray(X), 1.0)
    V2_ = self._oct.bspsum(T.reshape(-1, 1), C, X.reshape(-1, 1), 1.0, nout=1)
    np.testing.assert_allclose(V2, np.asarray(V2_).T, rtol=1e-12, atol=1e-12)

    V3 = jax.jit(bspsum.bspsum3)(
        jnp.asarray(T), jnp.asarray(C.T), jnp.asarray(X))
    V3_ = self._oct.bspsum(T.reshape(-1, 1), C, X.reshape(-1, 1), 0.0, nout=1)
    np.testing.assert_allclose(V3, np.asarray(V3_).T, rtol=1e-12, atol=1e-12)
# pylint: enable=invalid-name


if __name__ == '__main__':
  absltest.main()
