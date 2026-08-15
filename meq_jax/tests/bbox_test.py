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

"""Tests for bbox."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import bbox
from meqpy import octave_utils
import numpy as np

# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)

# pylint: disable=invalid-name


class BboxTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._oct = octave_utils.create_meq_oct2py_instance()

  def _check_vs_octave(self, O_matlab, x, y):
    """Compares bboxmex(O, x, y) between Octave and JAX.

    O_matlab is in MATLAB orientation: rows are associated with x, columns
    with y. The JAX implementation takes the transposed array (as everywhere
    in meq_jax) with the coordinates in the same order, so that x is now
    associated with the columns of the transposed array (its 'y' argument)
    and vice versa. Both return the same [x1 y1 x2 y2] bounding box.
    """
    b_oct = np.squeeze(np.asarray(
        self._oct.bboxmex(O_matlab.astype(bool), x, y)))
    b_jax = jax.jit(bbox.bboxmex)(
        jnp.asarray(O_matlab.T), jnp.asarray(x), jnp.asarray(y))
    self.assertEqual(b_jax.shape, (4,))
    # Outputs are exact copies of input coordinates: must match exactly.
    np.testing.assert_array_equal(np.asarray(b_jax), b_oct)

  @parameterized.named_parameters(
      ('circular', 1),
      ('diverted', 2),
      ('diverted2', 3),
      ('squashed', 5),
      ('doublet', 82),
      ('droplets', 84),
  )
  def test_bbox_vs_matlab_equilibrium(self, shot):
    """Bounding box of the plasma domain Opy, as used in fbtt.m."""
    oct = self._oct
    oct.eval(f"[L,LX] = fgs('ana', {shot}, 0);")
    Opy = np.asarray(oct.eval('LX.Opy;', nout=1))  # (nzy, nry) int8
    zy = np.squeeze(oct.eval('L.zy;', nout=1))
    ry = np.squeeze(oct.eval('L.ry;', nout=1))
    nD = int(np.max(Opy))
    self.assertGreaterEqual(nD, 1)

    # Whole plasma domain (b = bboxmex(logical(Opy), L.zy, L.ry)).
    self._check_vs_octave(Opy > 0, zy, ry)
    # Individual domains (Opy == iD), relevant for doublet/droplet shots.
    for iD in range(1, nD + 1):
      self._check_vs_octave(Opy == iD, zy, ry)

  @parameterized.named_parameters(
      ('dense', 0.5),
      ('sparse', 0.05),
  )
  def test_bbox_random(self, density):
    """Random masks and (unsorted) random coordinates."""
    prng = jax.random.PRNGKey(0)
    prng_o, prng_x, prng_y = jax.random.split(prng, 3)

    nx, ny = 17, 23
    O = np.asarray(jax.random.uniform(prng_o, (nx, ny)) < density)
    x = np.asarray(jax.random.uniform(prng_x, (nx,), minval=0.5, maxval=1.5))
    y = np.asarray(jax.random.uniform(prng_y, (ny,), minval=-1.0, maxval=1.0))
    self.assertTrue(O.any())
    self._check_vs_octave(O, x, y)

  def test_bbox_single_point(self):
    """A single selected point yields a degenerate bounding box."""
    nx, ny = 9, 11
    O = np.zeros((nx, ny), dtype=bool)
    O[3, 7] = True
    x = np.linspace(0.6, 1.2, nx)
    y = np.linspace(-0.8, 0.8, ny)
    self._check_vs_octave(O, x, y)
    b = jax.jit(bbox.bboxmex)(jnp.asarray(O.T), x, y)
    np.testing.assert_array_equal(
        np.asarray(b), np.array([x[3], y[7], x[3], y[7]]))

  def test_bbox_empty(self):
    """No selected points returns all zeros (libmeq/bbox.c init case)."""
    nx, ny = 9, 11
    O = np.zeros((nx, ny), dtype=bool)
    x = np.linspace(0.6, 1.2, nx)
    y = np.linspace(-0.8, 0.8, ny)
    self._check_vs_octave(O, x, y)
    b = jax.jit(bbox.bboxmex)(jnp.asarray(O.T), x, y)
    np.testing.assert_array_equal(np.asarray(b), np.zeros(4))


if __name__ == '__main__':
  absltest.main()

# pylint: enable=invalid-name
