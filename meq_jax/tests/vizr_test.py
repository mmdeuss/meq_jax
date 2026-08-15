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

"""Tests for vizr."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import vizr
from meqpy import octave_utils
import numpy as np

# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)

# pylint: disable=invalid-name


class VizrTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._oct = octave_utils.create_meq_oct2py_instance()

  def _check_vs_octave(self, Fx, Opy, ry, iry, rBt, drx, dzx):
    """Compares vizrmex between Octave and JAX.

    Fx and Opy are in MATLAB orientation, Fx(z, r); the JAX implementation
    takes the transposed arrays.
    """
    # oct2py converts integer arrays to double by default, but vizrmex
    # insists on an int8 OPY, so cast on the Octave side.
    self._oct.push(
        ['Fx_t', 'Opy_t', 'ry_t', 'iry_t', 'rBt_t', 'drx_t', 'dzx_t'],
        [Fx, Opy.astype(np.float64), ry, iry, rBt, drx, dzx])
    self._oct.eval(
        '[Wp_t, Ft0_t, Vp_t] = '
        'vizrmex(Fx_t, int8(Opy_t), ry_t, iry_t, rBt_t, drx_t, dzx_t);')
    Wp_oct = self._oct.eval('Wp_t;', nout=1)
    Ft0_oct = self._oct.eval('Ft0_t;', nout=1)
    Vp_oct = self._oct.eval('Vp_t;', nout=1)
    Wp, Ft0, Vp = jax.jit(vizr.vizrmex)(
        jnp.asarray(Fx.T), jnp.asarray(Opy.T), ry, iry, rBt, drx, dzx)
    # The JAX version sums the integrand in a different order than the C
    # loop, so the results are not bitwise identical.
    np.testing.assert_allclose(Wp, Wp_oct, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Ft0, Ft0_oct, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Vp, Vp_oct, rtol=1e-12, atol=1e-12)
    return np.asarray(Wp), np.asarray(Ft0), np.asarray(Vp)

  @parameterized.named_parameters(
      ('circular', 1),
      ('diverted', 2),
      ('diverted2', 3),
      ('squashed', 5),
      ('doublet', 82),
      ('droplets', 84),
  )
  def test_vizr_vs_matlab(self, shot):
    """Compares volume integrals on fgs equilibria against vizrmex."""
    oct = self._oct
    oct.eval(f"[L,LX] = fgs('ana', {shot}, 0);")
    Fx = np.asarray(oct.eval('LX.Fx;', nout=1))  # (nzx, nrx)
    Opy = np.asarray(oct.eval('LX.Opy;', nout=1))  # (nzx-2, nrx-2) int8
    ry = np.squeeze(oct.eval('L.ry;', nout=1))
    iry = np.squeeze(oct.eval('L.iry;', nout=1))
    rBt = oct.eval('LX.rBt;', nout=1)[0, 0]
    drx = oct.eval('L.drx;', nout=1)[0, 0]
    dzx = oct.eval('L.dzx;', nout=1)[0, 0]
    nD = int(np.max(Opy))
    self.assertGreaterEqual(nD, 1)

    # Integrals over all full domains, as in meqint.m: the raw indicized Opy
    # is passed directly and everything with Opy > 0 counts as plasma.
    Wp, _, Vp = self._check_vs_octave(Fx, Opy, ry, iry, rBt, drx, dzx)
    # Sanity check against the values stored by fgs.
    np.testing.assert_allclose(Wp, oct.eval('LX.Wp;', nout=1)[0, 0], rtol=1e-9)
    np.testing.assert_allclose(Vp, oct.eval('LX.Vp;', nout=1)[0, 0], rtol=1e-9)

    # Integrals over each individual domain, as in meqagconfun.m
    # (vizrmex(Fx, int8(Opy==iD), ...)); the JAX version receives the
    # boolean array Opy == iD.
    for iD in range(1, nD + 1):
      self._check_vs_octave(Fx, Opy == iD, ry, iry, rBt, drx, dzx)

  def test_vizr_random(self):
    """Random flux map and domain mapping."""
    prng = jax.random.PRNGKey(0)
    prng_fx, prng_opy, prng_ry = jax.random.split(prng, 3)

    nzx, nrx = 12, 15
    Fx = np.asarray(jax.random.normal(prng_fx, (nzx, nrx)))
    Opy = np.asarray(
        jax.random.randint(prng_opy, (nzx - 2, nrx - 2), 0, 4), dtype=np.int8)
    ry = np.asarray(
        jax.random.uniform(prng_ry, (nrx - 2,), minval=0.6, maxval=1.1))
    iry = 1.0 / ry
    rBt = 1.43
    drx, dzx = 0.021, 0.046

    self._check_vs_octave(Fx, Opy, ry, iry, rBt, drx, dzx)
    for iD in range(1, 4):
      self._check_vs_octave(Fx, Opy == iD, ry, iry, rBt, drx, dzx)

  def test_vizr_empty_domain(self):
    """No plasma at all yields zero integrals."""
    nzx, nrx = 10, 11
    Fx = np.ones((nzx, nrx))
    Opy = np.zeros((nzx - 2, nrx - 2), dtype=np.int8)
    ry = np.linspace(0.7, 1.1, nrx - 2)
    iry = 1.0 / ry
    Wp, Ft0, Vp = self._check_vs_octave(Fx, Opy, ry, iry, 1.0, 0.02, 0.05)
    np.testing.assert_array_equal([Wp, Ft0, Vp], np.zeros(3))


if __name__ == '__main__':
  absltest.main()

# pylint: enable=invalid-name
