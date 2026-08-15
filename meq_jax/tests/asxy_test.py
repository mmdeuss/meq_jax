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

"""Tests for asxy.asxy against the Octave asxymex implementation.

Conventions (see also asxyjac_test.py):

* MATLAB stores ``LX.Fx`` as an (nzx, nrx) array, and ``asxymex`` is called
  with x = L.G.zx (the FIRST Matlab dimension) and y = L.G.rx (the SECOND
  Matlab dimension). The JAX implementation takes the transposed (row-major)
  array of shape (nrx, nzx), with rows indexed by rx and columns by zx, so we
  pass ``LX.Fx.T`` (and ``L.Oasx.T``).

* MATLAB returns variable-length column vectors, one entry per axis/saddle
  found, in grid-scan order. The JAX version returns fixed-shape (nrx, nzx)
  arrays padded with -inf at grid cells where nothing was found, so we filter
  finite entries before comparing.

* ``ixA``/``ixX`` are 1-based linear cell indices into the MATLAB
  column-major (nzx, nrx) array, i.e. ix = (j_r - 1)*nzx + i_z with 1-based
  (i_z, j_r). The JAX implementation builds exactly the same value from its
  row-major (nrx, nzx) layout: ix = i_r*nzx + i_z + 1 with 0-based
  (i_r, i_z). The two are identical, so ix can be compared directly, and it
  also provides a canonical sort key (found points are sorted by ix in case
  the two implementations enumerate them in a different order).
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from meq_jax._src import asxy
from meqpy import octave_utils
import numpy as np

# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)

# pylint: disable=invalid-name


_OCTAVE_CMD = """
[zA,rA,FA,dz2FA,dr2FA,drzFA,ixA,zX,rX,FX,dz2FX,dr2FX,drzFX,ixX,stat] = ...
  asxymex(LX.Fx,L.G.zx,L.G.rx,L.P.dasm,L.dzx,L.drx,L.idzx,L.idrx,L.Oasx,L.dimw);
"""

_A_FIELDS = ('zA', 'rA', 'FA', 'dz2FA', 'dr2FA', 'drzFA')
_X_FIELDS = ('zX', 'rX', 'FX', 'dz2FX', 'dr2FX', 'drzFX')


def _pull_vector(oct_instance, name):
  """Pulls a possibly empty Octave column vector as a 1D numpy array."""
  val = oct_instance.eval(f'{name};', nout=1)
  return np.atleast_1d(np.asarray(val, dtype=np.float64).squeeze())


class AsxyTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._oct = octave_utils.create_meq_oct2py_instance()

  @parameterized.named_parameters(
      ('circular', 1),
      ('diverted', 2),
      ('diverted2', 3),
      ('squashed', 5),
      ('doublet', 82),
      ('droplets', 84),
      ('doublet_with_mantle_current', 88),
  )
  def test_asxy_vs_matlab(self, shot):
    """Compare asxy.asxy outputs against the compiled asxymex."""
    self._oct.eval(f"[L,LX] = fgs('ana', {shot}, 0);")

    Fx = self._oct.eval('LX.Fx;', nout=1).T
    zx = np.squeeze(self._oct.eval('L.G.zx;', nout=1))
    rx = np.squeeze(self._oct.eval('L.G.rx;', nout=1))
    dc = self._oct.eval('L.P.dasm;', nout=1)[0, 0]
    dzx = self._oct.eval('L.dzx;', nout=1)[0, 0]
    drx = self._oct.eval('L.drx;', nout=1)[0, 0]
    idzx = self._oct.eval('L.idzx;', nout=1)[0, 0]
    idrx = self._oct.eval('L.idrx;', nout=1)[0, 0]
    Lxy = self._oct.eval('L.Oasx;', nout=1).T
    dimw = self._oct.eval('L.dimw;', nout=1)[0, 0]

    outputs = jax.jit(asxy.asxy)(
        Fx, zx, rx, dc, dzx, drx, idzx, idrx, Lxy, dimw
    )
    (zA, rA, FA, dz2FA, dr2FA, drzFA, ixA,
     zX, rX, FX, dz2FX, dr2FX, drzFX, ixX, stat) = map(np.asarray, outputs)

    self._oct.eval(_OCTAVE_CMD)

    stat_ = bool(self._oct.eval('stat;', nout=1))
    self.assertEqual(bool(stat), stat_)

    for jax_vals, ix, fields, ix_name in (
        ((zA, rA, FA, dz2FA, dr2FA, drzFA), ixA, _A_FIELDS, 'ixA'),
        ((zX, rX, FX, dz2FX, dr2FX, drzFX), ixX, _X_FIELDS, 'ixX'),
    ):
      # Filter out the -inf padding. The mask is shared by all outputs of one
      # kind (axis or saddle).
      mask = np.isfinite(ix)
      ix_jax = ix[mask].astype(np.int64)
      ix_oct = _pull_vector(self._oct, ix_name).astype(np.int64)

      with self.subTest(ix_name=ix_name, part='indices'):
        # Sort both by cell index. The linear cell index is exact (integer)
        # and unique per point so it is a canonical sort key.
        order_jax = np.argsort(ix_jax)
        order_oct = np.argsort(ix_oct)
        np.testing.assert_array_equal(ix_jax[order_jax], ix_oct[order_oct])

      for jax_val, field in zip(jax_vals, fields):
        with self.subTest(ix_name=ix_name, part=field):
          val_jax = jax_val[mask][order_jax]
          val_oct = _pull_vector(self._oct, field)[order_oct]
          # The arithmetic of the six-point interpolant is identical up to
          # the order of floating point operations (e.g. Matlab multiplies by
          # a precomputed 1/h while JAX divides by h), so we allow a small
          # absolute/relative tolerance rather than exact equality.
          np.testing.assert_allclose(val_jax, val_oct, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
  absltest.main()

# pylint: enable=invalid-name
