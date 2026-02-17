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
fun = @asxymex;
x0 = {LX.Fx,L.G.zx,L.G.rx,L.P.dasm,L.dzx,L.drx,L.idzx,L.idrx,L.Oasx,L.dimw};

% Compute jacobians
[zA,rA,FA,~,~,~,ixA,zX,rX,FX,~,~,~,ixX,~] = fun(x0{:});

% F0 sizes
nA = numel(ixA);
nX = numel(ixX);

% Compute other jacobians for axes
[dFAdFx , dzAdFx , drAdFx , ddz2FAdFx , ddr2FAdFx , ddrzFAdFx ] = ...
  asxyJac(LX.Fx, L.dzx, L.drx, L.idzx, L.idrx, ixA);

% Compute other jacobians for saddle points
[dFXdFx , dzXdFx , drXdFx , ddz2FXdFx , ddr2FXdFx , ddrzFXdFx ] = ...
  asxyJac(LX.Fx, L.dzx, L.drx, L.idzx, L.idrx, ixX);
"""


class AsxyJacTest(parameterized.TestCase):

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
  def test_asxyjac_vs_matlab(self, shot):
    """Compare against asxyJac.m."""
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

    _, _, _, _, _, _, ixA, _, _, _, _, _, _, ixX, _ = asxy.asxy(
        Fx, zx, rx, dc, dzx, drx, idzx, idrx, Lxy, dimw)

    ixA = ixA[np.isfinite(ixA)].astype(np.int32)
    ixX = ixX[np.isfinite(ixX)].astype(np.int32)

    dFAdFx, dzAdFx, drAdFx, ddz2FAdFx, ddr2FAdFx, ddrzFAdFx = asxy.jac(
        Fx, zx, rx, dzx, drx, idzx, idrx, ixA)

    dFXdFx, dzXdFx, drXdFx, ddz2FXdFx, ddr2FXdFx, ddrzFXdFx = asxy.jac(
        Fx, zx, rx, dzx, drx, idzx, idrx, ixX)

    self._oct.eval(_OCTAVE_CMD)

    dFAdFx_ = self._oct.eval('dFAdFx;', nout=1).todense()
    dzAdFx_ = self._oct.eval('dzAdFx;', nout=1).todense()
    drAdFx_ = self._oct.eval('drAdFx;', nout=1).todense()
    ddz2FAdFx_ = self._oct.eval('ddz2FAdFx;', nout=1).todense()
    ddr2FAdFx_ = self._oct.eval('ddr2FAdFx;', nout=1).todense()
    ddrzFAdFx_ = self._oct.eval('ddrzFAdFx;', nout=1).todense()
    dFXdFx_ = self._oct.eval('dFXdFx;', nout=1).todense()
    dzXdFx_ = self._oct.eval('dzXdFx;', nout=1).todense()
    drXdFx_ = self._oct.eval('drXdFx;', nout=1).todense()
    ddz2FXdFx_ = self._oct.eval('ddz2FXdFx;', nout=1).todense()
    ddr2FXdFx_ = self._oct.eval('ddr2FXdFx;', nout=1).todense()
    ddrzFXdFx_ = self._oct.eval('ddrzFXdFx;', nout=1).todense()

    np.testing.assert_allclose(dFAdFx_, dFAdFx, atol=1e-13)
    np.testing.assert_allclose(dzAdFx_, dzAdFx, atol=1e-13)
    np.testing.assert_allclose(drAdFx_, drAdFx, atol=1e-13)
    np.testing.assert_allclose(ddz2FAdFx_, ddz2FAdFx, atol=1e-13)
    np.testing.assert_allclose(ddr2FAdFx_, ddr2FAdFx, atol=1e-13)
    np.testing.assert_allclose(ddrzFAdFx_, ddrzFAdFx, atol=1e-13)
    np.testing.assert_allclose(dFXdFx_, dFXdFx, atol=1e-13)
    np.testing.assert_allclose(dzXdFx_, dzXdFx, atol=1e-13)
    np.testing.assert_allclose(drXdFx_, drXdFx, atol=1e-13)
    np.testing.assert_allclose(ddz2FXdFx_, ddz2FXdFx, atol=1e-13)
    np.testing.assert_allclose(ddr2FXdFx_, ddr2FXdFx, atol=1e-13)
    np.testing.assert_allclose(ddrzFXdFx_, ddrzFXdFx, atol=1e-13)

  @parameterized.named_parameters(
      ('circular', 1),
      ('diverted', 2),
      ('diverted2', 3),
      ('squashed', 5),
      ('doublet', 82),
      ('droplets', 84),
      ('doublet_with_mantle_current', 88),
  )
  def test_asxyjac_analytic(self, shot):
    """Compare against JAX autodiff."""
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

    _, _, _, _, _, _, ixA, _, _, _, _, _, _, ixX, _ = asxy.asxy(
        Fx, zx, rx, dc, dzx, drx, idzx, idrx, Lxy, dimw)

    ixA = ixA[np.isfinite(ixA)].astype(np.int32)
    ixX = ixX[np.isfinite(ixX)].astype(np.int32)

    dFAdFx, dzAdFx, drAdFx, ddz2FAdFx, ddr2FAdFx, ddrzFAdFx = asxy.jac(
        Fx, zx, rx, dzx, drx, idzx, idrx, ixA)

    dFXdFx, dzXdFx, drXdFx, ddz2FXdFx, ddr2FXdFx, ddrzFXdFx = asxy.jac(
        Fx, zx, rx, dzx, drx, idzx, idrx, ixX)

    def fn(f):
      output = asxy.six_pt_interpolant(f, zx, rx, dzx, drx, idzx, idrx)
      _, _, _, _, _, _, Fs, _, _, zs, rs, dz2fs, dr2fs, drzfs = output
      return Fs, zs, rs, dz2fs, dr2fs, drzfs

    dFsdFx, dzSdFx, drSdFx, ddz2FSdFx, ddr2FSdFx, ddrzFSdFx = jax.jacfwd(fn)(Fx)
    dFsdFx = np.reshape(dFsdFx, 2*(np.size(Fx),))
    dzSdFx = np.reshape(dzSdFx, 2*(np.size(Fx),))
    drSdFx = np.reshape(drSdFx, 2*(np.size(Fx),))
    ddz2FSdFx = np.reshape(ddz2FSdFx, 2*(np.size(Fx),))
    ddr2FSdFx = np.reshape(ddr2FSdFx, 2*(np.size(Fx),))
    ddrzFSdFx = np.reshape(ddrzFSdFx, 2*(np.size(Fx),))

    np.testing.assert_allclose(dFAdFx, dFsdFx[ixA-1], atol=1e-13)
    np.testing.assert_allclose(dzAdFx, dzSdFx[ixA-1], atol=1e-13)
    np.testing.assert_allclose(drAdFx, drSdFx[ixA-1], atol=1e-13)
    np.testing.assert_allclose(ddz2FAdFx, ddz2FSdFx[ixA-1], atol=1e-13)
    np.testing.assert_allclose(ddr2FAdFx, ddr2FSdFx[ixA-1], atol=1e-13)
    np.testing.assert_allclose(ddrzFAdFx, ddrzFSdFx[ixA-1], atol=1e-13)

    np.testing.assert_allclose(dFXdFx, dFsdFx[ixX-1], atol=1e-13)
    np.testing.assert_allclose(dzXdFx, dzSdFx[ixX-1], atol=1e-13)
    np.testing.assert_allclose(drXdFx, drSdFx[ixX-1], atol=1e-13)
    np.testing.assert_allclose(ddz2FXdFx, ddz2FSdFx[ixX-1], atol=1e-13)
    np.testing.assert_allclose(ddr2FXdFx, ddr2FSdFx[ixX-1], atol=1e-13)
    np.testing.assert_allclose(ddrzFXdFx, ddrzFSdFx[ixX-1], atol=1e-13)


if __name__ == '__main__':
  absltest.main()

# pylint: enable=invalid-name
