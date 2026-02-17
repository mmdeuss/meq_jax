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

import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from meq_jax._src import meqpdom
from meq_jax._src import types
from meqpy import octave_utils
import numpy as np


# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)

# pylint: disable=invalid-name

_TASKS = (
    'Vacuum',
    'Lim-X',
    'SND',
    'DND',
    'Double-Snowflake-Minus',
    'Double-Snowflake-Plus',
    'Doublet-div',
    'Doublet-div-nomantle',
    'Triplet-madness',
)
_SIPS = (-1., 1.)
_ILIM_ICSINT = {
    'nolim': (0, 0),
    'standard': (1, 0),
    'refined': (3, 1),
}
_ISADDLS = (0, 1)
_IHOLES = (0, 1)

_TEST_CASES = []
for task_, sIp_, (name, (ilim_, icsint_)), isaddl_, ihole_ in itertools.product(
    _TASKS, _SIPS, _ILIM_ICSINT.items(), _ISADDLS, _IHOLES
):
  _TEST_CASES.append(
      dict(
          testcase_name=
          f'{task_}_sIp{sIp_}_{name}_isaddl{isaddl_}_ihole{ihole_}',
          task=task_,
          sIp=sIp_,
          icsint=icsint_,
          ilim=ilim_,
          isaddl=isaddl_,
          ihole=ihole_,
      )
  )


_SETUP_CMD = """
sIp = {sIp};
icsint = {icsint};
ilim = {ilim};
L = fgs('ana',0,0,'cappav',2,'nc',16,'ac',0.6,...
        'debug',false,'nz',32,'nr',32,'infct',@qintmex,...
        'ilim',ilim,'icsint',icsint);
L.P.ihole = {ihole};
L.dimw = 16;
S = meq_test.generate_flux_map('{task}', sIp);
Fx = S.Fxh(L.rrx,L.zzx);
% Add infinitesimal flux to break symmetry
eps = 1e-10;
Fx = Fx + eps*(1:size(Fx)(2)) + eps*(1:size(Fx)(1))';
"""

_MEQPDOM_CMD = """
[rA,zA,FA,dr2FA,dz2FA,drzFA,rX,zX,FX,dr2FX,dz2FX,drzFX,rB,zB,FB,lB,lX,Opy,F0,F1,status,msg,id,dF0dFx,dF1dFx,ixI] = meqpdom(Fx,sIp,{isaddl},L);
"""


def compare(a, b, atol=1e-3):
  if isinstance(a, chex.Array):
    if a.size == 0:
      assert b.size == 0
    elif a.ndim == 1:
      np.testing.assert_allclose(a, np.squeeze(b), atol=atol)
    elif a.ndim == 2:
      np.testing.assert_allclose(a, b.T, atol=atol)
    else:
      np.testing.assert_allclose(a, b, atol=atol)
  else:
    np.testing.assert_allclose(a, b, atol=atol)


def compare_jac(a, b, L, atol=1e-3):
  a = np.reshape(a, (-1, L.nzx, L.nrx)).transpose((0, 2, 1))
  a = np.reshape(a, (-1, L.nzx * L.nrx))
  np.testing.assert_allclose(a, b, atol=atol)


class MeqpdomTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._oct = octave_utils.create_meq_oct2py_instance()

  @parameterized.named_parameters(_TEST_CASES)
  def test_meqpdom(self, task, sIp, icsint, ilim, isaddl, ihole):
    self._oct.eval(_SETUP_CMD.format(task=task,
                                     sIp=sIp,
                                     icsint=icsint,
                                     ilim=ilim,
                                     ihole=ihole))

    fx = self._oct.eval('Fx;', nout=1).T

    if icsint:
      taur = jnp.squeeze(self._oct.eval('L.taur;', nout=1))
      tauz = jnp.squeeze(self._oct.eval('L.tauz;', nout=1))
      taul = jnp.squeeze(self._oct.eval('L.taul;', nout=1))
      Mr = self._oct.eval('L.Mr;', nout=1).T
      Mz = self._oct.eval('L.Mz;', nout=1).T
      Ml = self._oct.eval('L.Ml;', nout=1).T
    else:
      taur = None
      tauz = None
      taul = None
      Mr = None
      Mz = None
      Ml = None

    L = types.StaticData(
        G=types.GeometryData(rx=jnp.squeeze(self._oct.eval('L.G.rx;', nout=1)),
                             zx=jnp.squeeze(self._oct.eval('L.G.zx;', nout=1)),
                             rl=jnp.squeeze(self._oct.eval('L.G.rl;', nout=1)),
                             zl=jnp.squeeze(self._oct.eval('L.G.zl;', nout=1)),
                             nl=int(self._oct.eval('L.G.nl;', nout=1).item())),
        P=types.ParameterData(
            itercs=int(self._oct.eval('L.P.itercs;', nout=1).item()),
            tolcs=self._oct.eval('L.P.tolcs;', nout=1).item(),
            xdoma=self._oct.eval('L.P.xdoma;', nout=1).item(),
            dasm=int(self._oct.eval('L.P.dasm;', nout=1).item()),
            icsint=icsint,
            ihole=ihole,
            ilim=ilim),
        dzx=self._oct.eval('L.dzx;', nout=1).item(),
        drx=self._oct.eval('L.drx;', nout=1).item(),
        idzx=self._oct.eval('L.idzx;', nout=1).item(),
        idrx=self._oct.eval('L.idrx;', nout=1).item(),
        taur=taur,
        tauz=tauz,
        taul=taul,
        Mr=Mr,
        Mz=Mz,
        Ml=Ml,
        nD=int(self._oct.eval('L.nD;', nout=1).item()),
        nzy=int(self._oct.eval('L.nzy;', nout=1).item()),
        nry=int(self._oct.eval('L.nry;', nout=1).item()),
        Oasx=self._oct.eval('L.Oasx;', nout=1).T,
        dimw=int(self._oct.eval('L.dimw;', nout=1).item()),
        kxl=jnp.squeeze(self._oct.eval('L.kxl;', nout=1).astype(jnp.int32)),
        clx=jnp.asarray(self._oct.eval('L.clx;', nout=1).T),
        lxy=self._oct.eval('L.lxy;', nout=1).T.astype(jnp.bool_),
        Oly=self._oct.eval('L.Oly;', nout=1).T,
        FN=float(self._oct.eval('L.FN;', nout=1).item()),
        nx=int(self._oct.eval('L.nx;', nout=1).item()),
        nrx=int(self._oct.eval('L.nrx;', nout=1).item()),
        nzx=int(self._oct.eval('L.nzx;', nout=1).item()),
    )

    (
        rA,
        zA,
        FA,
        dr2FA,
        dz2FA,
        drzFA,
        rX,
        zX,
        FX,
        dr2FX,
        dz2FX,
        drzFX,
        rB,
        zB,
        FB,
        lB,
        lX,
        Opy,
        F0,
        F1,
        _,
        dF0dFx,
        dF1dFx,
        ixI,
    ) = jax.jit(functools.partial(meqpdom.meqpdom, isaddl=isaddl, L=L))(fx, sIp)

    # The Matlab implementation of meqpdom.
    (
        rA_,
        zA_,
        FA_,
        dr2FA_,
        dz2FA_,
        drzFA_,
        rX_,
        zX_,
        FX_,
        dr2FX_,
        dz2FX_,
        drzFX_,
        rB_,
        zB_,
        FB_,
        lB_,
        lX_,
        Opy_,
        F0_,
        F1_,
        _,
        _,
        _,
        dF0dFx_,
        dF1dFx_,
        ixI_,
    ) = self._oct.eval(
        _MEQPDOM_CMD.format(isaddl=isaddl), verbose=False, nout=26)

    if not icsint:
      dF0dFx_ = dF0dFx_.todense()
      dF1dFx_ = dF1dFx_.todense()

    compare(zA[np.isfinite(zA)], zA_)
    compare(rA[np.isfinite(rA)], rA_)
    compare(zX[np.isfinite(zX)], zX_)
    compare(rX[np.isfinite(rX)], rX_)
    compare(FA[np.isfinite(FA)], FA_)
    compare(FX[np.isfinite(FX)], FX_)
    compare(dz2FA[np.isfinite(dz2FA)], dz2FA_)
    compare(dr2FA[np.isfinite(dr2FA)], dr2FA_)
    compare(drzFA[np.isfinite(drzFA)], drzFA_)
    compare(dz2FX[np.isfinite(dz2FX)], dz2FX_)
    compare(dr2FX[np.isfinite(dr2FX)], dr2FX_)
    compare(drzFX[np.isfinite(drzFX)], drzFX_)
    compare(rB[np.isfinite(rB)], rB_)
    compare(zB[np.isfinite(zB)], zB_)
    compare(FB[np.isfinite(FB)], FB_)
    compare(lB, lB_)
    compare(lX[np.isfinite(rB)], lX_)
    compare(Opy, Opy_)
    compare(F0, F0_)
    compare(F1, F1_)
    np.testing.assert_allclose(dF0dFx, dF0dFx_, atol=1e-9)
    np.testing.assert_allclose(dF1dFx, dF1dFx_, atol=1e-9)
    compare(ixI[ixI >= 0], ixI_)


if __name__ == '__main__':
  absltest.main()

# pylint: enable=invalid-name
