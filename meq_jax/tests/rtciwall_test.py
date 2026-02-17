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

"""Python test for rtciwall."""

import dataclasses
import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from meq_jax._src import rtciwall
from meq_jax._src import types
from meqpy import octave_utils
import numpy as np


# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)

_FIELDS_CMD = """
[L,~,LY] = fbt('ana', {shot}, 0, 'iterq',20,'noq',128,'pq',linspace(0,1,101), 'icsint', {icsint}, 'ilim', {ilim});

% Attributes for P
iterq = L.P.iterq;
tolq = L.P.tolq;
nFW = L.P.nFW;
icsint = L.P.icsint;

% Attributes for G
rW = L.G.rW;
zW = L.G.zW;
LG_nW = L.G.nW;
LG_aW = L.G.aW;
rx = L.G.rx;
zx = L.G.zx;

% Attributes for L
crW = L.crW;
czW = L.czW;
L_nW = L.nW;
kxW = L.kxW;
cWx = L.cWx;
idrx = L.idrx;
idzx = L.idzx;
drx = L.drx;
liurtemu = L.liurtemu;
dimw = L.dimw;

% Attributes for LY
FB = LY.FB;
Fx = LY.Fx;
FX = LY.FX;
nB = LY.nB;
lX = LY.lX;
nX = LY.nX;
Opy=LY.Opy;

% Random input
LY.aW = rand(L.G.nW,L.P.nFW);
LY_aW = LY.aW;
"""

_INPUT_FIELDS = (
    # Attributes for P
    'iterq',
    'tolq',
    'nFW',
    'icsint',
    # Attributes for G
    'rW',
    'zW',
    'LG_aW',
    'LG_nW',
    'rx',
    'zx',
    # Attributes for L
    'crW',
    'czW',
    'L_nW',
    'kxW',
    'cWx',
    'idrx',
    'idzx',
    'drx',
    'liurtemu',
    'dimw',
    # Attributes for LY
    'FB',
    'Fx',
    'FX',
    'LY_aW',
    'nB',
    'lX',
    'nX',
    'Opy',
)


# pylint: disable=invalid-name
class RTICWallTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._oct = octave_utils.create_meq_oct2py_instance()

  @parameterized.named_parameters(
      ('limiter', 1, 'false'),
      ('limiter_icsint', 1, 'true'),
      ('single null', 2, 'false'),
      ('single null_icsint', 2, 'true'),
      ('elongated', 11, 'false'),
      ('elongated_icsint', 11, 'true'),
      ('doublet', 82, 'false'),
      ('doublet_icsint', 82, 'true'),
      ('doublet_with_mantle', 88, 'false'),
      ('doublet_with_mantle_icsint', 88, 'true'),
  )
  def test_rtci_plasma(self, shot, icsint):
    if icsint == 'false':
      ilim = 1
    else:
      ilim = 3
    self._oct.eval(_FIELDS_CMD.format(shot=shot, icsint=icsint, ilim=ilim))
    self._fields = dict(zip(_INPUT_FIELDS, self._oct.pull(_INPUT_FIELDS)))

    def _process_1D(attr):
      attr = attr.squeeze()
      return jax.numpy.atleast_1d(attr)

    # Get self._fields for P
    P = types.ParameterData(
        iterq=int(self._fields['iterq'].item()),
        tolq=float(self._fields['tolq'].item()),
        icsint=bool(self._fields['icsint'].item()),
        nFW=int(self._fields['nFW'].item()),
    )

    G = types.GeometryData(
        rW=_process_1D(self._fields['rW']),
        zW=_process_1D(self._fields['zW']),
        nW=int(self._fields['LG_nW'].item()),
        aW=_process_1D(self._fields['LG_aW']),
        rx=_process_1D(self._fields['rx']),
        zx=_process_1D(self._fields['zx']),
    )

    L = types.StaticData(
        G=G,
        P=P,
        crW=_process_1D(self._fields['crW']),
        czW=_process_1D(self._fields['czW']),
        cWx=self._fields['cWx'].T,  # 2D
        kxW=_process_1D(self._fields['kxW']),
        nW=int(self._fields['L_nW'].item()),
        idrx=float(self._fields['idrx'].item()),
        idzx=float(self._fields['idzx'].item()),
        drx=float(self._fields['drx'].item()),
        liurtemu=bool(self._fields['liurtemu'].item()),
        dimw=int(self._fields['dimw'].item()),
    )
    if icsint == 'true':
      L = dataclasses.replace(
          L,
          taur=_process_1D(self._oct.eval('L.taur;')),
          tauz=_process_1D(self._oct.eval('L.tauz;')),
          Mr=self._oct.eval('L.Mr;').T,  # 2D
          Mz=self._oct.eval('L.Mz;').T,  # 2D
          nrx=int(self._oct.eval('L.nrx;').item()),
          nzx=int(self._oct.eval('L.nzx;').item()),
      )

    LY = types.OutputData(
        FB=_process_1D(self._fields['FB']),
        FX=_process_1D(self._fields['FX']),
        Fx=self._fields['Fx'].T,  # 2D
        aW=None
        if len(self._fields['LY_aW']) == 0
        else self._fields['LY_aW'].T,  # 2D
        nB=int(self._fields['nB'].item()),
        lX=_process_1D(self._fields['lX']).astype(bool),
        nX=int(self._fields['nX'].item()),
        Opy=self._fields['Opy'].T,  # 2D
    )

    # Python implementation of rtciwall
    aW_python, FW_python = jax.jit(functools.partial(rtciwall.rtciwall, L=L))(
        LY=LY
    )

    # Original Matlab implementation of rtciwall
    aW_matlab, FW_matlab = self._oct.eval('[aW,FW] = rtciwall(L,LY);', nout=2)

    np.testing.assert_allclose(aW_python.T, aW_matlab, atol=1e-10)
    np.testing.assert_allclose(FW_python[None], FW_matlab, atol=1e-10)


if __name__ == '__main__':
  absltest.main()

# pylint: enable=invalid-name
