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

"""Python version of tests/rtci_robust_test.m for rtcics."""

import functools

from absl.testing import absltest
import jax
from meq_jax._src import rtcics
from meq_jax._src import types
from meqpy import octave_utils
import numpy as np


# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)

# Copied from rtci_robust_test.m
_FIELDS_CMD = """
PP = {'icsint',true,'ilim',3};
L = fbt('ana',1,[],'iterq',20,PP{:});

% Custom map with saddle point at rX,zX
rX = 1;
zX = 0;
Fx = (L.rrx-rX).^2 - (L.zzx-zX).^2;
Opy = reshape(int8(Fx(L.lxy)>0 & L.rrx(L.lxy)>1),L.nzy,L.nry); % Plasma on the right hand side only

% Setup contours, r=rA-a*cos(oq), z=zA+a*sin(oq)
rA = 2;
zA = 0.1;
FA = 1; % Indicative value of flux "on-axis"
Opo = int8(1); % Identify origin as plasma domain (-> contours not gaps)
oq = pi/12*(-3:3).'/3; % Narrow range around X-point direction
crq = -cos(oq);
czq =  sin(oq);
cdrq = crq*L.idrx;
cdzq = czq*L.idzx;
F = 0; % Flux level to find (separatrix)

aq0 = 1.2*ones(size(oq)); % Initial guess, outside of plasma domain, past X-point

% Attributes for P
iterq = L.P.iterq;
tolq = L.P.tolq;

% Attributes for G
rx = L.G.rx;
zx = L.G.zx;

% Attributes for L
taur = L.taur;
tauz = L.tauz;
Mr = L.Mr;
Mz = L.Mz;
nrx = L.nrx;
nzx = L.nzx;
idrx = L.idrx;
idzx = L.idzx;
drx = L.drx;
"""

_INPUT_FIELDS = (
    # Attributes for P
    'iterq',
    'tolq',
    # Attributes for G
    'rx',
    'zx',
    # Attributes for L
    'taur',
    'tauz',
    'Mr',
    'Mz',
    'nrx',
    'nzx',
    'idrx',
    'idzx',
    'drx',
    # Input
    'aq0',
    'Fx',
    'F',
    'Opy',
    'rA',
    'zA',
    'Opo',
    'crq',
    'czq',
)


# pylint: disable=invalid-name
class RTCICSRobustTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._oct = octave_utils.create_meq_oct2py_instance()
    cls._oct.eval(_FIELDS_CMD)
    cls._fields = dict(zip(_INPUT_FIELDS, cls._oct.pull(_INPUT_FIELDS)))

  def test_rtcics_robust(self):

    def _process_1D(attr):
      attr = attr.squeeze()
      if attr.ndim == 0:
        return jax.numpy.array([attr])
      else:
        assert attr.ndim == 1
        return jax.numpy.array(attr)

    # Get fields for P
    P = types.ParameterData(
        iterq=int(self._fields['iterq'].item()),
        tolq=float(self._fields['tolq'].item()),
    )

    # Get fields for G
    # Passing transposed as JAX expects row-major.
    G = types.GeometryData(
        rx=_process_1D(self._fields['rx']),
        zx=_process_1D(self._fields['zx']),
    )

    # Get fields from Lfge
    L = types.StaticData(
        G=G,
        P=P,
        taur=_process_1D(self._fields['taur']),
        tauz=_process_1D(self._fields['tauz']),
        Mr=self._fields['Mr'].T,  # 2D
        Mz=self._fields['Mz'].T,  # 2D
        nrx=int(self._fields['nrx'].item()),
        nzx=int(self._fields['nzx'].item()),
        idrx=float(self._fields['idrx'].item()),
        idzx=float(self._fields['idzx'].item()),
        drx=float(self._fields['drx'].item()),
    )

    # Python implementation of rtcics
    a_python, s_python = jax.jit(functools.partial(rtcics.rtcics, L=L))(
        self._fields['aq0'].T,  # 2D
        self._fields['Fx'].T,  # 2D
        _process_1D(self._fields['F']),
        self._fields['Opy'].T,  # 2D
        _process_1D(self._fields['rA']),
        _process_1D(self._fields['zA']),
        _process_1D(self._fields['Opo']),
        _process_1D(self._fields['crq']),
        _process_1D(self._fields['czq']),
    )

    # Original Matlab implementation of rtcics
    a_matlab, s_matlab = self._oct.eval(
        '[a,s] = rtcics(aq0,Fx,0,Opy,rA,zA,Opo,crq,czq,L);', nout=2
    )

    np.testing.assert_allclose(a_python.T, a_matlab, atol=1e-10)
    np.testing.assert_equal(bool(s_python), bool(s_matlab[0, 0]))


if __name__ == '__main__':
  absltest.main()

# pylint: enable=invalid-name
