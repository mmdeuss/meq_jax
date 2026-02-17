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

"""Python test for find_contours."""

import dataclasses
import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from meq_jax._src import find_contours
from meq_jax._src import types
from meqpy import octave_utils
import numpy as np


# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)


# Inspired from tests/rtci_test.m
_FIELDS_CMD = """
[L,~,LY] = fbt('ana', {shot}, 0, 'iterq',20,'noq',128,'pq',linspace(0,1,101), 'icsint', {icsint}, 'ilim', {ilim});


F = LY.FB-L.fq*(LY.FB-LY.FA);


% Attributes for P
iterq = L.P.iterq;
tolq = L.P.tolq;
tokamak = L.P.tokamak;
icsint = L.P.icsint;

% Attributes for G
rx = L.G.rx;
zx = L.G.zx;

% Attributes for L
idrx = L.idrx;
idzx = L.idzx;
drx = L.drx;
liurtemu = L.liurtemu;
crq=L.crq;
czq=L.czq;

% Attributes for LY
Fx = LY.Fx;
Opy = LY.Opy;
Fo = LY.FA;
rO = LY.rA;
zO = LY.zA;
t = LY.t;
shot = LY.shot;

% Input
aq = rand(L.noq,L.npq)*0.5;
Opo = int8(1);
"""

_INPUT_FIELDS = (
    # Attributes for P
    'iterq',
    'tolq',
    'tokamak',
    'icsint',
    # Attributes for G
    'rx',
    'zx',
    # Attributes for L
    'idrx',
    'idzx',
    'drx',
    'liurtemu',
    'crq',
    'czq',
    'shot',
    # Attributes for LY
    'Fx',
    'Opy',
    'Fo',
    'rO',
    'zO',
    # Input
    'aq',
    'Opo',
    'F',
)


# pylint: disable=invalid-name
class FindContoursTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._oct = octave_utils.create_meq_oct2py_instance()

  @parameterized.named_parameters(
      ('limiter', 1, 'false'),
      ('single null', 2, 'false'),
      ('elongated', 11, 'false'),
      ('limiter_icsint', 1, 'true'),
      ('single null_icsint', 2, 'true'),
      ('elongated_icsint', 11, 'true'),
      # Do not add doublet here, as it has its own test in rtciplasma.py
  )
  def test_find_contours(self, shot, icsint):
    if icsint == 'false':
      ilim = 1
    else:
      ilim = 3
    self._oct.eval(_FIELDS_CMD.format(shot=shot, icsint=icsint, ilim=ilim))
    self._fields = dict(zip(_INPUT_FIELDS, self._oct.pull(_INPUT_FIELDS)))
    # note: this only test the case where L.P.icsint is False
    # when L.P.icsint is True, find_contours callss rtcics.py, which is tested
    # in rtcics_test.py

    def _process_1D(attr):
      attr = attr.squeeze()
      return jax.numpy.atleast_1d(attr)

    # Get fields for P
    P = types.ParameterData(
        iterq=int(self._fields['iterq'].item()),
        tolq=float(self._fields['tolq'].item()),
        icsint=bool(self._fields['icsint'].item()),
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
        idrx=float(self._fields['idrx'].item()),
        idzx=float(self._fields['idzx'].item()),
        drx=float(self._fields['drx'].item()),
        liurtemu=bool(self._fields['liurtemu'].item()),
        shot=int(self._fields['shot'].item()),
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
        Fx=self._fields['Fx'].T,  # 2D
        Opy=self._fields['Opy'].T,  # 2D
    )

    # Python implementation of find_contours
    rq_python, zq_python, aq_python = jax.jit(
        functools.partial(find_contours.find_contours, L=L)
    )(
        LY=LY,
        aq=self._fields['aq'].T,  # 2D
        F=_process_1D(self._fields['F']),
        rO=_process_1D(self._fields['rO']),
        zO=_process_1D(self._fields['zO']),
        crq=_process_1D(self._fields['crq']),
        czq=_process_1D(self._fields['czq']),
        Fo=_process_1D(self._fields['Fo']),
        Opo=_process_1D(self._fields['Opo']),
    )

    # Original Matlab implementation of find_contours
    rq_matlab, zq_matlab, aq_matlab = self._oct.eval(
        '[rq,zq,aq] = find_contours(L,LY,F,aq,rO,zO,crq,czq,Fo,Opo);', nout=3
    )

    np.testing.assert_allclose(rq_python.T, rq_matlab, atol=1e-10)
    np.testing.assert_allclose(zq_python.T, zq_matlab, atol=1e-10)
    np.testing.assert_allclose(aq_python.T, aq_matlab, atol=1e-10)


if __name__ == '__main__':
  absltest.main()

# pylint: enable=invalid-name
