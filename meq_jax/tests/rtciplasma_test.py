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

"""Python tests for rtciplasma and other helper functions."""

import dataclasses
import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from meq_jax._src import rtciplasma
from meq_jax._src import types
from meqpy import octave_utils
import numpy as np


# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)


_FIELDS_CMD = """
[L,~,LY] = fbt('ana', {shot}, 20, 'iterq',20,'noq',128,'pq',linspace(0,1,101),'icsint',{icsint},'ilim',{ilim});

% Attributes for P
iterq = L.P.iterq;
tolq = L.P.tolq;
icsint = L.P.icsint;

% Attributes for G
rx = L.G.rx;
zx = L.G.zx;

% Attributes for L
idrx = L.idrx;
idzx = L.idzx;
drx = L.drx;
liurtemu = L.liurtemu;
crq = L.crq;
czq = L.czq;
noq = L.noq;
npq = L.npq;
nD = L.nD;
pinit = L.pinit;
pq = L.pq;
fq = L.fq;

% Attributes for LY
aq = LY.aq;
FA = LY.FA;
FB = LY.FB;
Fx = LY.Fx;
F0 = LY.F0;
F1 = LY.F1;
rX = LY.rX;
zX = LY.zX;
rA = LY.rA;
zA = LY.zA;
nA = LY.nA;
nB = LY.nB;
Opy=LY.Opy;
shot=LY.shot;
"""


_INPUT_FIELDS = (
    # Attributes for P
    'iterq',
    'tolq',
    'icsint',
    # Attributes for G
    'rx',
    'zx',
    # Attributes for L
    'crq',
    'czq',
    'idrx',
    'idzx',
    'drx',
    'liurtemu',
    'noq',
    'npq',
    'nD',
    'pinit',
    'pq',
    'fq',
    # Attributes for LY
    'aq',
    'F0',
    'F1',
    'FA',
    'FB',
    'Fx',
    'rX',
    'zX',
    'rA',
    'zA',
    'nA',
    'nB',
    'Opy',
    'shot',
)


# pylint: disable=invalid-name
class RTICPlasmaTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._oct = octave_utils.create_meq_oct2py_instance()

  @parameterized.named_parameters(
      ('limiter_icsint', 1, 'true'),
      ('single null_icsint', 2, 'true'),
      ('elongated_icsint', 11, 'true'),
      ('doublet_icsint', 82, 'true'),
      ('doublet_with_mantle_icsint', 88, 'true'),
      ('limiter', 1, 'false'),
      ('single null', 2, 'false'),
      ('elongated', 11, 'false'),
      ('doublet', 82, 'false'),
      ('doublet_with_mantle', 88, 'false'),
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

    def _process_2D(attr):
      attr = attr.squeeze().T
      if attr.ndim == 1:
        return jax.numpy.array(attr[None])
      else:
        return jax.numpy.array(attr)

    def _process_3D(attr):
      attr = attr.squeeze()
      if attr.ndim == 2:
        attr = attr[..., None]
      return jax.numpy.transpose(attr, (2, 1, 0))

    # Get self._fields for P
    P = types.ParameterData(
        iterq=int(self._fields['iterq'].item()),
        tolq=float(self._fields['tolq'].item()),
        icsint=bool(self._fields['icsint'].item()),
    )

    G = types.GeometryData(
        rx=_process_1D(self._fields['rx']),
        zx=_process_1D(self._fields['zx']),
    )

    L = types.StaticData(
        shot=int(self._fields['shot'].item()),
        G=G,
        P=P,
        idrx=float(self._fields['idrx'].item()),
        idzx=float(self._fields['idzx'].item()),
        drx=float(self._fields['drx'].item()),
        liurtemu=bool(self._fields['liurtemu'].item()),
        pinit=float(self._fields['pinit'].item()),
        nD=int(self._fields['nD'].item()),
        noq=int(self._fields['noq'].item()),
        npq=int(self._fields['npq'].item()),
        crq=_process_1D(self._fields['crq']),
        czq=_process_1D(self._fields['czq']),
        pq=_process_1D(self._fields['pq']),
        fq=_process_1D(self._fields['fq']),
    )
    if icsint == 'true':
      L = dataclasses.replace(
          L,
          taur=_process_1D(self._oct.eval('L.taur;')),
          tauz=_process_1D(self._oct.eval('L.tauz;')),
          Mr=_process_2D(self._oct.eval('L.Mr;')),
          Mz=_process_2D(self._oct.eval('L.Mz;')),
          nrx=int(self._oct.eval('L.nrx;').item()),
          nzx=int(self._oct.eval('L.nzx;').item()),
      )

    LY = types.OutputData(
        aq=_process_3D(self._fields['aq']),  # 3D
        F0=_process_1D(self._fields['F0']),
        F1=_process_1D(self._fields['F1']),
        FA=_process_1D(self._fields['FA']),
        FB=_process_1D(self._fields['FB']),
        Fx=_process_2D(self._fields['Fx']),  # 2D
        Opy=_process_2D(self._fields['Opy']),  # 2D
        nA=int(self._fields['nA'].item()),
        nB=int(self._fields['nB'].item()),
        rA=_process_1D(self._fields['rA']),
        zA=_process_1D(self._fields['zA']),
        rX=_process_1D(self._fields['rX']),
        zX=_process_1D(self._fields['zX']),
    )

    # Run Matlab implementation of rtciplasma
    res_matlab = self._oct.eval(
        """
    [rq, zq, aq, rO, zO, crq, czq, nDeff] = rtciplasma(L, LY);
        """,
        nout=8,
    )
    res_matlab_0 = res_matlab[0]
    res_matlab_1 = res_matlab[1]
    res_matlab_2 = res_matlab[2]
    if shot not in [82, 88]:
      res_matlab_0 = res_matlab_0[..., None]
      res_matlab_1 = res_matlab_1[..., None]
      res_matlab_2 = res_matlab_2[..., None]

    # Run Python implementation of rtciplasma
    res_python = jax.jit(functools.partial(rtciplasma.rtciplasma, L=L))(LY=LY)

    # Compare results
    np.testing.assert_allclose(
        res_python[0].transpose((2, 1, 0)), res_matlab_0, atol=1e-8
    )
    np.testing.assert_allclose(
        res_python[1].transpose((2, 1, 0)), res_matlab_1, atol=1e-8
    )
    np.testing.assert_allclose(
        res_python[2].transpose((2, 1, 0)), res_matlab_2, atol=1e-8
    )
    np.testing.assert_allclose(
        res_python[3].transpose((1, 0)), res_matlab[3], atol=1e-8
    )
    np.testing.assert_allclose(
        res_python[4].transpose((1, 0)), res_matlab[4], atol=1e-8
    )
    np.testing.assert_allclose(
        res_python[5].transpose((1, 0)), res_matlab[5], atol=1e-8
    )
    np.testing.assert_allclose(
        res_python[6].transpose((1, 0)), res_matlab[6], atol=1e-8
    )
    np.testing.assert_equal(int(res_python[7]), int(res_matlab[7].item()))

  def test_get_mantle_directions(self):
    """Test get_mantle_directions."""

    # Copied from get_mantle_directions Matlab code in rtciplasma.m
    self._oct.eval(
        """
    rO = rand(20, 30);
    zO = rand(20, 30);
    rX = rand(1, 1);
    zX = rand(1, 1);
    rA = rand(2, 1);
    zA = rand(2, 1);

    rRHS = rX + (zA(1) - zA(2));
    zRHS = zX - (rA(1) - rA(2));
    rLHS = rX - (zA(1) - zA(2));
    zLHS = zX + (rA(1) - rA(2));

    % mask if a point on the separatrix should use the upper lobe axis as
    % origin or the RHS or LHS points (the lower lobe axis as origin is used if
    % all of these are not true)
    maskA1 = (rO - rX) * (rA(1) - rA(2)) + (zO - zX) * (zA(1) - zA(2)) > 0; % upper lobe
    maskRHS = inpolygon(rO, zO, [rA(1);rX;rA(2);rRHS], [zA(1);zX;zA(2);zRHS]); % right side of doublet
    maskLHS = inpolygon(rO, zO, [rA(1);rX;rA(2);rLHS], [zA(1);zX;zA(2);zLHS]); % left side of doublet

    % add axis origin rays according to mask
    cr = (rO - rA(1)) .* maskA1 + (rO - rA(2)) .* ~maskA1;
    cz = (zO - zA(1)) .* maskA1 + (zO - zA(2)) .* ~maskA1;

    % add RHS and LHS origin rays acoording to mask
    cr(maskRHS) = rRHS - rO(maskRHS);
    cz(maskRHS) = zRHS - zO(maskRHS);
    cr(maskLHS) = rLHS - rO(maskLHS);
    cz(maskLHS) = zLHS - zO(maskLHS);

    % norm ray directions
    c_norm = sqrt(cr.*cr + cz.*cz);
    cr = cr ./ c_norm;
    cz = cz ./ c_norm;
    """,
        nout=1,
    )
    cr_matlab = self._oct.pull('cr')
    cz_matlab = self._oct.pull('cz')

    cr_python, cz_python = jax.jit(rtciplasma.get_mantle_directions)(
        rO=jax.numpy.array(self._oct.pull('rO').squeeze()),
        zO=jax.numpy.array(self._oct.pull('zO').squeeze()),
        rX=jax.numpy.array(self._oct.pull('rX').squeeze()),
        zX=jax.numpy.array(self._oct.pull('zX').squeeze()),
        rA=jax.numpy.array(self._oct.pull('rA').squeeze()),
        zA=jax.numpy.array(self._oct.pull('zA').squeeze()),
    )

    np.testing.assert_allclose(
        cr_matlab, cr_python.reshape((20, 30)), rtol=1e-10
    )
    np.testing.assert_allclose(
        cz_matlab, cz_python.reshape((20, 30)), rtol=1e-10
    )

  def test_inpolygon(self):
    """Test inpolygon."""
    # rq_ordered_matlab, zq_ordered_matlab = oct.eval(
    in_matlab = self._oct.eval(
        """
    L = linspace(0,2*pi,6);
    xv = cos(L);
    yv = sin(L);

    xq = randn(250,1);
    yq = randn(250,1);

    in = inpolygon(xq,yq,xv,yv);
    """,
        nout=1,
    )

    in_python = rtciplasma.inpolygon(
        xq=self._oct.pull('xq').squeeze(),
        yq=self._oct.pull('yq').squeeze(),
        xv=self._oct.pull('xv').squeeze(),
        yv=self._oct.pull('yv').squeeze(),
    )

    np.testing.assert_allclose(in_python[:, None], in_matlab, rtol=1e-10)

  def test_x_point_ordering(self):
    """Test x_point_ordering."""

    # Copied from x_point_ordering Matlab code in rtciplasma.m
    self._oct.eval(
        """
    rq = rand(20, 1);
    zq = rand(20, 1);
    rX = rand(1, 1);
    zX = rand(1, 1);
    rA = rand(1, 1);
    zA = rand(1, 1);

    angle = atan2(rq - rA, zq - zA);
    angleX = atan2(rX - rA, zX - zA);
    % have to add epsilon here for the case the X-point itself is part of rq,zq
    [~, order] = sort(mod(angle - angleX + sqrt(eps), 2 * pi));
    rq_ordered = rq(order);
    zq_ordered = zq(order);
    """,
        nout=1,
    )
    rq_ordered_matlab = self._oct.pull('rq_ordered')
    zq_ordered_matlab = self._oct.pull('zq_ordered')

    rq_ordered_python, zq_ordered_python = rtciplasma.x_point_ordering(
        rq=self._oct.pull('rq').squeeze(),
        zq=self._oct.pull('zq').squeeze(),
        rX=self._oct.pull('rX').squeeze(),
        zX=self._oct.pull('zX').squeeze(),
        rA=self._oct.pull('rA').squeeze(),
        zA=self._oct.pull('zA').squeeze(),
    )

    np.testing.assert_allclose(
        rq_ordered_python[:, None], rq_ordered_matlab, rtol=1e-10
    )
    np.testing.assert_allclose(
        zq_ordered_python[:, None], zq_ordered_matlab, rtol=1e-10
    )


if __name__ == '__main__':
  absltest.main()

# pylint: enable=invalid-name
