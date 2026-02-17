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

"""Python tests for fsgi main and auxiliary functions."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import fsgi
from meq_jax._src import types
from meqpy import octave_utils
import numpy as np


# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)

_FIELDS_CMD = """
[L,~,LY] = fbt('ana', {shot}, 0, 'iterq',20,'noq',128,'pq',linspace(0,1,101), 'icsint', {icsint}, 'ilim', {ilim});  % See fbtxana.m
"""


# pylint: disable=invalid-name
class FsgiTest(parameterized.TestCase):

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
  def test_fsgi(self, shot, icsint):
    if icsint == 'false':
      ilim = 1
    else:
      ilim = 3
    self._oct.eval(_FIELDS_CMD.format(shot=shot, icsint=icsint, ilim=ilim))

    res_matlab = self._oct.eval(
        """
    rq = rand(L.noq,L.npq,L.nD);
    zq = rand(L.noq,L.npq,L.nD);
    aq = rand(L.noq,L.npq,L.nD);
    rO = rand(L.noq,L.nD);
    zO = rand(L.noq,L.nD);

    [Q0Q,Q1Q,Q2Q,Q3Q,Q4Q,iqQ,ItQ,LpQ,rbQ,Q5Q,SlQ,...
    VQ,AQ,IpVQ,FtPVQ,...
    rgeom,zgeom,aminor,epsilon,kappa,delta,deltal,deltau,...
    rrmax,zrmax,rrmin,zrmin,rzmax,zzmax,rzmin,zzmin] = fsgi(L, LY, rq, zq, aq, rO, zO);
    """,
        nout=31,
    )

    def _process_1D(name):
      attr = self._oct.eval(name).squeeze().T
      if shot not in [82, 88]:
        assert attr.ndim == 0
        return jnp.array([attr])
      else:
        return jnp.array(attr)

    def _process_2D(name):
      attr = self._oct.eval(name).squeeze().T
      if shot not in [82, 88]:
        assert attr.ndim == 1
        return jnp.array(attr[None])
      else:
        return jnp.array(attr)

    crq = jnp.array(self._oct.eval('L.crq;').squeeze())
    assert crq.ndim == 1

    L = types.StaticData(
        shot=int(self._oct.eval('LY.shot;').item()),  # static argument
        M1q=jnp.array(self._oct.eval('L.M1q;').squeeze().T),  # 2D
        M2q=jnp.array(self._oct.eval('L.M2q;').squeeze().T),  # 2D
        M3q=jnp.array(self._oct.eval('L.M3q;').squeeze().T),  # 2D
        doq=self._oct.eval('L.doq;').item(),
        nD=int(self._oct.eval('L.nD;').item()),
        nQ=int(self._oct.eval('L.nQ;').item()),
        crq=crq,
    )

    # Special case
    rX = self._oct.eval('LY.rX;').squeeze()
    if shot not in [1, 11]:
      assert rX.ndim == 0
      rX = jnp.array([rX])
    else:
      rX = jnp.array(rX)

    LY = types.OutputData(
        FA=_process_1D('LY.FA;'),
        FB=_process_1D('LY.FB;'),
        F1=_process_1D('LY.F1;'),
        F0=_process_1D('LY.F0;'),
        lX=_process_1D('LY.lX;').astype(bool),
        nA=int(self._oct.eval('LY.nA;').item()),
        nB=int(self._oct.eval('LY.nB;').item()),
        rA=_process_1D('LY.rA;'),
        zA=_process_1D('LY.zA;'),
        rB=_process_1D('LY.rB;'),
        zB=_process_1D('LY.zB;'),
        rX=rX,
        dr2FA=_process_1D('LY.dr2FA;'),
        dz2FA=_process_1D('LY.dz2FA;'),
        drzFA=_process_1D('LY.drzFA;'),
        iTQ=_process_2D('LY.iTQ;'),  # 2D
        PpQ=_process_2D('LY.PpQ;'),  # 2D
        TTpQ=_process_2D('LY.TTpQ;'),  # 2D
        TQ=_process_2D('LY.TQ;'),  # 2D
        IpD=_process_1D('LY.IpD;'),  # 2D
    )

    # Python implementation of fsgi
    rq = self._oct.pull('rq').squeeze()
    zq = self._oct.pull('zq').squeeze()
    aq = self._oct.pull('aq').squeeze()
    rO = self._oct.pull('rO').squeeze()
    zO = self._oct.pull('zO').squeeze()

    if shot not in [82, 88]:
      rq = rq[..., None]
      zq = zq[..., None]
      aq = aq[..., None]
      rO = rO[..., None]
      zO = zO[..., None]

    res_jax = jax.jit(functools.partial(fsgi.fsgi, L=L))(
        LY=LY,
        rq=rq.transpose((2, 1, 0)),
        zq=zq.transpose((2, 1, 0)),
        aq=aq.transpose((2, 1, 0)),
        rO=rO.transpose((1, 0)),
        zO=zO.transpose((1, 0)),
    )

    for idx in range(len(res_matlab)):
      np.testing.assert_allclose(res_matlab[idx], res_jax[idx].T, rtol=1e-8)

  def test_shapmantle(self):
    """Test shapmantle."""
    res_matlab = self._oct.eval(
        """
    noq = 20;
    npq = 10;

    rq = rand(noq, npq);
    zq = rand(noq, npq);
    rS = rand(noq, 1);
    zS = rand(noq, 1);

    rB = rand(1, 1);
    zB = rand(1, 1);

    [rgeom,zgeom,aminor,epsilon,kappa,delta,deltal,deltau,rrmax,zrmax,rrmin,zrmin,rzmax,zzmax,rzmin,zzmin]=shapmex([rS,rq],[zS,zq],rB,zB,nan,nan,nan,nan);
    """,
        nout=16,
    )
    res_jax = jax.jit(fsgi.shapmantle)(
        zq=self._oct.pull('zq').squeeze().T,
        rq=self._oct.pull('rq').squeeze().T,
        rS=self._oct.pull('rS').squeeze(),
        zS=self._oct.pull('zS').squeeze(),
        rB=self._oct.pull('rB').squeeze(),
        zB=self._oct.pull('zB').squeeze(),
    )
    for idx in range(len(res_matlab)):
      np.testing.assert_allclose(
          res_matlab[idx][1:], res_jax[idx][:, None], rtol=1e-10
      )

  def test_cumtrapz(self):
    """Test cumtrapz."""

    Q_matlab = self._oct.eval(
        """
    X = 0:pi/10:pi;
    Y = sin(X');
    Q = cumtrapz(X,Y);
    """,
        nout=1,
    )

    Q_jax = jax.jit(fsgi.cumtrapz)(
        X=self._oct.pull('X').squeeze(),
        Y=self._oct.pull('Y').squeeze(),
    )

    np.testing.assert_allclose(Q_jax[:, None], Q_matlab, rtol=1e-10)

  def test_fsavgmantle(self):
    """Test fsavgmantle on random input."""

    self._oct.eval("""
    noq = 20;
    npq = 10;
    nQ = npq;
    nX = 5;

    rq = rand(noq, npq);
    zq = rand(noq, npq);
    rS = rand(noq, 1);
    zS = rand(noq, 1);
    M3q = rand(npq, npq);
    M2q = rand(noq, noq);
    doq = rand(1, 1);
    FAB = 1;
    VQ = rand(nQ, 1);
    rX = rand(nX, 1);
    lXm = 1;
    rBm = rand(1, 1);
    iTQ = rand(nQ, 1);
    ItS = rand(1, 1);
    LpS = rand(1, 1);
    SlS = rand(1, 1);

    drdpsi = 0.5 * (rq - rS) * M3q / FAB;
    dzdpsi = 0.5 * (zq - zS) * M3q / FAB;
    drdtheta = M2q * rq * 2 * doq;
    dzdtheta = M2q * zq * 2 * doq;
    % grad_psi perp grad_theta => |grad_psi| * dl = det([grad_psi, grad_theta]) = det(J^-1)
    dl = sqrt(drdtheta.^2 + dzdtheta.^2);
    detJ = drdpsi .* dzdtheta - dzdpsi .* drdtheta;
    dpsi = -dl ./ detJ; % minus due to point orientation
    idpsidl = -detJ;

    % useful things
    irq = 1 ./ rq;
    irX = 1 / rX(1);
    % more useful things
    dpsidV          = 1 ./ (2 * pi * sum(rq  .*     idpsidl, 1));
    int_iR_idpsi_dl =                sum(irq .*     idpsidl, 1);
    int_iR_dpsi_dl  =                sum(irq .*  dpsi .* dl, 1);
    int_R_dl        =                sum( rq .*          dl, 1);

    % flux surface averages
    Q1Q = [0,                                                -dpsidV ].'; % -dpsi/dV
    Q0Q = [irX,     2 * pi * sum(          idpsidl, 1) .*     dpsidV ].'; % <1/R>, 1/R * 1/Bp = 2 pi / gradpsi
    Q2Q = [irX*irX, 2 * pi * int_iR_idpsi_dl           .*     dpsidV ].'; % <1/R^2>
    Q3Q = [0,       2 * pi * int_iR_dpsi_dl            .*     dpsidV ].'; % <|grad psi|^2/R^2>
    Q4Q = [0,       2 * pi * sum( rq .* dpsi .* dl, 1) .*     dpsidV ].'; % <|grad psi|^2>
    Q5Q = [0,                int_R_dl                  .* abs(dpsidV)].'; % <|grad psi|/(2*pi)>
    iqQ = [0, -iTQ' ./ int_iR_idpsi_dl].';
    ItQ = [ItS, -int_iR_dpsi_dl / (2 * pi * mu0)].';
    LpQ = [LpS, sum(dl, 1)].';
    SlQ = [SlS, 2 * pi * int_R_dl].';
    rbQ = SlQ ./ (2 * pi * LpQ);

    irX2 = 1/rBm; % r coordinate defining mantle flux is at mantle X point
    Q0Q(end) = irX2;
    Q1Q(end) = 0;
    Q2Q(end) = irX2*irX2;
    Q3Q(end) = 0;
    Q4Q(end) = 0;
    iqQ(end) = 0;
    Q5Q(end) = 0;
    """)

    lXm = bool(self._oct.pull('lXm').squeeze())
    res_jax = jax.jit(functools.partial(fsgi.fsavgmantle, lXm=lXm))(
        rq=self._oct.pull('rq').T,
        zq=self._oct.pull('zq').T,
        rS=self._oct.pull('rS').squeeze(),
        zS=self._oct.pull('zS').squeeze(),
        M3q=self._oct.pull('M3q').T,
        M2q=self._oct.pull('M2q').T,
        doq=self._oct.pull('doq').squeeze(),
        FAB=self._oct.pull('FAB').squeeze(),
        rX=self._oct.pull('rX').squeeze(),
        lXm=bool(self._oct.pull('lXm').squeeze()),
        rBm=self._oct.pull('rBm').squeeze(),
        iTQ=self._oct.pull('iTQ').squeeze(),
        ItS=self._oct.pull('ItS').squeeze(),
        LpS=self._oct.pull('LpS').squeeze(),
        SlS=self._oct.pull('SlS').squeeze(),
    )
    for idx, field in enumerate([
        'Q0Q',
        'Q1Q',
        'Q2Q',
        'Q3Q',
        'Q4Q',
        'iqQ',
        'ItQ',
        'LpQ',
        'rbQ',
        'Q5Q',
        'SlQ',
    ]):
      np.testing.assert_allclose(
          res_jax[idx][:, None], self._oct.pull(field), rtol=1e-10
      )

  def test_fsg2mantle(self):
    """Test fsg2mantle."""

    self._oct.eval(
        """
    rq = rand(20, 10);
    zq = rand(20, 10);

    dr = rq([2:end,1],:) - rq;
    rc = 0.5 * (rq([2:end,1],:) + rq);
    zc = 0.5 * (zq([2:end,1],:) + zq);
    AQ = sum(zc .* dr, 1).';
    VQ = sum(2 * pi * zc .* rc .* dr, 1).';
    """,
        nout=1,
    )
    vq_matlab = self._oct.pull('VQ')
    aq_matlab = self._oct.pull('AQ')

    vq_jax, aq_jax = jax.jit(fsgi.fsg2mantle)(
        rq=self._oct.pull('rq').T,
        zq=self._oct.pull('zq').T,
    )

    np.testing.assert_allclose(vq_jax[:, None], vq_matlab, rtol=1e-10)
    np.testing.assert_allclose(aq_jax[:, None], aq_matlab, rtol=1e-10)

  def test_fsg2mex(self):
    """Test fsg2mex."""

    self._oct.eval(
        """
    noq = 20;
    npq = 10;

    aq = rand(noq, npq);
    rA = rand(1, 1);
    crq = rand(noq, 1);
    doq = rand(1, 1);

    [VQ,AQ] = fsg2mexm(aq,rA,crq,doq);
    """,
        nout=1,
    )
    vq_matlab = self._oct.pull('VQ')
    aq_matlab = self._oct.pull('AQ')

    vq_jax, aq_jax = jax.jit(fsgi.fsg2mex)(
        aq=self._oct.pull('aq').T,
        rA=self._oct.pull('rA').squeeze(),
        crq=self._oct.pull('crq').squeeze(),
        doq=self._oct.pull('doq').squeeze(),
    )

    np.testing.assert_allclose(vq_jax[:, None], vq_matlab, rtol=1e-8)
    np.testing.assert_allclose(aq_jax[:, None], aq_matlab, rtol=1e-8)


if __name__ == '__main__':
  absltest.main()

# pylint: enable=invalid-name
