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

"""Test for meqintQ.py."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from meq_jax._src import meqintQ
from meq_jax._src import types
from meqpy import meqpy_impl
import numpy as np


# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)


# pylint: disable=invalid-name
class PlasmaDomainTest(parameterized.TestCase):

  def test_meqintQ(self):
    meqpy = meqpy_impl.MeqPy()

    # Use the integrals_test function to get a circular equilibrium L and LY
    # Then follow the logic in integrals_test.test_meqintQ
    # Note: meq/tests/integrals_test.m inherits matlab.unittest.TestCase,
    # which Octave cannot load. We instead call the verbatim copy of the
    # static helper meq_test.getCircularEquilibrium from the octave_compat
    # shim (see conftest.py / octave_compat/meq_test.m).
    meqpy.octave_eval(r"""
    sIp = 1
    L = fbt('ana',1,[],'pql',0.1);

    %% Fake circular equilibrium
    r0 = 1.001; z0 = 0;
    FA = sIp; FB = 0;
    rBt = 1; bp=1; qA = 1.5;
    [L,LY] = meq_test.getCircularEquilibrium(L,r0,z0,FA,FB,rBt);
    LY.qA = qA;
    LY.Wk = bp*(1.5e-7*pi*L.P.r0*LY.Ip.^2);

    % plasma current from basis function coefficients
    [~,TpDg,ITpDg] = L.bfct(1,L.bfp,LY.Fx,LY.FA,LY.FB,LY.Opy,L.ry,L.iry);

    %% Coefficient scaling
    ag = zeros(L.ng,1);
    [res, ~,~,dresdag] = ...
      meqagcon(L,LY,LY.FA,LY.FB,LY.rA,LY.dr2FA,LY.dz2FA,LY.drzFA,ag,LY.Fx,LY.Opy,TpDg,ITpDg);
    ag = ag-dresdag\res;

    % Directly check corresponding with matlab code.
    % [Ip,Wk,Wp,~,~,~,Vp,Ft,Ft0] = ...
    %   meqint(L.fPg,L.fTg,TpDg,ITpDg,ag,LY.Fx,LY.Opy,L,rBt);

    [IpQ,WkQ,WpQ,VpQ,FtQ,Ft0Q,OpQ] = ...
      meqintQ(L,FA,FB,rBt,ag, LY.Fx, LY.Opy);

    """)
    IpQ_oct = meqpy.octave_eval('IpQ', nout=1)
    WkQ_oct = meqpy.octave_eval('WkQ', nout=1)
    WpQ_oct = meqpy.octave_eval('WpQ', nout=1)
    VpQ_oct = meqpy.octave_eval('VpQ', nout=1)
    FtQ_oct = meqpy.octave_eval('FtQ', nout=1)
    Ft0Q_oct = meqpy.octave_eval('Ft0Q', nout=1)
    OpQ_oct = meqpy.octave_eval('OpQ', nout=1)

    def _process_int(name):
      return int(meqpy.octave_eval(name, nout=1).item())

    def _process_float(name):
      return float(meqpy.octave_eval(name, nout=1).item())

    def _process_1D(name):
      attr = meqpy.octave_eval(name).squeeze().T
      return jax.numpy.atleast_1d(attr)

    def _process_2D(name):
      attr = meqpy.octave_eval(name).squeeze().T
      if attr.ndim == 1:
        return jax.numpy.array(attr[None])
      else:
        return jax.numpy.array(attr)

    bfp = _process_1D('L.bfp;')
    Bfp = types.BfpData(nP=int(bfp[0]), nT=int(bfp[1]))

    L_meqintq = types.StaticData(
        bfp=Bfp,
        fPg=_process_1D('L.fPg;'),
        fTg=_process_1D('L.fTg;'),
        TDg=_process_2D('L.TDg'),
        idsx=_process_float('L.idsx'),
        drx=_process_float('L.drx'),
        dsx=_process_float('L.dsx'),
        dzx=_process_float('L.dzx;'),
        ny=_process_int('L.ny;'),
        nD=_process_int('L.nD;'),
        pQ=_process_1D('L.pQ;'),
        rry=_process_2D('L.rry;'),
    )

    fast_meqintQ = jax.jit(
        functools.partial(meqintQ.meqintQ, Bfp=Bfp, L_ny=L_meqintq.ny)
    )

    IpQ, WkQ, WpQ, VpQ, FtQ, Ft0Q, OpQ = fast_meqintQ(
        L=L_meqintq,
        F0=jax.numpy.atleast_1d(_process_float('FA')),
        F1=jax.numpy.atleast_1d(_process_float('FB')),
        rBt=_process_float('rBt'),
        ag=meqpy.octave_eval('ag', nout=1),  # leave as 2D
        Fx=_process_2D('LY.Fx'),
        Opy=_process_2D('LY.Opy'),
    )
    np.testing.assert_allclose(IpQ, IpQ_oct, atol=1e-13)
    np.testing.assert_allclose(WkQ, WkQ_oct, atol=1e-13)
    np.testing.assert_allclose(WpQ, WpQ_oct, atol=1e-13)
    np.testing.assert_allclose(VpQ, VpQ_oct, atol=1e-13)
    np.testing.assert_allclose(FtQ, FtQ_oct, atol=1e-13)
    np.testing.assert_allclose(Ft0Q, Ft0Q_oct, atol=1e-13)
    np.testing.assert_allclose(OpQ, OpQ_oct, atol=1e-13)


if __name__ == '__main__':
  absltest.main()
