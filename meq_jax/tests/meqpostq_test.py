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

"""Python tests for meqpostq.

Note: in practice, icsint is always true.
"""

import dataclasses
import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import meqpostq
from meq_jax._src import types
from meqpy import octave_utils
import numpy as np


# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)

_FIELDS_CMD = """
[L,~,LY] = fbt('ana', {shot}, 0, 'iterq', 20,'noq', 128, 'pq', linspace(0,1,101), 'icsint', {icsint}, 'ilim', {ilim}, 'iqR', linspace(0,1,20), 'raS', linspace(0,1,20));
"""


# pylint: disable=invalid-name
class MeqpostqTest(parameterized.TestCase):

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
  def test_meqpostq(self, shot, icsint):
    """Tests that Matlab and JAX implementations of meqpostq are equivalent.

    The Matlab code is taken from meq/meqpostq_test.m.
    """

    if icsint == 'false':
      ilim = 1
    else:
      ilim = 3
    self._oct.eval(_FIELDS_CMD.format(shot=shot, icsint=icsint, ilim=ilim))

    def _process_int(name):
      return int(self._oct.eval(name).item())

    def _process_float(name):
      return float(self._oct.eval(name).item())

    def _process_bool(name):
      return bool(self._oct.eval(name).item())

    def _process_1D(name):
      attr = self._oct.eval(name).squeeze().T
      return jnp.atleast_1d(attr)

    def _process_2D(name):
      attr = self._oct.eval(name).squeeze().T
      if attr.ndim == 1:
        return jnp.array(attr[None])
      else:
        return jnp.array(attr)

    def _process_3D(name):
      attr = self._oct.eval(name).squeeze()
      if attr.ndim == 2:
        attr = attr[..., None]
      return jax.numpy.transpose(attr, (2, 1, 0))

    P = types.ParameterData(
        iterq=_process_int('L.P.iterq;'),
        tolq=_process_float('L.P.tolq;'),
        icsint=_process_bool('L.P.icsint;'),
        nFW=_process_int('L.P.nFW;'),
        naR=_process_int('L.P.naR;'),
        r0=_process_float('L.P.r0;'),
        raS=None
        if len(self._oct.eval('L.P.raS;')) == 0
        else _process_1D('L.P.raS'),
        iqR=None
        if len(self._oct.eval('L.P.iqR;')) == 0
        else _process_1D('L.P.iqR;'),
    )

    G = types.GeometryData(
        rW=_process_1D('L.G.rW;'),
        zW=_process_1D('L.G.zW;'),
        nW=_process_int('L.G.nW;'),
        aW=_process_1D('L.G.aW;'),
        rx=_process_1D('L.G.rx;'),
        zx=_process_1D('L.G.zx;'),
    )

    L = types.StaticData(
        P=P,
        G=G,
        M1q=_process_2D('L.M1q;'),
        M2q=_process_2D('L.M2q;'),
        M3q=_process_2D('L.M3q;'),
        doq=_process_float('L.doq;'),
        nR=_process_int('L.nR;'),
        nS=_process_int('L.nS;'),
        nD=_process_int('L.nD;'),
        nQ=_process_int('L.nQ;'),
        raN=_process_float('L.raN;'),
        noq=_process_int('L.noq;'),
        npq=_process_int('L.npq;'),
        crq=_process_1D('L.crq;'),
        czq=_process_1D('L.czq;'),
        crW=_process_1D('L.crW;'),
        czW=_process_1D('L.czW;'),
        cWx=_process_2D('L.cWx;'),  # 2D
        pq=_process_1D('L.pq;'),
        fq=_process_1D('L.fq;'),
        pinit=_process_float('L.pinit;'),
        c95=_process_1D('L.c95;'),
        i95=_process_1D('L.i95;').astype(int),
        kxW=_process_1D('L.kxW;'),
        nW=_process_int('L.nW;'),
        idrx=_process_float('L.idrx;'),
        idzx=_process_float('L.idzx;'),
        drx=_process_float('L.drx;'),
        liurtemu=_process_bool('L.liurtemu;'),
        dimw=_process_int('L.dimw;'),
        shot=_process_int('LY.shot;'),  # static argument
    )
    if icsint == 'true':
      L = dataclasses.replace(
          L,
          taur=_process_1D('L.taur;'),
          tauz=_process_1D('L.tauz;'),
          Mr=_process_2D('L.Mr;'),  # 2D
          Mz=_process_2D('L.Mz;'),  # 2D
          nrx=_process_int('L.nrx;'),
          nzx=_process_int('L.nzx;'),
      )

    LY = types.OutputData(
        aq=_process_3D('LY.aq;'),  # 3D
        FA=_process_1D('LY.FA;'),
        FB=_process_1D('LY.FB;'),
        FX=_process_1D('LY.FX;'),
        F1=_process_1D('LY.F1;'),
        F0=_process_1D('LY.F0;'),
        lX=_process_1D('LY.lX;').astype(bool),
        Fx=_process_2D('LY.Fx;'),  # 2D
        Opy=_process_2D('LY.Opy;'),  # 2D
        nA=_process_int('LY.nA;'),
        nB=_process_int('LY.nB;'),
        aW=None
        if len(self._oct.eval('LY.aW;')) == 0
        else _process_2D('LY.aW;'),  # 2D. TODO(adedieu): could be 1D.
        nX=_process_int('LY.nX;'),
        rA=_process_1D('LY.rA;'),
        zA=_process_1D('LY.zA;'),
        rB=_process_1D('LY.rB;'),
        zB=_process_1D('LY.zB;'),
        rX=_process_1D('LY.rX;'),
        zX=_process_1D('LY.zX;'),
        dr2FA=_process_1D('LY.dr2FA;'),
        dz2FA=_process_1D('LY.dz2FA;'),
        drzFA=_process_1D('LY.drzFA;'),
        iTQ=_process_2D('LY.iTQ;'),
        PpQ=_process_2D('LY.PpQ;'),
        TTpQ=_process_2D('LY.TTpQ;'),
        TQ=_process_2D('LY.TQ;'),
        IpD=_process_1D('LY.IpD;'),
        Ip=_process_float('LY.Ip;'),
        rBt=_process_float('LY.rBt;'),
    )
    LY_jax = jax.jit(functools.partial(meqpostq.meqpostq, L=L))(LY=LY)

    # This makes sure we can run meqpostq twice
    _ = jax.jit(functools.partial(meqpostq.meqpostq, L=L))(LY=LY_jax)

    # Call matlab implementation of meqpostq. LY is mutated
    LY_matlab = self._oct.eval("""[LY] = meqpostq(L,LY)""", nout=1)

    # Test each field in LY_jax
    LY_jax_names = [x for x in dir(LY_jax) if not x.startswith('_')]

    for k, v_matlab in LY_matlab.items():
      v_matlab = jnp.atleast_1d(v_matlab.squeeze())
      if k in LY_jax_names:
        v_jax = getattr(LY_jax, k)
        if v_jax is not None:
          if isinstance(v_jax, jax.Array):
            v_jax = jnp.atleast_1d(v_jax.squeeze())
            if v_jax.ndim == 2:
              v_jax = v_jax.T
            if v_jax.ndim == 3:
              v_jax = v_jax.transpose((2, 1, 0))
            # 3e-7 seems enough
            np.testing.assert_allclose(v_matlab, v_jax, rtol=1e-8, atol=1e-8)
          else:
            np.testing.assert_allclose(
                float(v_matlab.item()), float(v_jax), rtol=1e-8, atol=1e-8
            )

if __name__ == '__main__':
  absltest.main()

# pylint: enable=invalid-name
