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

"""Python tests for meqpost.

Note: in practice, icsint is always true.
"""

import dataclasses
import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import meqpost
from meq_jax._src import types
from meqpy import octave_utils
import numpy as np


# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)

_SETUP = """
[L,~,LY] = fbt('ana', {shot}, [], 'gsxe', {gsxe}, 'iterq', 20,'noq', 128, 'pq', linspace(0,1,101), 'icsint', {icsint}, 'ilim', {ilim}, 'iqR', linspace(0,1,20), 'raS', linspace(0,1,20), 'ifield', true, 'ivacuum', true, 'izxoverlap', true, 'izgrid', true);
L.P.ilackner = {ilackner};
"""


# pylint: disable=invalid-name
class MeqpostTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._oct = octave_utils.create_meq_oct2py_instance()

  @parameterized.named_parameters(
      ('limiter_icsint_gsxe1_ilackner1', 1, 'true', 1, 1),
      ('limiter_icsint_gsxe3_ilackner2', 1, 'true', 3, 2),
      ('single_null_icsint_gsxe1_ilackner1', 2, 'true', 1, 1),
      ('single_null_icsint_gsxe3_ilackner2', 2, 'true', 3, 2),
      ('elongated_icsint_gsxe1_ilackner1', 11, 'true', 1, 1),
      ('elongated_icsint_gsxe3_ilackner2', 11, 'true', 3, 2),
      ('doublet_icsint_gsxe1_ilackner1', 82, 'true', 1, 1),
      ('doublet_icsint_gsxe3_ilackner2', 82, 'true', 3, 2),
      ('doublet_with_mantle_icsint_gsxe1_ilackner1', 88, 'true', 1, 1),
      ('doublet_with_mantle_icsint_gsxe3_ilackner2', 88, 'true', 3, 2),
      ('limiter_gsxe1_ilackner1', 1, 'false', 1, 1),
      ('limiter_gsxe3_ilackner2', 1, 'false', 3, 2),
      ('single_null_gsxe1_ilackner1', 2, 'false', 2, 1),
      ('single_null_gsxe3_ilackner2', 2, 'false', 3, 2),
      ('elongated_gsxe1_ilackner1', 11, 'false', 1, 1),
      ('elongated_gsxe3_ilackner2', 11, 'false', 3, 2),
      ('doublet_gsxe1_ilackner1', 82, 'false', 1, 1),
      ('doublet_gsxe3_ilackner2', 82, 'false', 3, 2),
      ('doublet_with_mantle_gsxe1_ilackner1', 88, 'false', 1, 1),
      ('doublet_with_mantle_gsxe3_ilackner2', 88, 'false', 3, 2),
  )
  def test_meqpost(self, shot, icsint, gsxe, ilackner):
    """Tests that Matlab and JAX implementations of meqpost are equivalent."""

    if icsint == 'false':
      ilim = 1
    else:
      ilim = 3
    self._oct.eval(
        _SETUP.format(
            shot=shot, icsint=icsint, ilim=ilim, ilackner=ilackner, gsxe=gsxe
        )
    )

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

    ilackner = _process_int('L.P.ilackner;')
    if ilackner not in (1, 2):
      raise ValueError(f'ilackner={ilackner} is invalid')

    def _process_list(var, tuple_element=False):
      llist = self._oct.eval(var, nout=1)

      res = []
      for l in llist:
        val = l[0].squeeze().astype(int)
        if val.shape == ():
          el = val.item()
          if tuple_element:
            el = (el,)
          res.append(el)
        else:
          res.append(tuple(val))
      return tuple(res)

    # Bfct data
    if shot == 88:  # doublet_with_mantle
      ngi_list = _process_list('L.bfp.PbfgenD.ngi;')
      bfp_list = _process_list('L.bfp.PbfgenD.bfp;')
      iDbf_list = _process_list('L.bfp.PbfgenD.iDbf;', tuple_element=True)

      ng = _process_int('L.bfp.PbfgenD.ng')
      nbf = _process_int('L.bfp.PbfgenD.nbf')
      nD = _process_int('L.bfp.PbfgenD.nD;')

      # Index set
      igm = _process_1D('L.bfp.igm;').astype(int)
      fPg3 = _process_1D('L.bfp.fPg;')
      fPg3 = fPg3[igm - 1]
      fTg3 = _process_1D('L.bfp.fTg;')
      fTg3 = fTg3[igm - 1]

      Bfp = types.BfpData(
          ngi_list=ngi_list,
          bfp_list=bfp_list,
          iDbf_list=iDbf_list,
          ng=ng,
          nbf=nbf,
          nD=nD,
          igm=igm,
          fPg3=fPg3,
          fTg3=fTg3,
          shot=shot,
      )

    elif shot in [82, 84]: # doublet
      ngi_list = _process_list('L.bfp.ngi;')
      bfp_list = _process_list('L.bfp.bfp;')
      # Must be a list of lists
      iDbf_list = _process_list('L.bfp.iDbf;', tuple_element=True)

      ng = _process_int('L.bfp.ng')
      nbf = _process_int('L.bfp.nbf')
      nD = _process_int('L.bfp.nD;')

      Bfp = types.BfpData(
          ngi_list=ngi_list,
          bfp_list=bfp_list,
          iDbf_list=iDbf_list,
          ng=ng,
          nbf=nbf,
          nD=nD,
          shot=shot,
      )

    else:
      bfp = _process_1D('L.bfp;')
      Bfp = types.BfpData(nP=int(bfp[0]), nT=int(bfp[1]))

    P = types.ParameterData(
        gsxe=_process_int('L.P.gsxe;'),
        ilackner=ilackner,
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
        ##### New arguments
        ifield=_process_bool('L.P.ifield;'),
        ivacuum=_process_bool('L.P.ivacuum;'),
        izgrid=_process_bool('L.P.izgrid;'),
        izxoverlap=_process_bool('L.P.izxoverlap;'),
    )

    G = types.GeometryData(
        rW=_process_1D('L.G.rW;'),
        zW=_process_1D('L.G.zW;'),
        nW=_process_int('L.G.nW;'),
        aW=_process_1D('L.G.aW;'),
        rx=_process_1D('L.G.rx;'),
        zx=_process_1D('L.G.zx;'),
        ##### New arguments
        Bma=_process_2D('L.G.Bma;'),
        Bmx=_process_2D('L.G.Bmx;'),
        Bmu=_process_2D('L.G.Bmu;')
        if self._oct.eval('L.G.Bmu;').shape[1] > 0
        else None,
        Mfa=_process_2D('L.G.Mfa;'),
        Mfx=_process_2D('L.G.Mfx;'),
        Mfu=_process_2D('L.G.Mfu;')
        if self._oct.eval('L.G.Mfu;').shape[1] > 0
        else None,
        Tvu=_process_2D('L.G.Tvu;')
        if self._oct.eval('L.G.Tvu;').shape[1] > 0
        else None,
        Tius=_process_1D('L.G.Tius;')
        if len(self._oct.eval('L.G.Tius;')) > 0
        else None,
        lzx=_process_2D('L.G.lzx;'),
        Mza=_process_2D('L.G.Mza;'),
        Mzu=_process_2D('L.G.Mzu;'),
        Mzy=_process_2D('L.G.Mzy;'),
    )

    L = types.StaticData(
        P=P,
        G=G,
        bfp=Bfp,
        ##### Arguments from meqpostq_test.py
        M1q=_process_2D('L.M1q;'),
        M2q=_process_2D('L.M2q;'),
        M3q=_process_2D('L.M3q;'),
        doq=_process_float('L.doq;'),
        nD=_process_int('L.nD;'),
        nQ=_process_int('L.nQ;'),
        nR=_process_int('L.nR;'),
        nS=_process_int('L.nS;'),
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
        ##### Arguments from meqfx_test.py
        Mbe=_process_2D('L.Mbe;'),
        Tbc=_process_2D('L.Tbc;'),
        Tye=_process_2D('full(L.Tye);'),  # undo sparse matrix
        ne=_process_int('L.ne;'),
        nx=_process_int('L.nx;'),
        ny=_process_int('L.ny;'),
        rx=_process_1D('L.rx;'),
        nrx=_process_int('L.nrx;'),
        nzx=_process_int('L.nzx;'),
        nry=_process_int('L.nry;'),
        nzy=_process_int('L.nzy;'),
        Mxe=_process_2D('L.Mxe;'),
        Mxy=_process_1D('L.Mxy;'),
        Mby=_process_1D('L.Mby;'),
        cx=_process_1D('L.cx;'),
        cq=_process_2D('L.cq;'),
        cr=_process_2D('L.cr;'),
        cs=_process_2D('L.cs;'),
        ci=_process_float('L.ci;'),
        co=_process_float('L.co;'),
        #### Arguments from meqintQ_test.py
        fPg=_process_1D('L.fPg;'),
        fTg=_process_1D('L.fTg;'),
        TDg=_process_2D('L.TDg'),
        idsx=_process_float('L.idsx'),
        dsx=_process_float('L.dsx'),
        rry=_process_2D('L.rry;'),
        pQ=_process_1D('L.pQ;'),
        dzx=_process_float('L.dzx;'),
        #### New arguments
        i4pirxdzx=_process_1D('L.i4pirxdzx;'),
        i4pirxdrx=_process_1D('L.i4pirxdrx;'),
        i4pirzdzz=_process_1D('L.i4pirzdzz;'),
        i4pirzdrz=_process_1D('L.i4pirzdrz;'),
        nzz=_process_int('L.nzz;'),
        nrz=_process_int('L.nrz;'),
        zzy=_process_1D('L.zzy;'),
        smalldia=_process_bool('L.smalldia;'),
        lxy=_process_2D('L.lxy;'),
        nn=_process_bool('L.nn;'),
        bfct=self._oct.eval('func2str(L.bfct);'),
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
        ### New arguments
        lB=_process_1D('LY.lB;').astype(bool),
        dr2FX=_process_1D('LY.dr2FX;'),
        dz2FX=_process_1D('LY.dz2FX;'),
        drzFX=_process_1D('LY.drzFX;'),
        ag=_process_1D('LY.ag;'),
        Iy=_process_2D('LY.Iy;'),
        Ia=_process_1D('LY.Ia;'),
    )

    LY_jax = jax.jit(functools.partial(meqpost.meqpost, L=L))(
        LY=LY,
        ag=LY.ag,
        Fx=LY.Fx,
        FA=LY.FA,
        FB=LY.FB,
        rA=LY.rA,
        zA=LY.zA,
        dr2FA=LY.dr2FA,
        dz2FA=LY.dz2FA,
        drzFA=LY.drzFA,
        rB=LY.rB,
        zB=LY.zB,
        lB=LY.lB,
        lX=LY.lX,
        rX=LY.rX,
        zX=LY.zX,
        FX=LY.FX,
        dr2FX=LY.dr2FX,
        dz2FX=LY.dz2FX,
        drzFX=LY.drzFX,
        rBt=LY.rBt,
        Ia=LY.Ia,
        Iu=LY.Iu,
        Iy=LY.Iy,
        Opy=LY.Opy,
        F0=LY.F0,
        F1=LY.F1,
    )

    # _ = jax.jit(functools.partial(meqpost.meqpost, L=L))(LY=LY_jax)

    # First, check that the sizes of the output fields in LY_jax are the same
    # as the inputs.
    LY_jax_names = [x for x in dir(LY_jax) if not x.startswith('_')]

    for k in LY_jax_names:
      v_jax = getattr(LY_jax, k)
      v_input = getattr(LY, k)
      if (
          v_jax is not None
          and v_input is not None
          and isinstance(v_input, jax.Array)
      ):
        assert v_jax.shape == v_input.shape

    # RUn the Matlab implementation, which mutates LY
    LY_matlab = self._oct.eval(
        """
    LY = meqpost(L, LY, LY.ag, LY.Fx, LY.FA, LY.FB, LY.rA, LY.zA, LY.dr2FA, ...
    LY.dz2FA, LY.drzFA, LY.rB, LY.zB, LY.lB, LY.lX, LY.rX, LY.zX, LY.FX, ...
    LY.dr2FX, LY.dz2FX, LY.drzFX, LY.rBt, LY.Ia, LY.Iu, LY.Iy, LY.Opy, LY.F0, LY.F1);
        """,
        nout=1,
    )

    # Test each field in LY_jax
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

            if k == 'qA':
              np.testing.assert_allclose(
                  v_matlab, v_jax[: LY_jax.nA], rtol=1e-8, atol=1e-8,
                  err_msg=f'field {k}',
              )
              continue

            if k in ['PpQg', 'PpQ', 'PQ']:
              # Large values being multipled in multi-domains
              atol = 1e-4
            elif k == 'jtorQ':
              # jtorQ = 2*pi*r0*(PpQ + TTpQ*Q2Q/mu0) is linear in PpQ, so it
              # inherits the multi-domain noise above amplified by 2*pi*r0.
              atol = 1e-3
            else:
              atol = 1e-8
            np.testing.assert_allclose(
                v_matlab[np.isfinite(v_matlab)],
                v_jax[np.isfinite(v_matlab)],
                rtol=1e-8,
                atol=atol,
                err_msg=f'field {k}',
            )
          else:
            np.testing.assert_allclose(
                float(v_matlab.item()), float(v_jax), rtol=1e-8, atol=1e-8,
                err_msg=f'field {k}',
            )

if __name__ == '__main__':
  absltest.main()

# pylint: enable=invalid-name
