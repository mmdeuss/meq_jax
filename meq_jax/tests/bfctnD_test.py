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

"""Test for bfctnD.py."""

import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import bfct_router
from meq_jax._src import types
from meqpy import octave_utils
import numpy as np


# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)

# pylint: disable=invalid-name

_TASKS = {
    'doublet': 82,
    'droplets': 84,
    'doublet_with_mantle_current': 88,
}
_BASIS_IDS = 1, 2, 3, 5, 11, 91

_TEST_CASES = []
for (name, shot), basis_id in itertools.product(_TASKS.items(), _BASIS_IDS):
  _TEST_CASES.append(
      dict(
          testcase_name=f'{name}_basis{basis_id}',
          shot_=shot,
          basis_id_=basis_id,
      )
  )

_OCTAVE_CMD = """[L,~,LY] = fbt('ana', {shot});"""


class BfctNDTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._oct = octave_utils.create_meq_oct2py_instance()

  @parameterized.named_parameters(_TEST_CASES)
  def test_bfct_nD(self, shot_, basis_id_):
    cmd = _OCTAVE_CMD.format(shot=shot_)
    self._oct.eval(cmd)

    def _process_1D(var):
      attr = self._oct.eval(var, nout=1).squeeze().T
      return jnp.atleast_1d(attr)

    def _process_2D(var):
      attr = self._oct.eval(var, nout=1).squeeze().T
      if attr.ndim == 1:
        return jnp.array(attr[None])
      else:
        return jnp.array(attr)

    def _process_int(var):
      return int(self._oct.eval(var).item())

    def _process_float(var):
      return float(self._oct.eval(var).item())

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

    # Shared params
    F0 = _process_1D('LY.F0;')
    F1 = _process_1D('LY.F1;')
    FN = _process_1D('L.pQ(:).^2;')
    Fx = _process_2D('LY.Fx')
    Opy = _process_2D('LY.Opy')
    ag = _process_1D('LY.ag;')
    ry = _process_1D('L.ry;')
    iry = _process_1D('L.iry;')
    ids = _process_float('L.idsx;')

    if shot_ == 88:
      # Doublet with mantle have their own logic, defined in bfdoublet.m
      ngi_list = _process_list('L.bfp.PbfgenD.ngi;')
      bfp_list = _process_list('L.bfp.PbfgenD.bfp;')
      # Must be a list of lists
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
          shot=shot_,
      )

    else:
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
          shot=shot_,
      )

    ################################## BFCT1 ###################################
    if basis_id_ == 1:
      # Matlab
      res_matlab = self._oct.eval(
          """
      [Tyg, TpDg, ITpDg] = L.bfct(1, L.bfp, LY.Fx, LY.F0, LY.F1, LY.Opy, L.ry, L.iry);
          """,
          nout=3,
      )
      Tyg_matlab = res_matlab[0]
      TpDg_matlab = res_matlab[1]
      ITpDg_matlab = res_matlab[2]

      # Jax
      jit_bfct1 = jax.jit(functools.partial(bfct_router.bfct1, Bfp=Bfp))

      Tyg_jax, TpDg_jax, _, ITpDg_jax = jit_bfct1(
          Fx=Fx, F0=F0, F1=F1, Opy=Opy, ry=ry, iry=iry
      )

      # Compare
      np.testing.assert_allclose(Tyg_jax, Tyg_matlab.T, atol=1e-8)
      np.testing.assert_allclose(TpDg_jax, TpDg_matlab.T, atol=1e-8)
      np.testing.assert_allclose(ITpDg_jax, ITpDg_matlab.T, atol=1e-8)

    ################################## BFCT2 ###################################
    if basis_id_ == 2:
      # Matlab
      res_matlab = self._oct.eval(
          """
      [gQg, IgQg] = L.bfct(2, L.bfp, L.pQ(:).^2, LY.F0, LY.F1);
          """,
          nout=2,
      )
      gQg_matlab = res_matlab[0]
      IgQg_matlab = res_matlab[1]

      # Jax
      jit_bfct2 = jax.jit(functools.partial(bfct_router.bfct2, Bfp=Bfp))
      gQg_jax, IgQg_jax = jit_bfct2(FN=FN, F0=F0, F1=F1)

      # Compare
      np.testing.assert_allclose(
          gQg_jax, gQg_matlab.transpose(2, 1, 0), atol=1e-8
      )
      np.testing.assert_allclose(
          IgQg_jax, IgQg_matlab.transpose(2, 1, 0), atol=1e-8
      )

    ################################## BFCT3 ###################################
    if basis_id_ == 3:
      # Matlab
      res_matlab = self._oct.eval(
          """
      [aPpg,aTTpg,aPg,ahqTg] = L.bfct(3, L.bfp, LY.ag, LY.F0, LY.F1, L.fPg, L.fTg, L.idsx);
      """,
          nout=4,
      )
      aPpg_matlab = res_matlab[0]
      aTTpg_matlab = res_matlab[1]
      aPg_matlab = res_matlab[2]
      ahqTg_matlab = res_matlab[3]

      # Jax
      jit_bfct3 = jax.jit(functools.partial(bfct_router.bfct3, Bfp=Bfp))
      aPpg_jax, aTTpg_jax, aPg_jax, ahqTg_jax = jit_bfct3(
          ag=ag, F0=F0, F1=F1, ids=ids
      )

      # Compare
      np.testing.assert_allclose(aPpg_jax, aPpg_matlab[:, 0], atol=1e-8)
      np.testing.assert_allclose(aTTpg_jax, aTTpg_matlab[:, 0], atol=1e-8)
      np.testing.assert_allclose(aPg_jax, aPg_matlab[:, 0], atol=1e-8)
      np.testing.assert_allclose(ahqTg_jax, ahqTg_matlab[:, 0], atol=1e-8)

    ################################## BFCT5 ###################################
    if basis_id_ == 5:
      # Matlab
      res_matlab = self._oct.eval(
          """
      [gA, IgA] = L.bfct(5, L.bfp, LY.Fx, LY.F0, LY.F1);
      """,
          nout=2,
      )
      gA_matlab = res_matlab[0]
      IgA_matlab = res_matlab[1]

      # Jax
      jit_bfct5 = jax.jit(functools.partial(bfct_router.bfct5, Bfp=Bfp))
      gA_jax, IgA_jax = jit_bfct5(F0=F0, F1=F1)

      np.testing.assert_allclose(gA_jax, gA_matlab.T, atol=1e-8)
      np.testing.assert_allclose(IgA_jax, IgA_matlab.T, atol=1e-8)

    ################################## BFCT11 ##################################
    if basis_id_ == 11:
      # Matlab
      res_matlab = self._oct.eval(
          """
      [dTygdFy,dTygdF0,dTygdF1,dITygdF0,dITygdF1] = L.bfct(11, L.bfp, LY.Fx, LY.F0, LY.F1, LY.Opy, L.ry, L.iry);
      """,
          nout=5,
      )
      dTygdFy_matlab = res_matlab[0]
      dTygdF0_matlab = res_matlab[1]
      dTygdF1_matlab = res_matlab[2]
      dITygdF0_matlab = res_matlab[3]
      dITygdF1_matlab = res_matlab[4]

      # Jax
      jit_bfct11 = jax.jit(functools.partial(bfct_router.bfct11, Bfp=Bfp))
      dTygdFy_jax, dTygdF0_jax, dTygdF1_jax, dITygdF0_jax, dITygdF1_jax = (
          jit_bfct11(Fx=Fx, F0=F0, F1=F1, Opy=Opy, ry=ry, iry=iry)
      )

      # Compare
      np.testing.assert_allclose(dTygdFy_jax, dTygdFy_matlab.T, atol=1e-8)
      np.testing.assert_allclose(dTygdF0_jax, dTygdF0_matlab.T, atol=1e-8)
      np.testing.assert_allclose(dTygdF1_jax, dTygdF1_matlab.T, atol=1e-8)
      np.testing.assert_allclose(dITygdF0_jax, dITygdF0_matlab.T, atol=1e-8)
      np.testing.assert_allclose(dITygdF1_jax, dITygdF1_matlab.T, atol=1e-8)

    ################################## BFCT91 ##################################
    if basis_id_ == 91:
      # Matlab
      res_matlab = self._oct.eval(
          """
      [g, Ig] = L.bfct(91, L.bfp, LY.Fx, LY.F0, LY.F1);
          """,
          nout=2,
      )

      g_matlab = res_matlab[0]
      Ig_matlab = res_matlab[1]

      # Jax
      jit_bfct91 = jax.jit(functools.partial(bfct_router.bfct91, Bfp=Bfp))
      g_jax, Ig_jax = jit_bfct91(F=Fx, F0=F0, F1=F1)

      np.testing.assert_allclose(g_jax, g_matlab.T, atol=1e-8, rtol=1e-8)
      np.testing.assert_allclose(Ig_jax, Ig_matlab.T, atol=1e-8)


# pylint: enable=invalid-name


if __name__ == '__main__':
  absltest.main()
