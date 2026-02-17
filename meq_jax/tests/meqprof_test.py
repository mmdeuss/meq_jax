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

"""Tests for meqprof."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

from meq_jax._src import meqprof
from meq_jax._src import types
from meqpy import octave_utils


# pylint: disable=invalid-name
jax.config.update("jax_enable_x64", True)


class MeqprofTest(parameterized.TestCase):

  @parameterized.product(shot=[1], smalldia=[True])  # 2, 3, 5],
  def test_meqprof(self, shot: int, smalldia: bool):
    """Test the meqprof code against the Matlab implementation and jacobian."""
    oct2py = octave_utils.create_meq_oct2py_instance()
    oct2py.eval(f"""
      [L, LX] = fgs('ana', {shot}, 0, 'debug', 0, 'tol', 1e-6);
      FN = L.pQ.^2;
      """)

    LX = oct2py.pull("LX")
    F0 = LX["F0"].item()
    F1 = LX["F1"].item()
    ag = LX["ag"].squeeze()
    rBt = LX["rBt"].item()
    FN = oct2py.pull("FN").squeeze()
    idsx = oct2py.eval("L.idsx;").item()
    [nP, nT] = oct2py.eval("L.bfp;").squeeze().astype(int)
    Bfp = types.BfpData(nP=int(nP), nT=int(nT))
    meqprof_jit = jax.jit(
        functools.partial(meqprof.meqprof, Bfp=Bfp, smalldia=smalldia)
    )
    with self.subTest(name="meqprof"):
      outs = meqprof_jit(ag, FN, F0, F1, rBt, idsx)
      outs_ = oct2py.eval(
          f"""
        [PpQ,TTpQ,PQ,TQ,iTQ,PpQg,TTpQg] = meqprof(L.fPg,L.fTg,LX.ag,FN,LX.F0,...
          LX.F1,LX.rBt,L.bfct,L.bfp,L.idsx,{str(smalldia).lower()});
        """,
          nout=7,
      )
      for x, x_ in zip(outs, outs_):
        self.assertTrue(jnp.allclose(x.squeeze(), x_.squeeze()))

    with self.subTest(name="meqprof_grad"):
      jacs_ = oct2py.eval(
          f"""
        [dPpQdag,dTTpQdag,dPQdag,dTQdag,diTQdag,...
          dPpQdF0,dTTpQdF0,dPQdF0,dTQdF0,diTQdF0, ...
          dPpQdF1,dTTpQdF1,dPQdF1,dTQdF1,diTQdF1] = meqprofJac(L.fPg,L.fTg, ...
          LX.ag,FN,LX.F0,LX.F1,LX.rBt,L.bfct,L.bfp,L.idsx,...
          {str(smalldia).lower()});
        """,
          nout=7,
      )
      dPpQ, dTTpQ, dPQ, dTQ, diTQ, *_ = jax.jacrev(
          meqprof_jit, argnums=(0, 2, 3)
      )(ag, FN, F0, F1, rBt, idsx)
      jacs = []
      # ds will be d / dag, d / dF0, d / dF1
      for ds in zip(dPpQ, dTTpQ, dPQ, dTQ, diTQ):
        jacs.extend(ds)
      for x, x_ in zip(jacs, jacs_):
        self.assertTrue(jnp.allclose(x.squeeze(), x_.squeeze()))


if __name__ == "__main__":
  absltest.main()
