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

"""Tests for meqcde."""
import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

from meq_jax._src import meqcde
from meq_jax._src import meqcdefun
from meq_jax._src import types
from meqpy import octave_utils

# Turn on float64 for comparison with Matlab implementation
jax.config.update("jax_enable_x64", True)


FUN_NAMES = tuple(meqcdefun.meqcdefun().keys())


# pylint: disable=invalid-name
class MeqcdeTest(parameterized.TestCase):

  @parameterized.product(
      fun_name=FUN_NAMES,
      shot=[1, 2, 3, 5],
  )
  def test_meqcde(self, fun_name: str, shot: int):
    """Test the meqcde.py code against the Matlab implementation."""
    oct2py = octave_utils.create_meq_oct2py_instance()
    oct2py.eval(f"""
      [L,LX,LY] = fge('ana',{shot},[0,1e-6], ...
        'usepreconditioner',false,...
        'ssinit',false,...
        'selu','e','nu',30,...
        'cde','{fun_name}',...
        'agcon',{{'bp','qA'}},...
        'wcompa',0,...
        'iLpext',true);
      LX  = meqxk(LX,2);
      LYp = meqxk(LY,1);
      LY  = meqxk(LY,2);
      Ie = [LY.Ia; LY.Iu];
      """)
    l = types.StaticData(
        P=types.ParameterData(
            iLpext=oct2py.eval("L.P.iLpext;").item(),
        ),
        Mye=oct2py.eval("L.Mye;").T if "rigid" in fun_name else None,
        ny=oct2py.eval("L.ny;").item(),
        nD=oct2py.eval("L.nD;").item(),
        ng=oct2py.eval("L.ng;").astype(int).item(),
    )
    LX = oct2py.pull("LX")
    lx = types.InputData(
        ag=LX["ag"].squeeze(),
        Rp=LX["Rp"].squeeze(),
        Ini=LX["Ini"].item(),
        IniD=LX["IniD"].squeeze(),
        Lp=LX["Lp"].squeeze(),
        t=LX["t"].squeeze(),
        Ip=LX["Ip"].item(),
    )
    LY = oct2py.pull("LY")
    LYP = oct2py.pull("LYp")
    ly = types.OutputData(
        Fx=LY["Fx"].squeeze().T,
        F0=LY["F0"][0, :],
        F1=LY["F1"][0, :],
        Opy=LY["Opy"].squeeze().T,
        ag=LY["ag"].squeeze(),
        Iy=LY["Iy"].squeeze().T,
        Ia=LY["Ia"].squeeze().T,
        Iu=LY["Iu"].squeeze().T,
    )
    lyp = types.OutputData(
        Fx=LYP["Fx"].squeeze().T,
        F0=LYP["F0"][0, :],
        F1=LYP["F1"][0, :],
        Opy=LYP["Opy"].squeeze().T,
        ag=LYP["ag"].squeeze(),
        Iy=LYP["Iy"].squeeze().T,
        Ia=LYP["Ia"].squeeze().T,
        Iu=LYP["Iu"].squeeze().T,
        t=LYP["t"].squeeze(),
    )
    fun_names = [fun_name]
    iDs = (oct2py.eval("L.cdec(:,2);")).item().astype(int).squeeze().tolist()
    iis = (oct2py.eval("L.cdec(:,4);").item().astype(int).squeeze()).tolist()
    if isinstance(iDs, int):
      iDs = [iDs]
      iis = [iis]
    cdeconc = [
        types.ConcData(fun_name=fn, domain_id=iD, lx_name="", index=ii)
        for fn, iD, ii in zip(fun_names, iDs, iis)
    ]
    fn = jax.jit(functools.partial(meqcde.meqcde, l, lx, lyp, cdeconc))
    res, *jacs = fn(
        ly.Fx,
        ly.ag,
        ly.Iy,
        ly.F0,
        ly.F1,
        jnp.concatenate([ly.Ia, ly.Iu]),
        ly.Opy,
        )

    res_octave, *jacs_octave = oct2py.eval(
        "meqcde(L,LX,LYp,LY.Fx,LY.ag,LY.Iy,LY.F0,LY.F1,Ie,LY.Opy);",
        nout=15,
        )
    self.assertTrue(jnp.allclose(res, res_octave[0, 0], atol=1e-9))
    self.assertLen(jacs, len(jacs_octave))
    for i, j in zip(jacs, jacs_octave):
      j = j.squeeze()
      i = i.squeeze()
      # NOTE: sometimes the matlab implementation returns flattened
      # jacobians so we need to reshape them to compare them.
      try:
        did_pass = jnp.allclose(i, j)
      except ValueError:
        did_pass = jnp.allclose(i.reshape(j.shape), j)
      self.assertTrue(did_pass)

# pylint: enable=invalid-name

if __name__ == "__main__":
  absltest.main()
