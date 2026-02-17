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

"""Tests for meqagcon."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import bfct
from meq_jax._src import meqagcon
from meq_jax._src import types
from meqpy import octave_utils
import numpy as np


# Turn on float64 for comparison with Matlab implementation
jax.config.update("jax_enable_x64", True)


# pylint: disable=invalid-name
class MeqagconTest(parameterized.TestCase):

  @parameterized.parameters(1, 2, 3, 5)
  def test_meqagcon(self, shot: int):
    """Test the meqagcon.py code against the Matlab implementation."""
    oct2py = octave_utils.create_meq_oct2py_instance()
    oct2py.eval(f"""
      [L, LX] = fgs('ana', {shot}, 0, 'debug', 0, 'tol', 1e-6);
      LY = LX;
      [~, TpDg, ITpDg] = L.bfct(1,L.bfp,LY.Fx,LY.F0,LY.F1,LY.Opy,L.ry,L.iry);
      """)

    bfp = oct2py.eval("L.bfp;").squeeze().astype(int)

    l = types.StaticData(
        Ip0=oct2py.eval("L.Ip0;").item(),
        fPg=oct2py.eval("L.fPg;").squeeze(),
        fTg=oct2py.eval("L.fTg;").squeeze(),
        xscal=oct2py.eval("L.xscal;").squeeze(),
        ind=types.IndexData(
            ixg=oct2py.eval("L.ind.ixg;").squeeze().astype(int)
        ),
        P=types.ParameterData(
            r0=oct2py.eval("L.P.r0;").item(),
            b0=oct2py.eval("L.P.b0;").item(),
            Ipmin=oct2py.eval("L.P.Ipmin;").item(),
        ),
        bfp=types.BfpData(nP=int(bfp[0]), nT=int(bfp[1])),
        ry=oct2py.eval("L.ry;").squeeze(),
        iry=oct2py.eval("L.iry;").squeeze(),
        dsx=oct2py.eval("L.dsx;").item(),
        dzx=oct2py.eval("L.dzx;").item(),
        drx=oct2py.eval("L.drx;").item(),
        nD=int(oct2py.eval("L.nD;").item()),
    )
    LX = oct2py.pull("LX")
    lx = types.InputData(
        ag=LX["ag"].squeeze(),
        bp=LX["bp"].item(),
        Ip=LX["Ip"].item(),
        li=LX["li"].item(),
        qA=LX["qA"].item(),
        rBt=LX["rBt"].item(),
        Wk=LX["Wk"].item(),
    )
    LY = oct2py.pull("LY")
    ly = types.OutputData(
        Fx=LY["Fx"].squeeze().T,
        F0=jnp.array(LY["F0"].item()),
        F1=jnp.array(LY["F1"].item()),
        Opy=LY["Opy"].squeeze().T,
        ag=lx.ag,
        rA=jnp.atleast_1d(LY["rA"].item()),
        dr2FA=jnp.atleast_1d(LY["dr2FA"].item()),
        dz2FA=jnp.atleast_1d(LY["dz2FA"].item()),
        drzFA=jnp.atleast_1d(LY["drzFA"].item()),
    )
    _, TpDg, _, ITpDg = bfct.bfct1(
        Fx=ly.Fx, F0=ly.F0, F1=ly.F1, Opy=ly.Opy, ry=l.ry, iry=l.iry, Bfp=l.bfp
    )
    fun_names = oct2py.eval("L.agconc(:,3);").squeeze().tolist()
    iDs = oct2py.eval("L.agconc(:,2);").squeeze().tolist()
    iis = (oct2py.eval("L.agconc(:,4);").squeeze() - 1).tolist()
    num_constraints = len(fun_names)
    agconc = [
        types.ConcData(
            fun_name=fn_name,
            domain_id=int(iD.item()),
            lx_name=fn_name,
            index=int(ii.item())
        )
        for fn_name, iD, ii in zip(fun_names, iDs, iis)
    ]
    argnums_to_diff = (0, 1, 6, 7, 9, 10, 2, 3, 4, 5)
    fun = functools.partial(meqagcon.meqagcon, l, lx, agconc, l.nD)
    args = (
        ly.F0,
        ly.F1,
        ly.rA,
        ly.dr2FA,
        ly.dz2FA,
        ly.drzFA,
        ly.ag,
        ly.Fx,
        ly.Opy,
        TpDg,
        ITpDg,
    )
    fun = jax.jit(fun)
    grad_fn = jax.jacfwd(fun, argnums=argnums_to_diff, has_aux=True)
    res, dresdco = fun(*args)
    jacs, _ = grad_fn(*args)
    # Matlab version
    res_matlab, *jacs_matlab = oct2py.eval(
        """
        meqagcon(L,LX,LY.F0,LY.F1,LY.rA,LY.dr2FA,LY.dz2FA,...
          LY.drzFA,LY.ag,LY.Fx,LY.Opy,TpDg,ITpDg);
        """,
        nout=12,
    )
    np.testing.assert_allclose(res, res_matlab.squeeze(), atol=1e-13)
    assert len(jacs) == len(jacs_matlab) - 1
    for i, j in zip(jacs, jacs_matlab[:-1]):
      i = i.reshape((num_constraints, -1))
      np.testing.assert_allclose(i, j, atol=1e-13)
    # NOTE: dresdco is the Jacobian of the residual with respect to the
    # coefficients, it is a diagonal matrix in Matlab.
    dresdco_matlab = jacs_matlab[-1]
    np.testing.assert_allclose(jnp.diag(dresdco), dresdco_matlab, atol=1e-13)


# pylint: enable=invalid-name


if __name__ == "__main__":
  absltest.main()
