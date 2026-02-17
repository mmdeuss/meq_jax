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

"""Python tests for meqfx."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import meqfx
from meq_jax._src import types
from meqpy import octave_utils
import numpy as np


# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)

_SETUP = """
rand('state', 42);
[L,LX] = fgs('ana', {shot}, [], 'gsxe', {gsxe});
LX.Iu = 1e-4*L.Iu0.*(2*rand(L.G.nu,1)-1);
Ie = [LX.Ia; LX.Iu];
L.P.ilackner = {ilackner};
Fbe = L.Mbe*Ie;
Iyie = reshape(L.Tye*Ie,L.nzy,L.nry);
"""


# pylint: disable=invalid-name
class MeqFxTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._oct = octave_utils.create_meq_oct2py_instance()

  @parameterized.named_parameters(
      ('limiter_gsxe1_ilackner1', 1, 1, 1),
      ('limiter_gsxe1_ilackner2', 1, 1, 2),
      ('limiter_gsxe2_ilackner1', 1, 2, 1),
      ('limiter_gsxe2_ilackner2', 1, 2, 2),
      ('limiter_gsxe3_ilackner1', 1, 3, 1),
      ('limiter_gsxe3_ilackner2', 1, 3, 2),
      ('single_null_gsxe1_ilackner1', 2, 2, 1),
      ('single_null_gsxe1_ilackner2', 2, 2, 2),
      ('single_null_gsxe2_ilackner1', 2, 2, 1),
      ('single_null_gsxe2_ilackner2', 2, 2, 2),
      ('single_null_gsxe3_ilackner1', 2, 3, 1),
      ('single_null_gsxe3_ilackner2', 2, 3, 2),
      ('elongated_gsxe1_ilackner1', 11, 1, 1),
      ('elongated_gsxe1_ilackner2', 11, 1, 2),
      ('elongated_gsxe2_ilackner1', 11, 2, 1),
      ('elongated_gsxe2_ilackner2', 11, 2, 2),
      ('elongated_gsxe3_ilackner1', 11, 3, 1),
      ('elongated_gsxe3_ilackner2', 11, 3, 2),
      ('doublet_gsxe1_ilackner1', 82, 1, 1),
      ('doublet_gsxe1_ilackner2', 82, 1, 2),
      ('doublet_gsxe2_ilackner1', 82, 2, 1),
      ('doublet_gsxe2_ilackner2', 82, 2, 2),
      ('doublet_gsxe3_ilackner1', 82, 3, 1),
      ('doublet_gsxe3_ilackner2', 82, 3, 2),
  )
  def test_meqfx(self, shot, gsxe, ilackner):
    self._oct.eval(_SETUP.format(shot=shot, gsxe=gsxe, ilackner=ilackner))

    Fx_matlab = self._oct.eval('Fx = meqFx(L, LX.Iy, Ie, {Fbe, Iyie});', nout=1)

    def _process_int(name):
      return int(self._oct.eval(name).item())

    def _process_float(name):
      return float(self._oct.eval(name).item())

    def _process_bool(name):
      return bool(self._oct.eval(name).item())

    def _process_1D(name):
      attr = self._oct.eval(name, nout=1).squeeze().T
      return jnp.atleast_1d(attr)

    def _process_2D(name):
      attr = self._oct.eval(name, nout=1).squeeze().T
      if attr.ndim == 1:
        return jnp.array(attr[None])
      else:
        return jnp.array(attr)

    ilackner = _process_int('L.P.ilackner;')
    if ilackner not in (1, 2):
      raise ValueError(f'ilackner={ilackner} is invalid')

    P = types.ParameterData(
        gsxe=_process_int('L.P.gsxe;'),
        ilackner=ilackner,
    )

    L = types.StaticData(
        P=P,
        Mbe=_process_2D('L.Mbe;'),
        Tbc=_process_2D('L.Tbc;'),
        Tye=_process_2D('full(L.Tye);'),  # undo sparse matrix
        ne=_process_int('L.ne;'),
        nx=_process_int('L.nx;'),
        ny=_process_int('L.ny;'),
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
        idzx=_process_float('L.idzx;'),
    )
    Fx_python = jax.jit(functools.partial(meqfx.meqFx, L=L))(
        IyD=_process_2D('LX.Iy;')[None],  # add third dimension to Iy
        Ie=_process_2D('[LX.Ia; LX.Iu];'),
        Fb_ext=(_process_1D('Fbe;'), _process_2D('Iyie;')),
    )
    # TODO(adedieu): lower tolerance
    np.testing.assert_allclose(Fx_python.T, Fx_matlab, atol=1e-5)


if __name__ == '__main__':
  absltest.main()

# pylint: enable=invalid-name
