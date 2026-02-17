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

"""Tests for meqbfct1Jac_test.py, based on Iy_jacobian_test.m."""

import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from meq_jax._src import bfct
from meqpy import octave_utils
import numpy as np

# pylint: disable=invalid-name

# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)


_TASKS = {
    'circular': 1,
    'diverted': 2,
    'diverted2': 3,
    'squashed': 5,
    'doublet': 82,
    'droplets': 84,
    'doublet_with_mantle_current': 88,
}
_ICSINT = (0, 1)

_TEST_CASES = []
for (name, shot_), icsint_ in itertools.product(_TASKS.items(), _ICSINT):
  _TEST_CASES.append(
      dict(
          testcase_name=f'{name}_icsint{icsint_}',
          shot=shot_,
          icsint=icsint_,
      )
  )


_OCTAVE_CMD = """
% Cubic-spline interpolation parameters
PP = {{}};
if {icsint}
  PP = {{'icsint',true,'ilim',3,'tolcs',1e-12}};
end

% compute a plasma equilibrium using fbt
t = 0;
[L,LX,LY] = fgs('ana',{shot},t,'debug',0,PP{{:}});

Fx  = LX.Fx;
ag  = LX.ag;
sIp = sign(LX.Ip);

[~,~,~,~,~,~,~,~,~,~,~,~,~,~,...
 ~,~,~,Opy,F0,F1,~,~,~,dF0dFx,dF1dFx] = meqpdom(Fx,sIp,L.P.isaddl,L);

Tyg = L.bfct(1,L.bfp,Fx,F0,F1,Opy,L.ry,L.iry);
Iy = reshape(Tyg*ag,L.nzy,L.nry);

[dTygdFy,dTygdF0,dTygdF1] = L.bfct(11,L.bfp,Fx,F0,F1,Opy,L.ry,L.iry);

active_domains = int8(find(any(L.TDg.*ag.',1)));
mask = ismember(Opy(:),active_domains);

% Is this sane?
dYygdFy = dTygdFy;
dYygdF0 = dTygdF0;
dYygdF1 = dTygdF1;

nB = LY.nB;

% Full version for test
[dYDgdFy,dYDgdF0,dYDgdF1,dYDgdFx] = meqbfct1Jac(
    L,LY.nB,Opy,dYygdFy,dYygdF0,dYygdF1,dF0dFx,dF1dFx)
"""


class Meqbfct1JacTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._oct = octave_utils.create_meq_oct2py_instance()

  @parameterized.named_parameters(_TEST_CASES)
  def test_meqIyJac(self, shot, icsint):
    """Note that directly instantiating integral_test fails in octave.

    testCase = Iy_jacobian_test()

    So we replicate the logic here, and just check we get identical results to
    octave in python version.

    Args:
      shot: The shot number.
      icsint: Whether to use icsint.
    """
    self._oct.eval(_OCTAVE_CMD.format(shot=shot, icsint=icsint))

    Opy = self._oct.eval('Opy', nout=1)
    dYygdFy = self._oct.eval('dYygdFy', nout=1)
    if dYygdFy.ndim < 3:
      dYygdFy = dYygdFy[:, :, None]
    dYygdF0 = self._oct.eval('dYygdF0', nout=1)
    if dYygdF0.ndim < 3:
      dYygdF0 = dYygdF0[:, :, None]
    dYygdF1 = self._oct.eval('dYygdF1', nout=1)
    if dYygdF1.ndim < 3:
      dYygdF1 = dYygdF1[:, :, None]
    lxy = self._oct.eval('L.lxy', nout=1).astype(bool)
    nD = int(self._oct.eval('L.nD', nout=1).squeeze())
    ng = int(self._oct.eval('L.ng', nout=1).squeeze())
    nx = int(self._oct.eval('L.nx', nout=1).squeeze())
    ny = int(self._oct.eval('L.ny', nout=1).squeeze())
    nB = int(self._oct.eval('LY.nB', nout=1).squeeze())

    dF0dFx = self._oct.eval('dF0dFx', nout=1)
    if hasattr(dF0dFx, 'todense'):
      dF0dFx = dF0dFx.todense()
    dF1dFx = self._oct.eval('dF1dFx', nout=1)
    if hasattr(dF1dFx, 'todense'):
      dF1dFx = dF1dFx.todense()

    dYDgdFy_oct = self._oct.eval('dYDgdFy', nout=1)

    dYDgdF0_oct = self._oct.eval('dYDgdF0', nout=1)
    dYDgdF1_oct = self._oct.eval('dYDgdF1', nout=1)
    dYDgdFx_oct = self._oct.eval('dYDgdFx', nout=1)

    # Note - need to condition on lxy here, but it's an array
    # so can't use static_argnames.  Instead create a closure.
    fast_bfct1_jac = jax.jit(
        functools.partial(bfct.bfct1_jac, lxy=lxy),
        static_argnames=['nB', 'nD', 'ng', 'nx', 'ny'],
    )

    dYDgdFy, dYDgdF0, dYDgdF1, dYDgdFx = fast_bfct1_jac(
        Opy=Opy,
        dYygdFy=dYygdFy,
        dYygdF0=dYygdF0,
        dYygdF1=dYygdF1,
        dF0dFx=dF0dFx,
        dF1dFx=dF1dFx,
        nB=nB,
        nD=nD,
        ng=ng,
        nx=nx,
        ny=ny,
    )

    np.testing.assert_allclose(dYDgdFy, dYDgdFy_oct)
    np.testing.assert_allclose(dYDgdF0, dYDgdF0_oct)
    np.testing.assert_allclose(dYDgdF1, dYDgdF1_oct)
    np.testing.assert_allclose(dYDgdFx, dYDgdFx_oct)


if __name__ == '__main__':
  absltest.main()
