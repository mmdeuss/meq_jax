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

"""Tests for meqIyJac.py, based on Iy_jacobian_test.m."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from meq_jax._src import meqIyJac
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
for (name, shot_), icsint_ in itertools.product(
    _TASKS.items(), _ICSINT
):
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
[L,LX] = fgs('ana',{shot},t,'debug',0,PP{{:}});

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
[dIypdFy,dIypdF0,dIypdF1,dIypdFx] = meqIyJac(
  L,ag,mask,dTygdFy,dTygdF0,dTygdF1,dF0dFx,dF1dFx);
"""


class MeqIyJacTest(parameterized.TestCase):

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
    lxy = self._oct.eval('L.lxy', nout=1)

    ag = self._oct.eval('ag', nout=1)[:, 0]
    mask = self._oct.eval('mask', nout=1).astype(bool)[:, 0]
    dTygdFy = self._oct.eval('dTygdFy', nout=1)

    # Add "implicit singleton dimensions back (check please!)
    dTygdF0 = self._oct.eval('dTygdF0', nout=1)
    if dTygdF0.ndim < 3:
      dTygdF0 = dTygdF0[:, :, None]

    dTygdF1 = self._oct.eval('dTygdF1', nout=1)
    if dTygdF1.ndim < 3:
      dTygdF1 = dTygdF1[:, :, None]

    dF0dFx = self._oct.eval('dF0dFx', nout=1)
    dF1dFx = self._oct.eval('dF1dFx', nout=1)
    if not icsint:
      dF0dFx = dF0dFx.todense()
      dF1dFx = dF1dFx.todense()

    # Transpose arrays to match JAX convention
    dIypdFy_nomask, dIypdF0_nomask, dIypdF1_nomask, dIypdFx = meqIyJac.meqIyJac(
        lxy=lxy.T,
        ag=ag,
        mask=mask,
        dTygdFy=dTygdFy.T,
        dTygdF0=jax.numpy.transpose(dTygdF0, (2, 1, 0)),
        dTygdF1=jax.numpy.transpose(dTygdF1, (2, 1, 0)),
        dF0dFx=dF0dFx.T,
        dF1dFx=dF1dFx.T,
    )
    dIypdF0_nomask = dIypdF0_nomask.T
    dIypdF1_nomask = dIypdF1_nomask.T
    dIypdFx = dIypdFx.T

    # target outputs
    dIypdFy_oct = self._oct.eval('dIypdFy', nout=1)
    dIypdF0_oct = self._oct.eval('dIypdF0', nout=1)
    dIypdF1_oct = self._oct.eval('dIypdF1', nout=1)
    dIypdFx_oct = self._oct.eval('dIypdFx', nout=1)
    if icsint:
      dIypdFx = dIypdFx[mask]
    else:
      dIypdFx_oct = dIypdFx_oct.todense()

    np.testing.assert_allclose(
        dIypdFy_nomask[mask], dIypdFy_oct[:, 0], atol=1e-13
    )
    np.testing.assert_allclose(dIypdF0_nomask[mask], dIypdF0_oct, atol=1e-13)
    np.testing.assert_allclose(dIypdF1_nomask[mask], dIypdF1_oct, atol=1e-13)
    np.testing.assert_allclose(dIypdFx_oct, dIypdFx, atol=1e-13)


if __name__ == '__main__':
  absltest.main()
