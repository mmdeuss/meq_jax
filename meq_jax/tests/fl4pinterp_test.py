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

"""Tests for fl4pinterp, against meq/fl4pinterp.m.

Follows meq/tests/fl4pinterp_test.m: an ilim=2 equilibrium, both signs of Ip,
and limiter contours shifted and shrunk so that different vertices become the
candidate extremum. The 'spike' case pulls one vertex in towards the magnetic
axis, which is what makes a single interior point win.
"""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import fl4pinterp
from meqpy import octave_utils
import numpy as np

# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)

# pylint: disable=invalid-name

_SETUP = """
[L,LX] = fbt('ana',{shot},[],'nl',16,'ilim',2);
LX.Ip = {sIp}*LX.Ip;
if L.nD > 1
  LX.IpD = {sIp}*LX.IpD;
end
LX.qA = {sIp}*LX.qA;
LX = fbtx(L,LX);
LY = fbtt(L,LX);
"""

# Perturb the limiter contour, as the MEQ test does, so the extremum lands on
# different vertices and inside different segments.
_PERTURB = """
rl = 0.9*(L.G.rl - L.P.r0) + {dr} + L.P.r0;
zl = 0.9*L.G.zl + {dz};
FN = LY.FB(LY.nB);
[rlo,zlo,Flo,drFlo,dzFlo] = fl4pinterp(LY.Fx,L.kxl,L.clx,L.kxlh,L.clhx,FN,rl,zl);
"""

_CASES = [
    dict(testcase_name=f'shot{shot}_sIp{"p" if s > 0 else "m"}_dr{i}_dz{j}',
         shot=shot, sIp=s, dr=dr, dz=dz)
    for shot, s, (i, dr), (j, dz) in itertools.product(
        (1, 82), (-1, 1), enumerate((-0.04, 0.04)), enumerate((-0.04, 0.04))
    )
]


class Fl4pInterpTest(parameterized.TestCase):

  _oct_session = None

  @classmethod
  def _octave(cls):
    if cls._oct_session is None:
      cls._oct_session = octave_utils.create_meq_oct2py_instance()
    return cls._oct_session

  @parameterized.named_parameters(_CASES)
  def test_matches_octave(self, shot, sIp, dr, dz):
    oct_ = self._octave()
    oct_.eval(_SETUP.format(shot=shot, sIp=sIp))
    oct_.eval(_PERTURB.format(dr=dr, dz=dz))

    def f1(expr):
      return np.atleast_1d(np.squeeze(oct_.eval(expr + ';', nout=1)))

    Fx = np.asarray(oct_.eval('LY.Fx;', nout=1)).T  # JAX transposes the grid
    kl = f1('L.kxl').astype(np.int32) - 1
    cl = np.atleast_2d(oct_.eval('L.clx;', nout=1)).T
    klh = f1('L.kxlh').astype(np.int32) - 1
    clh = np.atleast_2d(oct_.eval('L.clhx;', nout=1)).T
    FN = float(f1('FN')[0])
    rl, zl = f1('rl'), f1('zl')

    got = jax.jit(fl4pinterp.fl4pinterp)(
        Fx=jnp.asarray(Fx), kl=jnp.asarray(kl), cl=jnp.asarray(cl),
        klh=jnp.asarray(klh), clh=jnp.asarray(clh), FN=jnp.asarray(FN),
        rl=jnp.asarray(rl), zl=jnp.asarray(zl),
    )
    want = (f1('rlo'), f1('zlo'), f1('Flo'), f1('drFlo'), f1('dzFlo'))

    for name, g, w in zip(('rl', 'zl', 'Fl', 'drFl', 'dzFl'), got, want):
      # The two implementations agree to one ULP or exactly (rl and zl are
      # bit-identical; the worst relative difference over these cases is
      # 2.2e-16, against a float64 epsilon of 2.2e-16). The tolerance is set
      # near that rather than at a comfortable 1e-8, so that a port which
      # drifts is caught instead of passing quietly; the headroom is for
      # platform differences in floating-point contraction.
      np.testing.assert_allclose(
          np.asarray(g).squeeze(), w, rtol=1e-14, atol=1e-15,
          err_msg=f'{name} differs from Octave',
      )


if __name__ == '__main__':
  absltest.main()

# pylint: enable=invalid-name
