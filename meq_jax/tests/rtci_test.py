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

"""Tests for rtci.rtcimex against the Octave rtcimex implementation.

This exercises the low-level Newton update directly (the higher-level
wrappers are covered by rtcics_test.py, rtciplasma_test.py, rtciwall_test.py
and find_contours_test.py).

Conventions:

* MATLAB stores ``Fx`` as (nzx, nrx) and ``aq`` as (noq, npq); the JAX
  implementation expects the transposed row-major layouts (nrx, nzx) and
  (npq, noq) respectively. ``Opy`` (nzy, nry) is likewise passed transposed;
  its row-major ravel then reproduces the Matlab column-major linear
  indexing used inside the mex.

* The initial guess ``aq0`` is deterministic and spans distances from well
  inside the plasma to outside the vessel, so that all update branches
  (plain Newton step, wrong-domain back-off, wrong-gradient escape and step
  clipping) are exercised.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from meq_jax._src import rtci
from meqpy import octave_utils
import numpy as np

# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)

# pylint: disable=invalid-name


# Setup mirrors find_contours.m for the non-icsint branch: rays originate at
# the magnetic axis and contours are sought at the flux values F.
_FIELDS_CMD = """
[L,~,LY] = fbt('ana', {shot}, 0, 'iterq', 20, 'noq', 64, 'pq', linspace(0,1,41), 'icsint', false, 'ilim', 1);

F = LY.FB - L.fq*(LY.FB - LY.FA);
ero = (LY.rA - L.G.rx(1))*L.idrx;
ezo = (LY.zA - L.G.zx(1))*L.idzx;
cdr = L.crq*L.idrx;
cdz = L.czq*L.idzx;
Fo = LY.FA;
Opo = int8(1);
% Deterministic initial guess covering [0.05, 1.15] m, i.e. from deep inside
% the plasma to far outside it.
aq0 = 0.05 + 1.1*abs(sin((1:L.noq)'*(1:numel(F))));

% Single Newton update
[aq1, dan1] = rtcimex(aq0, ero, ezo, LY.Fx, F, cdr, cdz, LY.Opy, Fo, Opo, L.drx);

% Full iteration as done in find_contours.m
aqk = aq0;
for k = 1:L.P.iterq
  [aqk, dank] = rtcimex(aqk, ero, ezo, LY.Fx, F, cdr, cdz, LY.Opy, Fo, Opo, L.drx);
end
"""


class RtciTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._oct = octave_utils.create_meq_oct2py_instance()

  @parameterized.named_parameters(
      ('limiter', 1),
      ('single_null', 2),
      ('elongated', 11),
  )
  def test_rtcimex_vs_matlab(self, shot):
    self._oct.eval(_FIELDS_CMD.format(shot=shot))

    def pull_1d(name):
      return np.atleast_1d(
          np.asarray(self._oct.eval(f'{name};', nout=1)).squeeze()
      )

    Fx = self._oct.eval('LY.Fx;', nout=1).T  # (nrx, nzx)
    Opy = self._oct.eval('LY.Opy;', nout=1).T  # (nry, nzy)
    aq0 = self._oct.eval('aq0;', nout=1).T  # (npq, noq)
    F = pull_1d('F')
    ero = pull_1d('ero')
    ezo = pull_1d('ezo')
    cdr = pull_1d('cdr')
    cdz = pull_1d('cdz')
    Fo = pull_1d('Fo')
    Opo = pull_1d('Opo').astype(np.int8)
    dap = self._oct.eval('L.drx;', nout=1)[0, 0]
    iterq = int(self._oct.eval('L.P.iterq;', nout=1)[0, 0])

    rtcimex = jax.jit(rtci.rtcimex)

    # Single Newton update.
    a1_jax, dan1_jax = rtcimex(
        a=aq0,
        er0=ero,
        ez0=ezo,
        Fx=Fx,
        F=F,
        c=cdr,
        s=cdz,
        Opy=Opy,
        Fo=Fo,
        Opo=Opo,
        dap=dap,
    )

    # Full iteration as done in find_contours.py.
    ak_jax = aq0
    dank_jax = None
    for _ in range(iterq):
      ak_jax, dank_jax = rtcimex(
          a=ak_jax,
          er0=ero,
          ez0=ezo,
          Fx=Fx,
          F=F,
          c=cdr,
          s=cdz,
          Opy=Opy,
          Fo=Fo,
          Opo=Opo,
          dap=dap,
      )

    aq1_matlab = self._oct.eval('aq1;', nout=1)
    dan1_matlab = self._oct.eval('dan1;', nout=1)[0, 0]
    aqk_matlab = self._oct.eval('aqk;', nout=1)
    dank_matlab = self._oct.eval('dank;', nout=1)[0, 0]

    # The update is pure (branchy) arithmetic with no iterative refinement
    # inside a single call, so the results should match to rounding error.
    # (The only intentional deviation is the "gradient reverses" test, where
    # the JAX version uses df*(Fo-fq) > 1e-13 instead of > 0; realistic
    # inputs do not fall inside that margin.)
    np.testing.assert_allclose(np.asarray(a1_jax).T, aq1_matlab, atol=1e-12)
    np.testing.assert_allclose(float(dan1_jax), dan1_matlab, atol=1e-12)
    np.testing.assert_allclose(np.asarray(ak_jax).T, aqk_matlab, atol=1e-10)
    np.testing.assert_allclose(float(dank_jax), dank_matlab, atol=1e-10)


if __name__ == '__main__':
  absltest.main()

# pylint: enable=invalid-name
