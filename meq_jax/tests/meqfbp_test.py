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

"""Test the meqfbp.py code."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from meq_jax._src import meqfbp
from meq_jax._src import types
from meqpy import octave_utils
import numpy as np


# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)


# pylint: disable=invalid-name
class MEQFBPTest(parameterized.TestCase):

  def test_oct2py(self):
    """Tests that meqfbp and the octave implementation are equivalent.

    The matlab implementation code is taken from "meq/tests/test_meqfbp.m"
    """
    oct2py = octave_utils.create_meq_oct2py_instance()
    oct2py.eval("L = liu('ana',0,0);")
    L = types.StaticData(
        Tbc=oct2py.eval('L.Tbc', nout=1).squeeze().T,
        Mby=oct2py.eval('L.Mby', nout=1).squeeze().T,
        cx=oct2py.eval('L.cx', nout=1).squeeze(),
        cq=oct2py.eval('L.cq', nout=1).squeeze().T,
        cr=oct2py.eval('L.cr', nout=1).squeeze().T,
        cs=oct2py.eval('L.cs', nout=1).squeeze().T,
        ci=float(oct2py.eval('L.ci', nout=1).squeeze()),
        co=float(oct2py.eval('L.co', nout=1).squeeze()),
    )

    Iy = oct2py.eval(
        """
        a0 = L.P.al*0.9;
        Iy = max(((-(L.ry'-L.P.r0).^2 + -(L.zy-0).^2)/a0^2 + 1 )*1e4,0);
        """,
        nout=1,
    ).T
    # check the shapes of the static data
    nr2, nz2 = Iy.shape
    assert L.Mby is not None
    self.assertEqual(L.Mby.shape, ((nz2 * nr2), 2 * (nz2 + nr2 + 2)))
    assert L.cq is not None
    self.assertEqual(L.cq.shape, (nz2, nr2))
    assert L.cr is not None
    self.assertEqual(L.cr.shape, (nz2, nr2))
    assert L.cs is not None
    self.assertEqual(L.cs.shape, (nz2, nr2))
    assert L.cx is not None
    self.assertEqual(L.cx.shape, (nr2,))
    assert L.Tbc is not None
    self.assertEqual(L.Tbc.shape, (2 * (nz2 + nr2), 2 * (nz2 + nr2 + 2)))

    # Test with different ilackner values.
    jit_meqfbp = jax.jit(meqfbp.meqfbp, static_argnames=['ilackner'])
    for ilackner in (2, 1):
      Fb2 = oct2py.eval(
          f"""
          L.P.ilackner = {ilackner};
          Fb2 = meqfbp(Iy,L);""",
          nout=1,
      ).squeeze()
      Fb_jax = jit_meqfbp(Iy, L, ilackner)
      np.testing.assert_allclose(Fb_jax, Fb2)


# pylint: enable=invalid-name


if __name__ == '__main__':
  absltest.main()
