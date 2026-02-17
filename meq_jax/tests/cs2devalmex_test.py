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

"""Compare Matlab and JAX implementations of cs2devalmex."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from meq_jax._src import cs2devalmex
from meqpy import octave_utils
import numpy as np


jax.config.update('jax_enable_x64', True)


# pylint: disable=invalid-name
class Cs2devalmexTest(parameterized.TestCase):

  def test_cs2devalmex_vs_matlab(self):
    oct2py = octave_utils.create_meq_oct2py_instance()
    oct2py.eval("""
    ncx = 8;
    ncy = 16;
    tx = linspace(0, 2, ncx + 4);
    ty = linspace(-1, 1, ncy + 4);
    c = randn(ncx, ncy);
    xp = linspace(0.4, 0.6, 3);
    yp = linspace(0.2, 0.4, 3);
    """)

    # set up knots
    tx = np.squeeze(oct2py.eval('tx;', nout=1))
    ty = np.squeeze(oct2py.eval('ty;', nout=1))

    # set up function to spline fit
    c = oct2py.eval('c;', nout=1).T

    # set up test points
    xp = np.squeeze(oct2py.eval('xp;', nout=1))
    yp = np.squeeze(oct2py.eval('yp;', nout=1))

    jax_results = cs2devalmex.cs2devalmex(tx, ty, c, xp, yp, 3.0)

    matlab_results = oct2py.eval(
        """
        [v, vx, vy, vxx, vxy, vyy, vxxx, vxxy, vxyy, vyyy, vxxxy, vxxyy, vxyyy, vxxxyy, vxxyyy, vxxxyyy] = cs2devalmex(tx, ty, c, xp, yp, 6);
        """,
        nout=16,
    )

    for jax_result, matlab_result in zip(jax_results, matlab_results):
      np.testing.assert_allclose(jax_result,
                                 np.squeeze(matlab_result),
                                 atol=1e-12)


if __name__ == '__main__':
  absltest.main()
