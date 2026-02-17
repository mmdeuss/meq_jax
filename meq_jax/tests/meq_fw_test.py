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

"""Test of flux functions."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import meq_fw
from meqpy import octave_utils
import numpy as np


# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)


# pylint: disable=invalid-name
class MeqFWTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._oct = octave_utils.create_meq_oct2py_instance()

  @parameterized.product(
      FB=(-2, 2),
      FX_ranges=[(-1, 1), (0, 1), (1, 2)],
      nX=(1, 2, 4),
      lX=(True, False),
      nFW=(0, 1, 2),
      dimw=(100,),
  )
  def test_meqfwmex(self, FB, FX_ranges, nX, lX, nFW, dimw):
    key = jax.random.key(42)
    FX = jax.random.uniform(
        key,
        shape=(1, 5),  # arbitrary number of points to 5. nX will slice it.
        minval=FX_ranges[0],
        maxval=FX_ranges[1],
    ).astype(jnp.float64)

    # Python implementation of meqFW
    FW = jax.jit(meq_fw.meq_fw, static_argnames=['nX', 'nFW', 'dimw'])(
        FB=jnp.array(FB, dtype=float),
        FX=FX,
        lX=jnp.array(lX, dtype=bool),
        nX=nX,
        nFW=nFW,
        dimw=dimw,
    )

    # Matlab implementation of meqFW
    FW_ = self._oct.feval('meqFW', FB, FX.T, lX, nX, nFW)
    np.testing.assert_allclose(FW, FW_[0], atol=1e-13)


# pylint: enable=invalid-name


if __name__ == '__main__':
  absltest.main()
