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

"""Tests for locR."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import locr
from meqpy import octave_utils
import numpy as np


# pylint: disable=invalid-name
jax.config.update("jax_enable_x64", True)


class LocRTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("standard", (200,), 5, 1),
      ("nar=2", (200,), 5, 2),
      ("non_mono", (200,), 5, 3, True),
  )
  def test_locrmex_vs_octave(self, shape, nR, naR, non_mono=False):
    """Tests that locr and octave are equivalent."""
    oct2py = octave_utils.create_meq_oct2py_instance()

    prngaQ, prngiqQ, prngiqR = jax.random.split(jax.random.PRNGKey(0), 3)
    aQ = jax.random.normal(prngaQ, shape)
    qQ = (
        -jnp.linspace(-2, 2, shape[0]) ** 2
        if non_mono
        else jnp.sort(jax.random.normal(prngiqQ, shape))
    )
    qR = jax.random.normal(prngiqR, (nR,))

    # MATLAB/C version
    oct2py.push("aQ", aQ)
    oct2py.push("qQ", qQ)
    oct2py.push("qR", qR)
    oct2py.push("naR", naR)
    oct2py.push("NF", jnp.nan)

    aR_matlab = oct2py.eval(
        """
        aR = locRmex(double(aQ), double(qQ), double(qR), naR, NF);
        """,
        nout=1,
    )

    # JAX version
    aR_jax = jax.jit(locr.locR,
                     static_argnames=["naR", "NF"])(aQ, qQ, qR, naR, jnp.nan)

    np.testing.assert_allclose(aR_matlab, aR_jax.T)


if __name__ == "__main__":
  absltest.main()
