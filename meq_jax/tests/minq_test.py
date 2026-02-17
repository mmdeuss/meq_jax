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

"""Tests for minq."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import minq
from meqpy import octave_utils
import numpy as np

# pylint: disable=invalid-name
jax.config.update("jax_enable_x64", True)


class MinQTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("standard max", (50,), 1),
      ("standard min", (50,), -1),
      ("non_mono max", (50,), 1, True),
      ("non_mono min", (50,), -1, True),
  )
  def test_minq_vs_octave(self, shape, sq, non_mono=False):
    """Tests that minQ jax and octave are equivalent."""
    oct2py = octave_utils.create_meq_oct2py_instance()

    prngaQ, prngiqQ, _ = jax.random.split(jax.random.PRNGKey(0), 3)
    aQ = jax.random.normal(prngaQ, shape)
    qQ = (
        -jnp.linspace(-2, 2, shape[0]) ** 2
        if non_mono
        else jnp.sort(jax.random.normal(prngiqQ, shape))
    )
    n = 10
    NF = jnp.nan

    # MATLAB/C version
    oct2py.push("aQ", aQ)
    oct2py.push("qQ", qQ)
    oct2py.push("sq", sq)
    oct2py.push("n", n)
    oct2py.push("NF", NF)

    aqmin_matlab, qmin_matlab = oct2py.eval(
        """
        [aqmin,qmin] = minQmex(double(aQ),double(qQ),sq,n,NF);
        """,
        nout=2,
    )

    # JAX version
    aqmin_jax, qmin_jax = jax.jit(
        minq.minQ, static_argnames=["n", "NF"])(aQ, qQ, jnp.array(sq), n, NF)

    np.testing.assert_allclose(aqmin_jax[:, None], aqmin_matlab, atol=1e-6)
    np.testing.assert_allclose(qmin_jax[:, None], qmin_matlab, atol=1e-6)


if __name__ == "__main__":
  absltest.main()
