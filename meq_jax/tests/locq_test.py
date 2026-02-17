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

"""Tests for locQ."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import locq
from meqpy import octave_utils
import numpy as np


# pylint: disable=invalid-name
jax.config.update('jax_enable_x64', True)


class LocQTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('standard', (10, 20), 5, 1),
      ('nar=2', (10, 20), 5, 2),
      ('non_mono', (10, 20), 5, 3, True),
  )
  def test_locq_vs_octave(self, shape, nR, naR, non_mono=False):
    """Tests that locq and octave are equivalent."""
    oct2py = octave_utils.create_meq_oct2py_instance()
    _, nQ = shape

    prngaQ, prngiqQ, prngiqR = jax.random.split(jax.random.PRNGKey(0), 3)
    aQ = jax.random.normal(prngaQ, shape)
    iqQ = (
        -jnp.linspace(-2, 2, nQ) ** 2
        if non_mono
        else jnp.sort(jax.random.normal(prngiqQ, (nQ,)))
    )
    iqR = jax.random.normal(prngiqR, (nR,))

    # MATLAB version
    aR_octave = oct2py.locQ(aQ, iqQ, iqR, naR, jnp.nan).squeeze()

    # JAX version
    jit_locQ = jax.jit(locq.locQ, static_argnames=['naR'])
    aR_jax = jit_locQ(aQ, iqQ, iqR, naR, jnp.nan).squeeze()

    np.testing.assert_allclose(aR_jax, aR_octave)


if __name__ == '__main__':
  absltest.main()
