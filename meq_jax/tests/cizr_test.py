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

"""Tests for cizr."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import cizr
from meqpy import octave_utils
import numpy as np

jax.config.update('jax_enable_x64', True)


# pylint: disable=invalid-name
class CizrTest(parameterized.TestCase):

  def test_cizr_vs_matlab(self):
    """Tests that cizr and the matlab implementation are equivalent."""
    oct2py = octave_utils.create_meq_oct2py_instance()

    prng = jax.random.PRNGKey(0)
    prng_fx, prng_opy, prng_vy, prng_fnq, prng_f0, prng_f1 = jax.random.split(
        prng, 6
    )

    nzx = 30
    nrx = 63
    nV = 10
    nQ = 10
    nD = 10
    Fx = jax.random.normal(prng_fx, (nzx, nrx))
    Opy = jax.random.randint(prng_opy, (nzx - 2, nrx - 2), 0, nD + 1)
    Vy = jax.random.normal(prng_vy, (nzx - 2, nrx - 2, nV))
    dsx = 0.01
    FNQ = jnp.sort(jax.random.uniform(prng_fnq, (nQ,)))
    F0 = jax.random.normal(prng_f0, (nD,))
    F1 = jax.random.lognormal(prng_f1, shape=(nD,)) + F0  # Ensure F1 > F0

    # JAX version
    jit_cizr = jax.jit(cizr.cizr)
    IVQD_jax = jit_cizr(Fx, Opy, Vy, dsx, FNQ, F0, F1)

    # MATLAB version
    # Need to be careful with axis order for arrays that are flattened in Matlab
    matlab_vy = Vy.transpose((1, 0, 2)).reshape((-1, nV))
    IVQD_matlab = oct2py.cizrmexm(Fx, Opy, matlab_vy, dsx, FNQ, F0, F1).T

    np.testing.assert_allclose(IVQD_jax, IVQD_matlab, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
  absltest.main()
