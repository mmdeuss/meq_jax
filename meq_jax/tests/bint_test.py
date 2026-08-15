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

"""Tests for bint."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import bint
from meqpy import octave_utils
import numpy as np

jax.config.update('jax_enable_x64', True)


# pylint: disable=invalid-name
class BintTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('grid_30x63', 30, 63, 50),
      ('grid_65x28', 65, 28, 200),
      ('ni=1', 30, 60, 1),
      ('minimal_grid', 2, 2, 5),
  )
  def test_bintmex_vs_octave(self, nr, nz, ni):
    """Tests that bintmex and the compiled Octave MEX are equivalent.

    Conventions: the JAX Fx has shape (nr, nz), i.e. transposed relative to
    the MATLAB Fx of shape (nz, nr), so that flattening the JAX array in
    row-major order matches flattening the MATLAB array in column-major
    order. Both implementations take the same 0-based linear index k into
    that flattened array (callers pass int32(k-1) where k comes from
    bintc.m, cf. meq/rtciwall.m and meq_jax/_src/rtciwall.py). The JAX c
    has shape (ni, 4) versus (4, ni) in MATLAB.
    """
    oct2py = octave_utils.create_meq_oct2py_instance()

    prng = jax.random.PRNGKey(0)
    prng_fx, prng_kr, prng_kz, prng_c = jax.random.split(prng, 4)

    Fx = jax.random.normal(prng_fx, (nr, nz))
    # Valid stencil origins: the MEX reads offsets k, k+1, k+nz, k+nz+1 of
    # the flattened array, so the origin (kr, kz) must not be on the upper
    # edges of the grid.
    kr = jax.random.randint(prng_kr, (ni,), 0, nr - 1)
    kz = jax.random.randint(prng_kz, (ni,), 0, nz - 1)
    k = (kr * nz + kz).astype(jnp.int32)
    c = jax.random.normal(prng_c, (ni, 4))

    # JAX version
    jit_bintmex = jax.jit(bint.bintmex)
    Fi_jax = jit_bintmex(Fx, k, c)

    # MATLAB version (compiled MEX; requires k as an int32 array)
    oct2py.push('Fx', np.asarray(Fx).T)
    oct2py.push('k', np.asarray(k, dtype=np.float64))
    oct2py.push('c', np.asarray(c).T)
    Fi_matlab = oct2py.eval('Fi = bintmex(Fx, int32(k(:)), c);', nout=1)

    np.testing.assert_allclose(
        Fi_jax, np.asarray(Fi_matlab).ravel(), rtol=1e-12, atol=1e-12
    )


if __name__ == '__main__':
  absltest.main()
