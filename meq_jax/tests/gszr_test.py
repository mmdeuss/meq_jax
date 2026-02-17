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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import gszr
from meqpy import octave_utils
import numpy as np

jax.config.update('jax_enable_x64', True)


class GSZRTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('standard', (30, 63)),
      ('small', (30, 31)),
  )
  def test_gszrjax_vs_gszrmex(self, shape):
    """Tests that gszrjax and gszrmex are equivalent."""
    nr2, nz2 = shape
    batch_size = 3
    prng = jax.random.PRNGKey(0)
    boundary_conditions = jax.random.normal(
        prng,
        (batch_size, 2 * nz2 + 2 * nr2 + 4),
    )
    filament_currents = jax.random.normal(prng, (batch_size, nr2, nz2))
    cx = jax.random.normal(prng, (nr2,))
    cq = jax.random.normal(prng, (nz2, nr2))
    cr = jax.random.normal(prng, (nz2, nr2))
    cs = jax.random.normal(prng, (nz2, nr2))
    ci = 0.3
    co = 0.2
    dz = 0.01

    jit_gszrmex = jax.jit(
        jax.vmap(
            lambda Fb, Iy: gszr.gszrmex(Fb, Iy, cx, cq, cr, cs, ci, co, dz)
        )
    )
    jit_gszrjax = jax.jit(
        jax.vmap(
            lambda Fb, Iy: gszr.gszrjax(Fb, Iy, cx, cq, cr, cs, ci, co, dz)
        )
    )
    fx_majic = jit_gszrmex(boundary_conditions, filament_currents)
    fx_jax = jit_gszrjax(boundary_conditions, filament_currents)
    self.assertTrue(jnp.allclose(fx_majic, fx_jax, equal_nan=True))

  def test_gszrjax_vs_octave(self):
    """Tests that gszrjax and octave are equivalent."""
    oct2py = octave_utils.create_meq_oct2py_instance()
    shape = (30, 63)
    nr2, nz2 = shape
    prng = jax.random.PRNGKey(4)
    boundary_conditions = jax.random.normal(
        prng,
        (2 * nz2 + 2 * nr2 + 4,),
    )
    filament_currents = jax.random.normal(prng, (nr2, nz2))
    cx = jax.random.normal(prng, (nr2,))
    cq = jax.random.normal(prng, (nz2, nr2))
    cr = jax.random.normal(prng, (nz2, nr2))
    cs = jax.random.normal(prng, (nz2, nr2))
    ci = 0.3
    co = 0.2
    dz = 0.01

    fx_octave = oct2py.feval(
        'gszrmex',
        boundary_conditions,
        filament_currents.T,
        cx,
        cq.T,
        cr.T,
        cs.T,
        ci,
        co,
        dz,
    ).T
    jit_gszrjax = jax.jit(
        lambda Fb, Iy: gszr.gszrjax(Fb, Iy, cx, cq, cr, cs, ci, co, dz)
    )
    fx_jax = jit_gszrjax(boundary_conditions, filament_currents)
    np.testing.assert_allclose(fx_jax, fx_octave, rtol=2e-4)


if __name__ == '__main__':
  absltest.main()
