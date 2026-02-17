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
import jax
from meq_jax._src import nfdb
from meqpy import octave_utils
import numpy as np


jax.config.update("jax_enable_x64", True)


class NFDBTest(absltest.TestCase):

  def test_nfdb_vs_octave(self):
    """Tests that nfdb and the octave implementation are equivalent."""
    oct2py = octave_utils.create_meq_oct2py_instance()
    shape = (32, 65)
    grid = jax.random.normal(jax.random.PRNGKey(4), shape)

    fb_octave = oct2py.feval("nfdbmex", grid.T).flatten()
    fb_jax = jax.jit(nfdb.nfdb)(grid)
    np.testing.assert_allclose(fb_jax, fb_octave, rtol=1e-6)


if __name__ == "__main__":
  absltest.main()
