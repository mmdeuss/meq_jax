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

"""Tests for utils for the Python rewrite of MEQ."""

from absl.testing import absltest
import chex
from meq_jax._src import utils
import numpy as np


@chex.dataclass
class SubData:
  a: chex.Array
  b: chex.Array


@chex.dataclass
class TestData:
  a: chex.Array
  b: chex.Array
  c: SubData


class UtilsTest(absltest.TestCase):

  def test_transpose(self):
    """Test that transpose correctly transposes only 2D JAX arrays."""
    data = TestData(
        a=np.zeros((2, 3, 4)),
        b=np.zeros(4),
        c=SubData(a=np.zeros((6, 7)), b=np.zeros((8, 9))),
    )
    transposed_data = utils.transpose(data)
    self.assertEqual(transposed_data.a.shape, (4, 3, 2))
    self.assertEqual(transposed_data.b.shape, (4,))
    self.assertEqual(transposed_data.c.a.shape, (7, 6))
    self.assertEqual(transposed_data.c.b.shape, (9, 8))


if __name__ == "__main__":
  absltest.main()
