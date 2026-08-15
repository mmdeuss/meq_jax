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

"""Tests for fbnd against the Octave fbndmex implementation.

fbndmex finds the boundary flux FB and limiting point (rB, zB) of the LCFS
given the flux on the limiter points and at the X-points. FN flags invalid
entries. In meqpdom the JAX version is called with fixed-size arrays whose
padded (masked) entries are set to FN, so entries equal to FN must never be
selected; this is exercised explicitly below by passing identically padded
arrays to both implementations.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import fbnd
from meqpy import octave_utils
import numpy as np

# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)


# pylint: disable=invalid-name
class FbndTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._oct = octave_utils.create_meq_oct2py_instance()

  def _compare(self, Fl, rl, zl, FX, rX, zX, FN):
    """Runs both implementations on identical inputs and compares outputs."""
    jit_fbnd = jax.jit(fbnd.fbnd)
    fb, rb, zb, lb, lx, kb = jit_fbnd(
        jnp.asarray(Fl), jnp.asarray(rl), jnp.asarray(zl),
        jnp.asarray(FX), jnp.asarray(rX), jnp.asarray(zX),
        jnp.float64(FN))

    col = lambda x: np.asarray(x, dtype=np.float64).reshape(-1, 1)
    FB_, rB_, zB_, lB_, lX_, kB_ = self._oct.fbndmex(
        col(Fl), col(rl), col(zl), col(FX), col(rX), col(zX), float(FN),
        nout=6)

    # Arithmetic is a pure selection, so results must be identical.
    np.testing.assert_allclose(fb, np.float64(FB_), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rb, np.float64(rB_), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(zb, np.float64(zB_), rtol=1e-12, atol=1e-12)
    # Boolean and integer outputs must match exactly.
    self.assertEqual(bool(lb), bool(np.asarray(lB_).item()))
    self.assertEqual(bool(lx), bool(np.asarray(lX_).item()))
    self.assertEqual(int(kb), int(np.asarray(kB_).item()))

  @parameterized.named_parameters(
      ('nl20_nx3_ip_pos', 20, 3, -0.5, 0),
      ('nl20_nx3_ip_neg', 20, 3, 0.5, 1),
      ('nl7_nx1_ip_pos', 7, 1, -0.5, 2),
      ('nl7_nx1_ip_neg', 7, 1, 0.5, 3),
      ('nl1_nx4_ip_pos', 1, 4, -0.5, 4),
      ('nl50_nx8_ip_neg', 50, 8, 0.5, 5),
  )
  def test_random(self, nl, nx, FN, seed):
    """Random fluxes; limiter or X-point may define the boundary."""
    prng = jax.random.PRNGKey(seed)
    keys = jax.random.split(prng, 6)
    Fl = jax.random.normal(keys[0], (nl,))
    rl = jax.random.uniform(keys[1], (nl,), minval=0.5, maxval=1.2)
    zl = jax.random.normal(keys[2], (nl,))
    FX = jax.random.normal(keys[3], (nx,))
    rX = jax.random.uniform(keys[4], (nx,), minval=0.5, maxval=1.2)
    zX = jax.random.normal(keys[5], (nx,))
    self._compare(Fl, rl, zl, FX, rX, zX, FN)

  @parameterized.named_parameters(
      ('ip_pos', -0.25),
      ('ip_neg', 0.25),
  )
  def test_x_point_defines_boundary(self, FN):
    """Force the X-point to be the extremum so that lX is True."""
    prng = jax.random.PRNGKey(42)
    keys = jax.random.split(prng, 5)
    nl, nx = 12, 3
    Fl = jax.random.normal(keys[0], (nl,))
    rl = jax.random.uniform(keys[1], (nl,), minval=0.5, maxval=1.2)
    zl = jax.random.normal(keys[2], (nl,))
    # Make one X-point flux more extreme than all limiter fluxes.
    FX = jnp.array([0.0, -jnp.sign(FN) * 10.0, 0.1])
    rX = jax.random.uniform(keys[3], (nx,), minval=0.5, maxval=1.2)
    zX = jax.random.normal(keys[4], (nx,))
    self._compare(Fl, rl, zl, FX, rX, zX, FN)

  @parameterized.named_parameters(
      ('ip_pos', -0.5),
      ('ip_neg', 0.5),
  )
  def test_padded_entries_equal_fn(self, FN):
    """Entries equal to FN emulate the masked padding used in meqpdom.

    meqpdom passes fixed-size arrays where invalid entries have been
    overwritten with FN; fbnd must never select those. Both implementations
    receive the identical padded arrays so kB indices are comparable.
    """
    prng = jax.random.PRNGKey(7)
    keys = jax.random.split(prng, 5)
    nl, nx = 16, 6
    Fl = jax.random.normal(keys[0], (nl,))
    # Pad half of the limiter and X-point entries with FN.
    Fl = Fl.at[::2].set(FN)
    FX = jax.random.normal(keys[1], (nx,))
    FX = FX.at[3:].set(FN)
    rl = jax.random.uniform(keys[2], (nl,), minval=0.5, maxval=1.2)
    zl = jax.random.normal(keys[3], (nl,))
    rX = jax.random.uniform(keys[4], (nx,), minval=0.5, maxval=1.2)
    zX = jnp.linspace(-0.7, 0.7, nx)
    self._compare(Fl, rl, zl, FX, rX, zX, FN)

  def test_fully_padded_minus_inf(self):
    """FN = -Inf*sign(Ip) as used in practice (L.FN = -Inf in fgs cases)."""
    FN = -np.inf
    prng = jax.random.PRNGKey(11)
    keys = jax.random.split(prng, 5)
    nl, nx = 10, 4
    Fl = jax.random.normal(keys[0], (nl,))
    Fl = Fl.at[2:7].set(FN)  # -inf padded entries
    FX = jax.random.normal(keys[1], (nx,)).at[1:].set(FN)
    rl = jax.random.uniform(keys[2], (nl,), minval=0.5, maxval=1.2)
    zl = jax.random.normal(keys[3], (nl,))
    rX = jax.random.uniform(keys[4], (nx,), minval=0.5, maxval=1.2)
    zX = jnp.linspace(-0.5, 0.5, nx)
    self._compare(Fl, rl, zl, FX, rX, zX, FN)

  @parameterized.named_parameters(
      ('ip_pos', -0.5),
      ('ip_neg', 0.5),
  )
  def test_no_valid_point(self, FN):
    """All fluxes are on the wrong side of FN so no boundary is found."""
    prng = jax.random.PRNGKey(3)
    keys = jax.random.split(prng, 5)
    nl, nx = 9, 2
    # For FN < 0 the extremum is a max: values below FN are all invalid.
    # For FN > 0 the extremum is a min: values above FN are all invalid.
    offs = jnp.sign(FN)
    Fl = FN + offs * jnp.abs(jax.random.normal(keys[0], (nl,)))
    FX = FN + offs * jnp.abs(jax.random.normal(keys[1], (nx,)))
    rl = jax.random.uniform(keys[2], (nl,), minval=0.5, maxval=1.2)
    zl = jax.random.normal(keys[3], (nl,))
    rX = jax.random.uniform(keys[4], (nx,), minval=0.5, maxval=1.2)
    zX = jnp.linspace(-0.5, 0.5, nx)
    self._compare(Fl, rl, zl, FX, rX, zX, FN)

  @parameterized.named_parameters(
      ('ip_pos', -0.5),
      ('ip_neg', 0.5),
  )
  def test_empty_x_points(self, FN):
    """nX = 0, as happens when no X-point candidates exist."""
    prng = jax.random.PRNGKey(5)
    keys = jax.random.split(prng, 3)
    nl = 11
    Fl = jax.random.normal(keys[0], (nl,))
    rl = jax.random.uniform(keys[1], (nl,), minval=0.5, maxval=1.2)
    zl = jax.random.normal(keys[2], (nl,))
    empty = jnp.zeros((0,))
    self._compare(Fl, rl, zl, empty, empty, empty, FN)

  def test_empty_limiter_and_x_points(self):
    """nl = nX = 0 (e.g. ilim=0 and no X-points): lB must be False."""
    empty = jnp.zeros((0,))
    self._compare(empty, empty, empty, empty, empty, empty, -0.5)
# pylint: enable=invalid-name


if __name__ == '__main__':
  absltest.main()
