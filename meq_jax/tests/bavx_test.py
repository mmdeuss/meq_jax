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

"""Tests for bavx."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import bavx
from meqpy import octave_utils
import numpy as np

# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)

# pylint: disable=invalid-name


def _field(oct, expr):
  """Fetches an Octave expression as a 1D numpy array."""
  return np.atleast_1d(np.squeeze(oct.eval(f'{expr};', nout=1)))


class BavxTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._oct = octave_utils.create_meq_oct2py_instance()

  def _load_xpoint_geometry(self, shot, use_hessian):
    """Loads X-point positions and inward vectors from an fgs equilibrium.

    Reproduces the two ways meqpdom.m constructs the (vrX, vzX) vectors:
    either pointing from the X-points towards the magnetic axis, or from the
    X-point Hessian (towards positive gradient direction).
    """
    oct = self._oct
    oct.eval(f"[L,LX] = fgs('ana', {shot}, 0);")
    rX = _field(oct, 'LX.rX')
    zX = _field(oct, 'LX.zX')
    rA = _field(oct, 'LX.rA')
    zA = _field(oct, 'LX.zA')
    sIp = np.sign(_field(oct, 'LX.Ip')[0])
    if use_hessian:
      dr2FX = _field(oct, 'LX.dr2FX')
      dz2FX = _field(oct, 'LX.dz2FX')
      drzFX = _field(oct, 'LX.drzFX')
      H1 = sIp * (dr2FX - dz2FX) * 0.5
      H0 = np.sqrt(H1 * H1 + drzFX * drzFX)
      vrX = np.sqrt(H0 + H1)
      vzX = np.sqrt(H0 - H1) * ((drzFX >= 0) * 2 - 1) * sIp
    else:
      vrX = rA[0] - rX
      vzX = zA[0] - zX
    # Orient the vectors towards the domain centre (magnetic axis), as
    # done in meqpdom.m before calling bavxmex.
    vsX = np.sign(vrX * (rA[0] - rX) + vzX * (zA[0] - zX))
    vrX = vrX * vsX
    vzX = vzX * vsX
    rx = _field(oct, 'L.rx')
    zx = _field(oct, 'L.zx')
    return rX, zX, vrX, vzX, rx, zx

  @parameterized.named_parameters(
      ('diverted_axis_a0', 2, False, 0.0),
      ('diverted_hessian_a0', 2, True, 0.0),
      ('diverted_hessian_a035', 2, True, 0.35),
      ('diverted2_axis_a0', 3, False, 0.0),
      ('diverted2_axis_a035', 3, False, 0.35),
      ('diverted2_hessian_a0', 3, True, 0.0),
      ('doublet_hessian_a035', 82, True, 0.35),
      ('droplets_axis_a0', 84, False, 0.0),
      ('droplets_hessian_a035', 84, True, 0.35),
  )
  def test_bavx_vs_matlab(self, shot, use_hessian, A):
    """Compares bavx (point list) and bavx2 (mesh) against bavxmex."""
    rX, zX, vrX, vzX, rx, zx = self._load_xpoint_geometry(shot, use_hessian)
    nX = rX.size
    self.assertGreater(nX, 0)

    # Query points: the full flattened grid plus the X-points themselves
    # (meqpdom.m calls bavxmex on both the grid and on rX, zX).
    rr, zz = np.meshgrid(rx, zx)
    pr = np.concatenate([rr.ravel(), rX])
    pz = np.concatenate([zz.ravel(), zX])

    # --- bavx: list-of-points version (7-argument bavxmex) ---
    k_oct = np.squeeze(
        self._oct.bavxmex(rX, zX, vrX, vzX, A, pr, pz)).astype(bool)
    mask = jnp.ones(nX, dtype=jnp.bool_)
    k_jax = jax.jit(bavx.bavx)(rX, zX, vrX, vzX, A, pr, pz, mask)
    # Boolean output: must match exactly.
    np.testing.assert_array_equal(np.asarray(k_jax), k_oct)

    # --- bavx2: mesh version (8-argument bavxmex) ---
    # The Octave output is L(z, r); the JAX output is transposed, (r, z).
    OX_oct = np.asarray(
        self._oct.bavxmex(rX, zX, vrX, vzX, A, rx, zx, 1.0)).astype(bool)
    OX_jax = jax.jit(bavx.bavx2, static_argnums=8)(
        rX, zX, vrX, vzX, A, rx, zx, mask, None)
    self.assertEqual(OX_jax.shape, (rx.size, zx.size))
    np.testing.assert_array_equal(np.asarray(OX_jax).T, OX_oct)

  @parameterized.named_parameters(
      ('a0', 0.0),
      ('a035', 0.35),
  )
  def test_bavx_mask_vs_matlab_slicing(self, A):
    """The mask argument must be equivalent to slicing the MATLAB inputs."""
    # Shot 3 has two X-points; select only the first one and pad the arrays
    # with junk entries, as meqpdom.py does with its fixed-size arrays.
    rX, zX, vrX, vzX, rx, zx = self._load_xpoint_geometry(3, use_hessian=True)
    self.assertEqual(rX.size, 2)

    dimw = 5  # padded, fixed-size arrays as used by meqpdom.py
    pad = dimw - rX.size

    def padded(v):
      # Junk padding values that would change the answer if not masked out.
      return np.concatenate([v, np.full(pad, 0.5 * (v[0] + 1.0))])

    selX = np.concatenate(
        [np.array([True, False]), np.zeros(pad, dtype=bool)])
    rXp, zXp = padded(rX), padded(zX)
    vrXp, vzXp = padded(vrX), padded(vzX)

    rr, zz = np.meshgrid(rx, zx)
    pr = np.concatenate([rr.ravel(), rX])
    pz = np.concatenate([zz.ravel(), zX])

    # Octave gets the sliced arrays (rX(selX) etc. in meqpdom.m).
    k_oct = np.squeeze(self._oct.bavxmex(
        rX[:1], zX[:1], vrX[:1], vzX[:1], A, pr, pz)).astype(bool)
    OX_oct = np.asarray(self._oct.bavxmex(
        rX[:1], zX[:1], vrX[:1], vzX[:1], A, rx, zx, 1.0)).astype(bool)

    k_jax = jax.jit(bavx.bavx)(rXp, zXp, vrXp, vzXp, A, pr, pz, selX)
    OX_jax = jax.jit(bavx.bavx2, static_argnums=8)(
        rXp, zXp, vrXp, vzXp, A, rx, zx, selX, None)

    np.testing.assert_array_equal(np.asarray(k_jax), k_oct)
    np.testing.assert_array_equal(np.asarray(OX_jax).T, OX_oct)

  def test_bavx_all_masked(self):
    """With everything masked out, no point is excluded (all True).

    This matches the nX = 0 case of the C implementation (libmeq/bavx.c),
    where the exclusion flag is initialised to false.
    """
    rX = np.array([0.8, 1.0])
    zX = np.array([-0.5, 0.6])
    vrX = np.array([0.1, -0.1])
    vzX = np.array([0.9, -0.8])
    mask = np.zeros(2, dtype=bool)
    pr = np.linspace(0.6, 1.2, 7)
    pz = np.linspace(-0.7, 0.7, 5)
    rr, zz = np.meshgrid(pr, pz)

    k_jax = jax.jit(bavx.bavx)(
        rX, zX, vrX, vzX, 0.0, rr.ravel(), zz.ravel(), mask)
    OX_jax = jax.jit(bavx.bavx2, static_argnums=8)(
        rX, zX, vrX, vzX, 0.3, pr, pz, mask, None)
    np.testing.assert_array_equal(np.asarray(k_jax), True)
    np.testing.assert_array_equal(np.asarray(OX_jax), True)

  @parameterized.named_parameters(
      ('a0', 0.0),
      ('a035', 0.35),
  )
  def test_bavx_synthetic_polygon(self, A):
    """Synthetic X-point polygon with random query points."""
    prng = jax.random.PRNGKey(0)
    prng_r, prng_z = jax.random.split(prng)

    # Three X-points on an ellipse around a plasma "centre", with inward
    # oriented vectors pointing at the centre.
    rc, zc = 0.9, 0.1
    theta = np.array([-2.2, -0.6, 1.9])
    rX = rc + 0.35 * np.cos(theta)
    zX = zc + 0.55 * np.sin(theta)
    vrX = rc - rX
    vzX = zc - zX

    n = 500
    pr = np.asarray(jax.random.uniform(prng_r, (n,), minval=0.4, maxval=1.4))
    pz = np.asarray(jax.random.uniform(prng_z, (n,), minval=-0.9, maxval=1.1))

    k_oct = np.squeeze(
        self._oct.bavxmex(rX, zX, vrX, vzX, A, pr, pz)).astype(bool)
    mask = jnp.ones(3, dtype=jnp.bool_)
    k_jax = jax.jit(bavx.bavx)(rX, zX, vrX, vzX, A, pr, pz, mask)
    np.testing.assert_array_equal(np.asarray(k_jax), k_oct)
    # Sanity check: some points inside, some outside.
    self.assertTrue(k_oct.any())
    self.assertFalse(k_oct.all())

    # Mesh version on a regular grid covering the polygon.
    gr = np.linspace(0.4, 1.4, 41)
    gz = np.linspace(-0.9, 1.1, 37)
    OX_oct = np.asarray(
        self._oct.bavxmex(rX, zX, vrX, vzX, A, gr, gz, 1.0)).astype(bool)
    OX_jax = jax.jit(bavx.bavx2, static_argnums=8)(
        rX, zX, vrX, vzX, A, gr, gz, mask, None)
    np.testing.assert_array_equal(np.asarray(OX_jax).T, OX_oct)


if __name__ == '__main__':
  absltest.main()

# pylint: enable=invalid-name
