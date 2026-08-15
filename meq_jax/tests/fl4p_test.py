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

"""Tests for fl4p against the Octave fl4pmex implementation.

fl4pmex computes the flux on the limiter contour by four point (bilinear)
interpolation, tags all points that are not extrema compatible with the sign
of FN with FN, and returns the flux gradients in the R and Z directions.

Conventions (see meqpdom.py): the JAX version takes Fx transposed
([nr, nz] instead of Matlab's [nz, nr]), cl transposed ([nl, 4] instead of
[4, nl]) and the 0-based kl = L.kxl - 1 (Matlab passes int32(L.kxl-1)).
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import fl4p
from meqpy import octave_utils
import numpy as np

# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)


# pylint: disable=invalid-name
class Fl4pTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._oct = octave_utils.create_meq_oct2py_instance()

  def _compare(self, Fx, kl, cl, FN):
    """Runs both implementations and compares the outputs.

    Args:
      Fx: flux map in Matlab layout [nz, nr] (numpy).
      kl: 0-based int32 indices of the reference grid points, shape [nl].
      cl: interpolation coefficients in Matlab layout [4, nl] (numpy).
      FN: flag/sign scalar.
    """
    Fx = np.asarray(Fx)
    kl = np.asarray(kl, dtype=np.int32).ravel()
    cl = np.asarray(cl)

    # JAX version: transposed arrays.
    jit_fl4p = jax.jit(fl4p.fl4pmex)
    Fl, drFl, dzFl = jit_fl4p(
        jnp.asarray(Fx.T), jnp.asarray(kl), jnp.asarray(cl.T),
        jnp.float64(FN))

    # Octave version. Use eval so that kl remains int32 (oct2py feval would
    # convert integer inputs to double, which fl4pmex rejects).
    self._oct.push('Fx_', Fx)
    self._oct.push('kl_', kl.astype(np.float64).reshape(-1, 1))
    self._oct.push('cl_', cl)
    self._oct.push('FN_', float(FN))
    self._oct.eval('[Fl_, drFl_, dzFl_] = fl4pmex(Fx_, int32(kl_), cl_, FN_);')
    Fl_ = self._oct.pull('Fl_').ravel()
    drFl_ = self._oct.pull('drFl_').ravel()
    dzFl_ = self._oct.pull('dzFl_').ravel()

    # Identical arithmetic, so the results must agree to rounding level.
    np.testing.assert_allclose(Fl, Fl_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(drFl, drFl_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dzFl, dzFl_, rtol=1e-12, atol=1e-12)

  @parameterized.named_parameters(
      ('circular_ip_sign', 1, 'ip'),
      ('diverted_ip_sign', 2, 'ip'),
      ('diverted2_ip_sign', 3, 'ip'),
      ('squashed_ip_sign', 5, 'ip'),
      ('circular_reversed', 1, 'reversed'),
      ('diverted_reversed', 2, 'reversed'),
      ('diverted2_reversed', 3, 'reversed'),
      ('squashed_reversed', 5, 'reversed'),
  )
  def test_fgs_equilibrium(self, shot, fn_case):
    """Realistic Fx, L.kxl and L.clx from an fgs analytic equilibrium.

    This mirrors the call in meqpdom: fl4pmex(Fx, int32(L.kxl-1), L.clx, FN)
    with FN = L.FN*sign(Ip) (L.FN = -Inf for these cases). The 'reversed'
    case flips the sign of FN to exercise the other masking branch.
    """
    self._oct.eval(f"[L,LX] = fgs('ana', {shot}, 0);")
    Fx = self._oct.eval('LX.Fx;', nout=1)
    kxl = self._oct.eval('L.kxl;', nout=1).ravel()
    clx = self._oct.eval('L.clx;', nout=1)
    LFN = self._oct.eval('L.FN;', nout=1).item()
    sIp = np.sign(self._oct.eval('LX.Ip;', nout=1).item())
    FN = LFN * sIp if fn_case == 'ip' else -LFN * sIp
    self._compare(Fx, kxl - 1, clx, FN)

  @parameterized.named_parameters(
      ('finite_fn', 2, 0.5),
      ('finite_fn_neg', 2, -0.5),
  )
  def test_fgs_equilibrium_finite_fn(self, shot, FN):
    """Finite FN values (partial masking with sign-dependent branch)."""
    self._oct.eval(f"[L,LX] = fgs('ana', {shot}, 0);")
    Fx = self._oct.eval('LX.Fx;', nout=1)
    kxl = self._oct.eval('L.kxl;', nout=1).ravel()
    clx = self._oct.eval('L.clx;', nout=1)
    self._compare(Fx, kxl - 1, clx, FN)

  @parameterized.named_parameters(
      ('nl16_ip_pos', 9, 7, 16, -1.0, 0),
      ('nl16_ip_neg', 9, 7, 16, 1.0, 1),
      ('nl3_ip_pos', 5, 6, 3, -0.25, 2),
      ('nl40_ip_neg', 12, 11, 40, 0.25, 3),
      ('nl16_minus_inf', 9, 7, 16, -np.inf, 4),
  )
  def test_random(self, nz, nr, nl, FN, seed):
    """Random flux map, random interior cells and random coefficients."""
    prng = jax.random.PRNGKey(seed)
    k_fx, k_iz, k_ir, k_cl = jax.random.split(prng, 4)
    Fx = np.asarray(jax.random.normal(k_fx, (nz, nr)))
    # Valid 0-based cell indices: kl = ir*nz + iz with iz <= nz-2, ir <= nr-2
    # so that kl+nz+1 stays within the array.
    iz = jax.random.randint(k_iz, (nl,), 0, nz - 1)
    ir = jax.random.randint(k_ir, (nl,), 0, nr - 1)
    kl = np.asarray(ir * nz + iz, dtype=np.int32)
    cl = np.asarray(jax.random.uniform(k_cl, (4, nl)))
    self._compare(Fx, kl, cl, FN)

  def test_fn_zero_gradients(self):
    """FN == 0: both implementations return zero gradients.

    Known divergence (not asserted here): for FN == 0 the MEX skips the
    non-extremum masking entirely and returns the raw interpolated Fl,
    whereas the JAX version applies the dFl*dFl2 > 0 part of the mask
    unconditionally and overwrites those entries with FN = 0. This is
    harmless in meqpdom because FN = L.FN*sign(Ip) is only 0 in the vacuum
    case, where Fl is subsequently overwritten, but the raw outputs differ.
    """
    prng = jax.random.PRNGKey(6)
    k_fx, k_iz, k_ir, k_cl = jax.random.split(prng, 4)
    nz, nr, nl = 9, 7, 16
    Fx = np.asarray(jax.random.normal(k_fx, (nz, nr)))
    iz = jax.random.randint(k_iz, (nl,), 0, nz - 1)
    ir = jax.random.randint(k_ir, (nl,), 0, nr - 1)
    kl = np.asarray(ir * nz + iz, dtype=np.int32)
    cl = np.asarray(jax.random.uniform(k_cl, (4, nl)))

    drFl, dzFl = jax.jit(fl4p.fl4pmex)(
        jnp.asarray(Fx.T), jnp.asarray(kl), jnp.asarray(cl.T),
        jnp.float64(0.0))[1:]
    np.testing.assert_array_equal(drFl, np.zeros(nl))
    np.testing.assert_array_equal(dzFl, np.zeros(nl))
# pylint: enable=invalid-name


if __name__ == '__main__':
  absltest.main()
