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

"""Tests for meq_jax._src.bfef against the Octave/MEX reference.

The reference implementations are meq/mexm/bfef.m (pure MATLAB, routed
through bfct.m) and the compiled bfefmex.mex. The mode selection maps to
the JAX functions as follows (see meq/include/bfctmex.h and meq/bfct.m):

  [g, Ig]          = bfefmex(91, [nP nT], F, F0, F1)   <->  bfef.f
  [gQg, IgQg]      = bfefmex( 2, [nP nT], FN, F0, F1)  <->  bfef.fN
  [gAg, IgAg]      = bfefmex( 5, [nP nT], ~, F0, F1)   <->  bfef.fA
  [fPg, fTg]       = bfefmex( 0, [nP nT])              <->  bfef.fPg
  [aPpg,~,aPg,~]   = bfef   ( 3, [nP nT], ag, F0, F1, fPg, fTg, ids)
                                                       <->  bfef.fag
For mode 3, with ag = fPg = 1, fTg = 0 and ids = 2*pi the MATLAB cP factor
(ids/2/pi) is exactly 1, so aPpg = alphapg and aPg = alphag.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import bfef
from meqpy import octave_utils
import numpy as np


# Turn on float64 for comparison with the Matlab implementation.
jax.config.update('jax_enable_x64', True)

# pylint: disable=invalid-name

# (nP, nT) basis-function counts. bfef supports nP <= 4, nT <= 4.
NP_NT_CASES = ((1, 1), (2, 1), (3, 2), (0, 1), (4, 4))

# (FA, FB) pairs giving positive and negative FBA = FB - FA.
FA_FB_CASES = ((0.3, 1.7), (0.5, -0.9))


def _named_cases():
  for nP, nT in NP_NT_CASES:
    for FA, FB in FA_FB_CASES:
      sign = 'pos' if FB > FA else 'neg'
      yield (f'_nP={nP},nT={nT},FBA_{sign}', nP, nT, FA, FB)


def _np_nt_cases():
  for nP, nT in NP_NT_CASES:
    yield (f'_nP={nP},nT={nT}', nP, nT)


class BfefTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._oct = octave_utils.create_meq_oct2py_instance()

  def _mex(self, mode, nP, nT, *args, nout):
    """Calls the compiled bfefmex with the given mode and parameters."""
    par = np.array([nP, nT], dtype=np.float64)
    return self._oct.bfefmex(mode, par, *args, nout=nout)

  @parameterized.named_parameters(*_named_cases())
  def test_f_scalar(self, nP, nT, FA, FB):
    """Tests f for scalar flux values inside and outside [FA, FB]."""
    ng = nP + nT
    # Flux points inside and outside the [FA, FB] range (i.e. normalized
    # flux inside and outside [0, 1]).
    F = np.array([FA - 0.4, FA, 0.5 * (FA + FB), FB, FB + 0.6])

    g_oct, Ig_oct = self._mex(91, nP, nT, F, FA, FB, nout=2)
    g_oct = np.asarray(g_oct).reshape(F.size, ng)
    Ig_oct = np.asarray(Ig_oct).reshape(F.size, ng)

    jit_f = jax.jit(bfef.f, static_argnums=(2, 3))
    for k, Fk in enumerate(F):
      g, Ig = jit_f(jnp.asarray(Fk - FA), jnp.asarray(FB - FA), nP, nT)
      self.assertEqual(g.shape, (ng,))
      self.assertEqual(Ig.shape, (ng,))
      np.testing.assert_allclose(g, g_oct[k], rtol=1e-12, atol=1e-12)
      np.testing.assert_allclose(Ig, Ig_oct[k], rtol=1e-12, atol=1e-12)

  @parameterized.named_parameters(*_named_cases())
  def test_f_grid(self, nP, nT, FA, FB):
    """Tests f for a grid-shaped flux input via vmap."""
    ng = nP + nT
    nz, nr = 5, 7
    prng = jax.random.PRNGKey(0)
    # Normalized flux values inside and outside [0, 1].
    FN = jax.random.uniform(prng, (nz, nr), minval=-0.25, maxval=1.25)
    Fx = FA + (FB - FA) * FN
    FBA = jnp.asarray(FB - FA)

    jit_f = jax.jit(
        jax.vmap(jax.vmap(lambda fxa: bfef.f(fxa, FBA, nP, nT)))
    )
    g_jax, Ig_jax = jit_f(Fx - FA)
    self.assertEqual(g_jax.shape, (nz, nr, ng))

    g_oct, Ig_oct = self._mex(91, nP, nT, np.asarray(Fx), FA, FB, nout=2)
    g_oct = np.asarray(g_oct).reshape(nz * nr, ng)
    Ig_oct = np.asarray(Ig_oct).reshape(nz * nr, ng)

    # Octave flattens the (nz, nr) grid column-major, so transpose the JAX
    # result before flattening for the comparison.
    g_jax = np.asarray(g_jax).transpose(1, 0, 2).reshape(nz * nr, ng)
    Ig_jax = np.asarray(Ig_jax).transpose(1, 0, 2).reshape(nz * nr, ng)
    np.testing.assert_allclose(g_jax, g_oct, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Ig_jax, Ig_oct, rtol=1e-12, atol=1e-12)

  @parameterized.named_parameters(*_np_nt_cases())
  def test_fN(self, nP, nT):
    """Tests fN for normalized flux values inside and outside [0, 1]."""
    ng = nP + nT
    FN = np.linspace(-0.5, 1.5, 17)  # Includes 0.0 and 1.0 exactly.

    gQ_oct, IgQ_oct = self._mex(2, nP, nT, FN, 0.0, 1.0, nout=2)
    gQ_oct = np.asarray(gQ_oct).reshape(FN.size, ng)
    IgQ_oct = np.asarray(IgQ_oct).reshape(FN.size, ng)

    # Grid-shaped input via vmap.
    jit_fN = jax.jit(jax.vmap(lambda fn: bfef.fN(fn, nP, nT)))
    gQ_jax, IgQ_jax = jit_fN(jnp.asarray(FN))
    np.testing.assert_allclose(gQ_jax, gQ_oct, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(IgQ_jax, IgQ_oct, rtol=1e-12, atol=1e-12)

    # Scalar input.
    jit_fN_scalar = jax.jit(bfef.fN, static_argnums=(1, 2))
    g0, Ig0 = jit_fN_scalar(jnp.asarray(FN[3]), nP, nT)
    np.testing.assert_allclose(g0, gQ_oct[3], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Ig0, IgQ_oct[3], rtol=1e-12, atol=1e-12)

  @parameterized.named_parameters(*_named_cases())
  def test_fA(self, nP, nT, FA, FB):
    """Tests fA (basis functions at the magnetic axis)."""
    ng = nP + nT
    gA_oct, IgA_oct = self._mex(5, nP, nT, 0.0, FA, FB, nout=2)
    gA_oct = np.asarray(gA_oct).reshape(ng)
    IgA_oct = np.asarray(IgA_oct).reshape(ng)

    jit_fA = jax.jit(bfef.fA, static_argnums=(1, 2))
    gA_jax, IgA_jax = jit_fA(jnp.asarray(FB - FA), nP, nT)
    np.testing.assert_allclose(gA_jax, gA_oct, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(IgA_jax, IgA_oct, rtol=1e-12, atol=1e-12)

  @parameterized.named_parameters(*_named_cases())
  def test_fag(self, nP, nT, FA, FB):
    """Tests fag (normalized-to-physical conversion factors)."""
    ng = nP + nT
    par = np.array([nP, nT], dtype=np.float64)
    ag = np.ones((ng, 1))
    fPg_col = np.ones((ng, 1))
    fTg_col = np.zeros((ng, 1))
    ids = 2.0 * np.pi  # Makes the cP = ids/2/pi factor exactly 1 in MATLAB.

    jit_fag = jax.jit(bfef.fag, static_argnums=(1, 2))
    alphapg_jax, alphag_jax = jit_fag(jnp.asarray(FB - FA), nP, nT)

    # Pure-MATLAB reference (bfef.m mode 3, routed through bfct.m).
    aPpg, _, aPg, _ = self._oct.bfef(
        3, par, ag, FA, FB, fPg_col, fTg_col, ids, nout=4
    )
    np.testing.assert_allclose(
        alphapg_jax, np.asarray(aPpg).reshape(ng), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        alphag_jax, np.asarray(aPg).reshape(ng), rtol=1e-12, atol=1e-12
    )

    # Compiled MEX reference. Unlike the .m route, the generic C bfctmex
    # mode 3 ignores the fPg/fTg arguments and uses the canonical ones from
    # its parameter struct, so the P' and TT' parts must be recombined:
    # aPpg = cP*alphapg*fPg and aTTpg = cT*alphapg*fTg with cT = ids*2e-7.
    # With ids = 2*pi, cT = mu0 and cP = ids*0.159154943091895 is 1 only to
    # ~2e-15 relative, still well within the 1e-12 tolerance.
    aPpg_mex, aTTpg_mex, aPg_mex, ahqTg_mex = self._mex(
        3, nP, nT, ag, FA, FB, fPg_col, fTg_col, ids, nout=4
    )
    cT = ids * 2e-7
    alphapg_mex = (
        np.asarray(aPpg_mex).reshape(ng)
        + np.asarray(aTTpg_mex).reshape(ng) / cT
    )
    alphag_mex = (
        np.asarray(aPg_mex).reshape(ng)
        + np.asarray(ahqTg_mex).reshape(ng) / cT
    )
    np.testing.assert_allclose(alphapg_jax, alphapg_mex, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(alphag_jax, alphag_mex, rtol=1e-12, atol=1e-12)

  @parameterized.named_parameters(*_np_nt_cases())
  def test_fPg(self, nP, nT):
    """Tests fPg (assignment of basis functions to P' or TT')."""
    ng = nP + nT
    fPg_oct, fTg_oct = self._mex(0, nP, nT, nout=2)

    jit_fPg = jax.jit(bfef.fPg, static_argnums=(0, 1))
    fPg_jax, fTg_jax = jit_fPg(nP, nT)
    np.testing.assert_allclose(
        fPg_jax, np.asarray(fPg_oct).reshape(ng), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        fTg_jax, np.asarray(fTg_oct).reshape(ng), rtol=1e-12, atol=1e-12
    )

  @parameterized.named_parameters(*_np_nt_cases())
  def test_dispatch(self, nP, nT):
    """Tests dispatch against the equivalent Octave indexing."""
    ng = nP + nT
    prng = jax.random.PRNGKey(1)
    prng_g, prng_Ig = jax.random.split(prng)
    g4 = jax.random.normal(prng_g, (6, 4))
    Ig4 = jax.random.normal(prng_Ig, (6, 4))

    # bfef dispatch selects the first nP and first nT of the 4 elementary
    # functions: x(:, [1:nP, 1:nT]) in MATLAB (see bfef.m, bfef_fN).
    self._oct.push('g4', np.asarray(g4))
    self._oct.push('Ig4', np.asarray(Ig4))
    self._oct.eval(
        f'gd = g4(:,[1:{nP},1:{nT}]); Igd = Ig4(:,[1:{nP},1:{nT}]);'
    )
    gd_oct = np.asarray(self._oct.pull('gd')).reshape(6, ng)
    Igd_oct = np.asarray(self._oct.pull('Igd')).reshape(6, ng)

    jit_dispatch = jax.jit(bfef.dispatch, static_argnums=(2, 3))
    # Batched input.
    gd_jax, Igd_jax = jit_dispatch(g4, Ig4, nP, nT)
    np.testing.assert_allclose(gd_jax, gd_oct, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Igd_jax, Igd_oct, rtol=1e-12, atol=1e-12)
    # Single-vector input.
    gd0_jax, Igd0_jax = jit_dispatch(g4[0], Ig4[0], nP, nT)
    np.testing.assert_allclose(gd0_jax, gd_oct[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Igd0_jax, Igd_oct[0], rtol=1e-12, atol=1e-12)


# pylint: enable=invalid-name


if __name__ == '__main__':
  absltest.main()
