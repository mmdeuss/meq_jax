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

"""Test of bfct functions."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import bfct
from meq_jax._src import bfct_router
from meq_jax._src import types
from meqpy import octave_utils
import numpy as np


# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)

# pylint: disable=invalid-name

NAMED_PARAMS = (
    ('Np=3,nT=3', (3, 3)),
    ('Np=1,nT=2', (1, 2)),
    ('Np=0,nT=1', (0, 1)),
    ('Np=1,nT=0', (1, 0)),
)


class BfctTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._oct = octave_utils.create_meq_oct2py_instance()
    cls._oct.eval("[L,~,LY] = fbt('ana',2); [Lry] = L.ry; [Liry] = L.iry;")

  @parameterized.named_parameters(*NAMED_PARAMS)
  def test_bfct_mode1(self, np_nt):
    nP, nT = np_nt
    LY, ry, iry = self._oct.pull(['LY', 'Lry', 'Liry'])
    Fx = LY['Fx']
    Opy = LY['Opy']
    FA = jnp.squeeze(LY['FA'])
    FB = jnp.squeeze(LY['FB'])
    ry = jnp.squeeze(ry)
    iry = jnp.squeeze(iry)

    # Python implementation of bfct1
    Bfp = types.BfpData(nP=nP, nT=nT)
    jit_bfct1 = jax.jit(functools.partial(bfct_router.bfct1, Bfp=Bfp))
    Tyg, TpDg, _, ITpDg = jit_bfct1(
        Fx=Fx.T, F0=FA, F1=FB, Opy=Opy.T, ry=ry, iry=iry
    )

    # Matlab implementation of bfab (mode 1). Note: bfabmex is the C equivalent.
    # and could be used as well.
    (Tyg_, TpDg_, ITpDg_) = self._oct.bfab(
        1, [nP, nT], Fx, FA, FB, Opy, ry, iry, nout=3
    )

    # Compare the results. Tyg is in row-major order, so we transpose it to
    # compare with Matlab's Tyg_.
    np.testing.assert_allclose(Tyg.T, Tyg_, atol=1e-13)
    np.testing.assert_allclose(TpDg, TpDg_[0], atol=1e-13)
    np.testing.assert_allclose(ITpDg, ITpDg_[0], atol=1e-13)

  @parameterized.named_parameters(*NAMED_PARAMS)
  def test_bfct_mode2(self, np_nt):
    nP, nT = np_nt
    n = 41
    FN = jnp.linspace(0, 1, n)

    # Python implementation of bfct2
    Bfp = types.BfpData(nP=nP, nT=nT)
    jit_bfct2 = jax.jit(functools.partial(bfct_router.bfct2, Bfp=Bfp))
    gQg, IgQg = jit_bfct2(FN=FN)

    # Matlab implementation of bfab (mode 2). Note: bfabmex is the C equivalent.
    # and could be used as well.
    (gQg_, IgQg_) = self._oct.bfab(2, [nP, nT], FN, nout=2)
    np.testing.assert_allclose(gQg, gQg_.T)
    np.testing.assert_allclose(IgQg, IgQg_.T)

  @parameterized.named_parameters(*NAMED_PARAMS)
  def test_bfct_mode3(self, np_nt):
    nP, nT = np_nt
    LY = self._oct.pull('LY')
    ids = self._oct.eval('L.idsx;').item()
    FA = jnp.squeeze(LY['FA'])
    FB = jnp.squeeze(LY['FB'])
    prng = jax.random.PRNGKey(0)
    # ag needs to have shape (nP+nT,) which is not the case for LY.ag
    ag = jax.random.normal(prng, (nP + nT,))
    fPg, fTg = bfct.bfab.fPg(nP, nT)

    # NOTE: changing this to bfab fails the test, there is some inconsistency
    # between the two implementations of bfab3 in octave. The C version is
    # is the preferred one so we use it here.
    (aPpg_, aTTpg_, aPg_, ahqTg_) = self._oct.bfabmex(
        3, [nP, nT], ag, FA, FB, fPg, fTg, ids, nout=4
    )
    Bfp = types.BfpData(nP=nP, nT=nT)
    jit_bfct3 = jax.jit(functools.partial(bfct_router.bfct3, Bfp=Bfp))
    aPpg, aTTpg, aPg, ahqTg = jit_bfct3(ag=ag, F0=FA, F1=FB, ids=ids)

    # Matlab implementation of bfab (mode 3). Note: bfabmex is the C equivalent.
    # and could be used as well.
    np.testing.assert_allclose(aTTpg, aTTpg_.squeeze())
    np.testing.assert_allclose(aPpg, aPpg_.squeeze())
    np.testing.assert_allclose(aPg, aPg_.squeeze())
    np.testing.assert_allclose(ahqTg, ahqTg_.squeeze())

  @parameterized.named_parameters(*NAMED_PARAMS)
  def test_bfct_mode5(self, np_nt):
    nP, nT = np_nt
    LY = self._oct.pull('LY')
    FA = jnp.squeeze(LY['FA'])
    FB = jnp.squeeze(LY['FB'])
    # Python implementation of bfct5
    Bfp = types.BfpData(nP=nP, nT=nT)
    jit_bfct5 = jax.jit(functools.partial(bfct_router.bfct5, Bfp=Bfp))
    gA, IgA = jit_bfct5(F0=FA, F1=FB)

    # Matlab implementation of bfab (mode 5). Note: bfabmex is the C equivalent.
    # and could be used as well.
    (gA_, IgA_) = self._oct.bfab(5, [nP, nT], LY['Fx'], FA, FB, nout=2)
    np.testing.assert_allclose(gA, gA_[0], atol=1e-13)
    np.testing.assert_allclose(IgA, IgA_[0], atol=1e-13)

  @parameterized.named_parameters(*NAMED_PARAMS)
  def test_bfct_mode_11(self, np_nt: tuple[int, int]):
    nP, nT = np_nt
    LY, ry, iry = self._oct.pull(['LY', 'Lry', 'Liry'])
    Fx = LY['Fx']
    Opy = LY['Opy']
    FA = jnp.squeeze(LY['FA'])
    FB = jnp.squeeze(LY['FB'])
    ry = jnp.squeeze(ry)
    iry = jnp.squeeze(iry)

    # Python implementation of bfct11
    Bfp = types.BfpData(nP=nP, nT=nT)
    jit_bfct11 = jax.jit(functools.partial(bfct_router.bfct11, Bfp=Bfp))

    jacs = jit_bfct11(Fx=Fx.T, F0=FA, F1=FB, Opy=Opy.T, ry=ry, iry=iry)
    jacs_ = self._oct.bfab(11, [nP, nT], Fx, FA, FB, Opy, ry, iry, nout=5)

    self.assertEqual(len(jacs), len(jacs_))
    for jac, jac_ in zip(jacs, jacs_):
      np.testing.assert_allclose(jac, jac_.T, atol=1e-13)

  @parameterized.named_parameters(*NAMED_PARAMS)
  def test_bfct_mode_15(self, np_nt):
    nP, nT = np_nt
    LY = self._oct.pull('LY')
    FA = jnp.squeeze(LY['FA'])
    FB = jnp.squeeze(LY['FB'])
    # Python implementation of bfct5
    Bfp = types.BfpData(nP=nP, nT=nT)
    jit_bfct15 = jax.jit(functools.partial(bfct.bfct15, Bfp=Bfp))
    jacs = jit_bfct15(F0=FA, F1=FB)

    # Matlab implementation of bfab (mode 15). Note: bfabmex is the C equivalent
    # and could be used as well.
    jacs_ = self._oct.bfab(15, [nP, nT], LY['Fx'], FA, FB, nout=4)

    self.assertEqual(len(jacs), len(jacs_))
    for jac, jac_ in zip(jacs, jacs_):
      np.testing.assert_allclose(jac, jac_.squeeze(), atol=1e-13)

  @parameterized.named_parameters(*NAMED_PARAMS)
  def test_bfct_mode_91(self, np_nt):
    nP, nT = np_nt
    LY = self._oct.pull('LY')
    Fx = LY['Fx']
    FA = jnp.squeeze(LY['FA'])
    FB = jnp.squeeze(LY['FB'])
    # Python implementation of bfct91
    Bfp = types.BfpData(nP=nP, nT=nT)
    jit_bfct91 = jax.jit(functools.partial(bfct_router.bfct91, Bfp=Bfp))
    gA, IgA = jit_bfct91(F=Fx.T, F0=FA, F1=FB)

    # Matlab implementation of bfab (mode 15). Note: bfabmex is the C equivalent
    # and could be used as well.
    gA_, IgA_ = self._oct.bfab(91, [nP, nT], LY['Fx'], FA, FB, nout=2)

    np.testing.assert_allclose(gA, gA_.T, atol=1e-13)
    np.testing.assert_allclose(IgA, IgA_.T, atol=1e-13)


# pylint: enable=invalid-name


if __name__ == '__main__':
  absltest.main()
