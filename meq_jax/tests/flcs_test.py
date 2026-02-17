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

"""Test of flcs."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from meq_jax._src import flcs
from meq_jax._src import types
from meq_jax._src import utils
from meqpy import meqpy_impl
import numpy as np


# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)


# pylint: disable=invalid-name
class PlasmaDomainTest(parameterized.TestCase):

  # @parameterized.parameters(1, -1)
  @parameterized.named_parameters(
      ('limiter, FN=1', 1, 1),
      ('single null, FN=1', 2, 1),
      ('elongated, FN=1', 11, 1),
      ('doublet, FN=1', 82, 1),
      ('limiter, FN=-1', 1, -1),
      ('single null, FN=-1', 2, -1),
      ('elongated, FN=-1', 11, -1),
      ('doublet, FN=-1', 82, -1),
  )
  def test_flcs_and_jacobian(self, shot, FN):
    meqpy = meqpy_impl.MeqPy()
    meqpy.init_fge(tokamak='ana', shot=shot, time=0,
                   source=meqpy_impl.MeqSource.MEQ_DIRECT,
                   icsint=True, ilim=3)
    get_int_field = lambda x: int(meqpy.octave_eval(f'Lfge.{x};', nout=1)[0, 0])

    # Get fields from Lfge.P
    tolcs = meqpy.octave_eval('Lfge.P.tolcs;', nout=1)[0, 0]
    P = types.ParameterData(itercs=get_int_field('P.itercs'),
                            tolcs=tolcs)
    G = types.GeometryData(rl=meqpy.octave_eval('Lfge.G.rl;', nout=1),
                           zl=meqpy.octave_eval('Lfge.G.zl;', nout=1),
                           rx=meqpy.octave_eval('Lfge.G.rx;', nout=1),
                           zx=meqpy.octave_eval('Lfge.G.zx;', nout=1))
    L = types.StaticData(G=G,
                         P=P,
                         taul=meqpy.octave_eval('Lfge.taul;', nout=1),
                         taur=meqpy.octave_eval('Lfge.taur;', nout=1),
                         tauz=meqpy.octave_eval('Lfge.tauz;', nout=1),
                         Mr=meqpy.octave_eval('Lfge.Mr;', nout=1),
                         Mz=meqpy.octave_eval('Lfge.Mz;', nout=1),
                         Ml=meqpy.octave_eval('Lfge.Ml;', nout=1),
                         nrx=get_int_field('nrx'),
                         nzx=get_int_field('nzx'),
                         idrx=meqpy.octave_eval('Lfge.idrx;', nout=1)[0, 0],
                         idzx=meqpy.octave_eval('Lfge.idzx;', nout=1)[0, 0])

    # Python implementation of flcs
    L = utils.transpose(L)
    Fx = meqpy.octave_eval('LXfge.Fx;', nout=1).T
    Fl, rl, zl, drFl, dzFl = jax.jit(
        functools.partial(flcs.flcs, FN=FN, L=L))(Fx)
    dFl = jax.jit(functools.partial(flcs.jac, L=L))(rl, zl)

    # Original Matlab implementation of flcs
    Fl_, rl_, zl_, drFl_, dzFl_ = meqpy.octave_eval(
        f'[Fl, rl, zl, drFl, dzFl] = flcs(LXfge.Fx, {FN}, Lfge);', nout=5)
    dFl_ = meqpy.octave_eval('dFl = flcsJac(rl, zl, Lfge);', nout=1)

    atol = 1e-10
    np.testing.assert_allclose(Fl, Fl_[:, 0], atol=atol)
    np.testing.assert_allclose(rl, rl_[:, 0], atol=atol)
    np.testing.assert_allclose(zl, zl_[:, 0], atol=atol)
    np.testing.assert_allclose(drFl, drFl_[:, 0], atol=atol)
    np.testing.assert_allclose(dzFl, dzFl_[:, 0], atol=atol)
    np.testing.assert_allclose(dFl, dFl_, atol=atol)
# pylint: enable=invalid-name


if __name__ == '__main__':
  absltest.main()
