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

"""Port of meq/tests/fge_vacuum_tests.m.

With no plasma the circuit equations are linear, so the whole run has a
closed-form reference: implicit Euler on

  Mee dIe/dt + Re Ie = Va,

which is what the MEQ test compares against. That makes this the one FGE
test with an answer that does not come from MEQ itself, and it covers the
vacuum run mode end to end -- 100 time steps through fgetk_environment,
rather than the single residual evaluation fgeF_test.py does.

The same trajectory is also compared against MEQ's own, which is the parity
part; the ODE check on its own would pass even if both implementations were
wrong in the same way.
"""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from meq_jax._src import fgetk_environment
from meq_jax._src import utils
from meqpy import meqpy_impl
import numpy as np

# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)

# pylint: disable=invalid-name

# Anamak shot 91 is the vacuum case with a quadrupole null, and selu='e' with
# nu=30 is the eigenmode model of the passive structure. Same call as
# fge_vacuum_tests.m.
_SETUP = """
Ts = 1e-3;
tv = (-0.3:Ts:-0.2);
[Lfge,LXfge] = fge('ana',91,tv,'selu','e','nu',30);

% fgetk_environment steps one slice at a time, so it is initialized from the
% first slice of the sequence fge returned.
num_steps = 1;
dt = Ts;
LXfge_working = meqxk(LXfge,1);
[LYfge,Stop,State] = fgetk_environment([],[],Lfge,LXfge_working,dt,'init');
"""


class FgeVacuumTest(parameterized.TestCase):

  _meq = None

  @classmethod
  def _meq_instance(cls):
    if cls._meq is None:
      cls._meq = meqpy_impl.MeqPy()
      cls._meq.octave_eval(_SETUP)
    return cls._meq

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    if cls._meq is not None:
      cls._meq.cleanup()
      cls._meq = None

  def setUp(self):
    super().setUp()
    self.meq = self._meq_instance()
    self.scalar = lambda e: float(
        np.asarray(self.meq.octave_eval(e + ';', nout=1)).reshape(-1)[0]
    )
    self.array = lambda e: np.asarray(self.meq.octave_eval(e + ';', nout=1))

  def _reference_ode(self):
    """Implicit Euler on the coil and passive circuit equations."""
    Ts = self.scalar('Ts')
    ne = int(self.scalar('Lfge.ne'))
    na = int(self.scalar('Lfge.G.na'))
    Mee = self.array('Lfge.Mee')
    Re = self.array('Lfge.Re').reshape(-1)
    Va = self.array('LXfge.Va')

    A = np.linalg.solve(Mee + Ts * np.diag(Re), Mee)
    B = np.linalg.solve(Mee + Ts * np.diag(Re), Ts * np.eye(ne, na))

    X = np.zeros((ne, Va.shape[1]))
    X[:, 0] = np.concatenate(
        [self.array('LXfge.Ia')[:, 0], self.array('LXfge.Iu')[:, 0]]
    )
    for kt in range(1, Va.shape[1]):
      X[:, kt] = A @ X[:, kt - 1] + B @ Va[:, kt]
    return X

  def _run_jax(self):
    """Runs the MEQ-JAX environment over the same time base."""
    state, static, lx, _, agconc, cdeconc = utils.init_from_octave(
        self.meq, self.meq.octave_eval('LYfge;', nout=1)
    )
    step = jax.jit(
        functools.partial(
            fgetk_environment.fgetk_environment,
            static=static, lx=lx, num_steps=1,
            agconc=agconc, cdeconc=cdeconc,
        )
    )

    Va = self.array('LXfge.Va')
    nt = Va.shape[1]
    Ia = np.zeros((int(self.scalar('Lfge.G.na')), nt))
    Iu = np.zeros((int(self.scalar('Lfge.G.nu')), nt))
    Ia[:, 0] = np.asarray(state.LYt.Ia).reshape(-1)
    Iu[:, 0] = np.asarray(state.LYt.Iu).reshape(-1)

    converged = []
    for kt in range(1, nt):
      _, _, state, info = step(state=state, voltages=Va[:, kt])
      Ia[:, kt] = np.asarray(state.LYt.Ia).reshape(-1)
      Iu[:, kt] = np.asarray(state.LYt.Iu).reshape(-1)
      converged.append(bool(info['isconverged']))
    return Ia, Iu, converged

  def test_vs_ODE(self):
    X = self._reference_ode()
    Ia, Iu, converged = self._run_jax()
    na = int(self.scalar('Lfge.G.na'))

    self.assertTrue(all(converged), 'FGE did not converge in vacuum')

    # Same tolerance as fge_vacuum_tests.m: 1e-12 of the current scale. The
    # worst deviation observed here is about 3e-15 of it, and MEQ's own is of
    # the same order, so the margin is the MEQ test's, not extra slack.
    tol = 1e-12
    np.testing.assert_allclose(
        Ia, X[:na], atol=tol * self.scalar('Lfge.Ia0'), rtol=0,
        err_msg='coil currents differ from the analytic ODE',
    )
    np.testing.assert_allclose(
        Iu, X[na:], atol=tol * self.scalar('Lfge.Iu0'), rtol=0,
        err_msg='passive currents differ from the analytic ODE',
    )

  def test_vs_octave(self):
    self.meq.octave_eval('LYoct = fget(Lfge,LXfge);')
    Ia, Iu, _ = self._run_jax()

    tol = 1e-12
    np.testing.assert_allclose(
        Ia, self.array('LYoct.Ia'), atol=tol * self.scalar('Lfge.Ia0'), rtol=0,
        err_msg='coil currents differ from Octave',
    )
    np.testing.assert_allclose(
        Iu, self.array('LYoct.Iu'), atol=tol * self.scalar('Lfge.Iu0'), rtol=0,
        err_msg='passive currents differ from Octave',
    )


if __name__ == '__main__':
  absltest.main()

# pylint: enable=invalid-name
