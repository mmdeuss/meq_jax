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

"""Port of meq/tests/fge_constraint_tests.m.

Drives the constrained scalars (plasma current, poloidal or toroidal beta,
and either edge safety factor or internal inductance) along a 300 Hz
sinusoid and checks that the solver tracks them over the whole run. Like the
vacuum test, the reference is the input rather than MEQ's output: a solver
that satisfies its own constraints is right by construction, whatever MEQ
does.

This exercises the four constraint sets MEQ supports for a single domain,
with the targets changing every time step -- nothing else in the suite
varies LX during a run.
"""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import fgetk_environment
from meq_jax._src import utils
from meqpy import meqpy_impl
import numpy as np

# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)

# pylint: disable=invalid-name

# Solver tolerance, and the time base, from fge_constraint_tests.m.
_TOL = 1e-9
_SETUP = """
tok = 'ana'; shot = 1; tstart = 0; tend = 0.005; dt = 5e-5;
Lbase = fge(tok,shot,tstart,'debug',0,'tolF',{tol},...
            'selu','e','nu',30,'izgrid',true);
"""

# Re-consolidating L is what makes the constraint set take effect; fgex then
# builds a target for each time slice.
_CASE = """
Lfge = Lbase;
Lfge.P.agcon = {{{agcon}}};
Lfge = fgec(Lfge.P,Lfge.G);

tvec = tstart:dt:tend;
LXfge = fgex(tok,tvec,Lfge);
for iC = 1:Lfge.nC
  fld = Lfge.agconc{{iC,3}};
  LXfge.(fld) = LXfge.(fld) + LXfge.(fld)(1)*0.001*sin(2*pi*300*LXfge.t);
end

num_steps = 1;
LXfge_working = meqxk(LXfge,1);
[LYfge,Stop,State] = fgetk_environment([],[],Lfge,LXfge_working,dt,'init');
"""

_CONSTRAINTS = [
    dict(testcase_name='Ip_bp_qA', agcon="'Ip','bp','qA'"),
    dict(testcase_name='Ip_bp_li', agcon="'Ip','bp','li'"),
    dict(testcase_name='Ip_bt_qA', agcon="'Ip','bt','qA'"),
    dict(testcase_name='Ip_bt_li', agcon="'Ip','bt','li'"),
]


class FgeConstraintTest(parameterized.TestCase):

  _meq = None

  @classmethod
  def _meq_instance(cls):
    """Returns a shared MEQ session with the (expensive) fge init done."""
    if cls._meq is None:
      cls._meq = meqpy_impl.MeqPy()
      cls._meq.octave_eval(_SETUP.format(tol=_TOL))
    return cls._meq

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    if cls._meq is not None:
      cls._meq.cleanup()
      cls._meq = None

  @parameterized.named_parameters(_CONSTRAINTS)
  def test_fge_constraints(self, agcon):
    meq = self._meq_instance()
    meq.octave_eval(_CASE.format(agcon=agcon))

    array = lambda e: np.asarray(meq.octave_eval(e + ';', nout=1))
    scalar = lambda e: float(array(e).reshape(-1)[0])

    fields = [
        str(f) for f in array('Lfge.agconc(:,3);').reshape(-1)
    ]
    targets = {f: array(f'LXfge.{f}').reshape(-1) for f in fields}

    state, static, lx, _, agconc, cdeconc = utils.init_from_octave(
        meq, meq.octave_eval('LYfge;', nout=1)
    )
    step = jax.jit(
        functools.partial(
            fgetk_environment.fgetk_environment,
            static=static, lx=lx, num_steps=1,
            agconc=agconc, cdeconc=cdeconc,
        )
    )

    Va = array('LXfge.Va')
    nt = Va.shape[1]
    got = {f: np.zeros(nt) for f in fields}
    for f in fields:
      got[f][0] = float(np.asarray(getattr(state.LYt, f)).reshape(-1)[0])

    for kt in range(1, nt):
      # The constrained scalars move every step, which is what fge_variation
      # is for: it overrides those LX fields for this slice only.
      variation = {f: jnp.asarray(targets[f][kt]) for f in fields}
      _, _, state, info = step(
          state=state, voltages=Va[:, kt], fge_variation=variation
      )
      self.assertTrue(bool(info['isconverged']), f'step {kt} did not converge')
      for f in fields:
        got[f][kt] = float(np.asarray(getattr(state.LYt, f)).reshape(-1)[0])

    # Tolerances from the MEQ test: ten times the solver tolerance, times the
    # scale of the quantity.
    scale = {'Ip': scalar('Lfge.Ip0'), 'bt': 100.0}
    for f in fields:
      np.testing.assert_allclose(
          got[f], targets[f], atol=10 * scale.get(f, 1.0) * _TOL, rtol=0,
          err_msg=f'did not satisfy the constraint on {f}',
      )


if __name__ == '__main__':
  absltest.main()

# pylint: enable=invalid-name
