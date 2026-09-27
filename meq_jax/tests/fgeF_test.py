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

"""Tests for fgeF.py, based on liuNLmeas_jacobian_test.m.

The matrix below is also the parity scoreboard for fgeF. Every configuration
MEQ supports is listed; those MEQ-JAX has not reached yet are generated as
skipped cases carrying the reason, rather than commented out. ``pytest -rs``
prints what is left.

One gap does not fit the matrix: the Newton-Raphson check at the end of
``test_fgeF`` runs the solve but asserts nothing (TODO(adedieu)), so
convergence is unverified for every configuration.
"""

import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meq_jax._src import fgeF
from meq_jax._src import fgeF_octave
from meq_jax._src import jax_root_finding
from meqpy import octave_utils
import numpy as np

# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)


# Parity status of each axis of the test matrix. Each axis maps every value
# MEQ supports to either None, meaning MEQ-JAX is at parity and the case runs,
# or a reason why not, in which case the case is generated but skipped. To
# close a gap, set it to None and fix what the comparison then shows.
_DOUBLETS = 'doublets not fully supported (BfpData, multi-domain CDE)'

_SHOTS = {
    'circular': 1,
    'diverted': 2,
    'diverted2': 3,
    'squashed': 5,
    'doublet': 82,
    'droplets': 84,
    'doublet_with_mantle_current': 88,
}

_TASKS = {
    'circular': None,
    'diverted': None,
    'diverted2': None,
    'squashed': None,
    'doublet': _DOUBLETS,
    'droplets': _DOUBLETS,
    'doublet_with_mantle_current': _DOUBLETS,
}

_TEST_TYPES = {
    'fbt': None,
    'fge': 'evolutive (fge) path not yet verified against Octave',
}

_ICSINT = {
    0: None,
    1: 'icsint=1 not yet verified against Octave',
}

_ALGOSNL = {
    'all-nl': None,
    'all-nl-Fx': "algoNL 'all-nl-Fx' not yet verified against Octave",
    'Newton-GS': "algoNL 'Newton-GS' is broken (TODO(adedieu))",
}


def _at_parity(axis):
  """Returns the values of an axis that MEQ-JAX matches Octave on."""
  return [value for value, reason in axis.items() if reason is None]


def _baseline(axis):
  """Returns the value an axis is held at while another axis is varied."""
  values = _at_parity(axis)
  if not values:
    raise ValueError('no value on this axis is at parity')
  return values[0]


def _make_case(task, test_type, icsint, algosNL, reason):
  return dict(
      testcase_name=(
          f'{task}_{test_type}_algosNL_{algosNL}_icsint_{icsint}'
      ),
      test_type=test_type,
      algosNL=algosNL,
      shot=_SHOTS[task],
      icsint=icsint,
      skip_reason=reason,
  )


# Cross product over the values at parity, so coverage grows back as gaps
# close.
_TEST_CASES = [
    _make_case(task, test_type_, icsint_, algosNL_, None)
    for task, test_type_, icsint_, algosNL_ in itertools.product(
        _at_parity(_TASKS),
        _at_parity(_TEST_TYPES),
        _at_parity(_ICSINT),
        _at_parity(_ALGOSNL),
    )
]

# One case per remaining gap, with the other axes at baseline, so
# un-skipping a case isolates a single feature.
_BASELINE_CASE = dict(
    task=_baseline(_TASKS),
    test_type=_baseline(_TEST_TYPES),
    icsint=_baseline(_ICSINT),
    algosNL=_baseline(_ALGOSNL),
)
for _axis_name, _axis in (
    ('task', _TASKS),
    ('test_type', _TEST_TYPES),
    ('icsint', _ICSINT),
    ('algosNL', _ALGOSNL),
):
  for _value, _reason in _axis.items():
    if _reason is not None:
      _TEST_CASES.append(
          _make_case(**{**_BASELINE_CASE, _axis_name: _value}, reason=_reason)
      )






# pylint: disable=invalid-name
class FgeFTest(parameterized.TestCase):

  _oct_session = None

  @classmethod
  def _octave(cls):
    """Returns a shared Octave session, created on first use.

    Created lazily rather than in setUpClass so that cases skipped for lack
    of parity are reported as skips even where MEQ is not installed.
    """
    if cls._oct_session is None:
      cls._oct_session = octave_utils.create_meq_oct2py_instance()
    return cls._oct_session

  @parameterized.named_parameters(_TEST_CASES)
  def test_fgeF(self, test_type, algosNL, shot, icsint, skip_reason):
    """Test computation of residual and jacobian.

    Args:
      test_type: The test type.
      algosNL: The non-linear solver to use. Might not be necessary?
      shot: The shot number.
      icsint: Whether to use icsint.
      skip_reason: Why this configuration is not yet at parity with Octave
        MEQ, or None if it is expected to pass.
    """
    if skip_reason is not None:
      self.skipTest(skip_reason)

    self._oct = self._octave()

    if test_type == 'fbt':
      cmd = fgeF_octave.FBT_SETUP
    elif test_type == 'fge':
      cmd = fgeF_octave.FGE_SETUP
    else:
      raise ValueError(f'Unknown test type: {test_type}')

    print(f'Running test: {test_type}, {algosNL}, {shot}, {icsint}')
    cmd = cmd.format(algosNL=algosNL, shot=shot, icsint=icsint, grid='')
    self._oct.eval(cmd)
    print('Done running octave commands')

    inputs = fgeF_octave.fgeF_inputs_from_octave(
        self._oct, shot=shot, test_type=test_type
    )
    x0 = inputs.x0
    u0 = inputs.u0
    L = inputs.L
    LX = inputs.LX
    opts = inputs.opts
    aux_list = inputs.aux
    agconc = inputs.agconc
    cdeconc = inputs.cdeconc

    # Step 1. Run JAX code
    fgeF_jit = jax.jit(
        functools.partial(
            fgeF.fgeF,
            L=L,
            # TODO(adedieu): make LX dynamic
            # LX=LX,
            opts=opts,
            aux=aux_list,
            agconc=agconc,
            cdeconc=cdeconc,
            u=u0,
        )
    )
    res_jax = fgeF_jit(x=x0, LX=LX, LYp=LX, xdot=None)

    residuals_jax = res_jax[0]
    LY_jax = res_jax[1]
    Jx_jax = res_jax[2]
    Ju_jax = res_jax[3]
    Jxdot_jax = res_jax[4]
    rowmask_jax = res_jax[5]

    # Step 2. Run Matlab code
    res_matlab = self._oct.eval(
        """
    [res, LY, Jx, Ju, Jxdot, rowmask] = fgeF(x0,L,LX,LX,opts,aux,u0);
        """,
        nout=6,
    )

    residuals_matlab = res_matlab[0]
    LY_matlab = res_matlab[1]
    Jx_matlab = res_matlab[2]
    Ju_matlab = res_matlab[3]
    Jxdot_matlab = res_matlab[4]
    rowmask_matlab = res_matlab[5]

    def _compare(A_jax, A_matlab):
      """Utils for comparing the JAX and Matlab structures."""
      A_jax_names = [x for x in dir(A_jax) if not x.startswith('_')]

      for k, v_matlab in A_matlab.items():
        v_matlab = jnp.atleast_1d(jnp.array(v_matlab).squeeze())
        if k in A_jax_names:
          v_jax = getattr(A_jax, k)
          if v_jax is None:
            assert v_matlab.shape[0] == 0
            continue

          # Compare the results
          if isinstance(v_jax, jax.Array):

            # There are some division errors in Matlab
            if v_matlab.squeeze().shape == ():
              val_matlab = float(v_matlab.item())
              if jnp.isnan(val_matlab):
                val_jax = float(v_jax.item())
                if not jnp.isnan(val_jax):
                  np.testing.assert_allclose(val_jax, 0.0)
                  continue

            # General case
            v_jax = jnp.atleast_1d(v_jax.squeeze())
            if v_jax.ndim == 2:
              v_jax = v_jax.T
            if v_jax.ndim == 3:
              v_jax = v_jax.transpose((2, 1, 0))

            if k == 'lX':
              # Copied from meqpdom_test.py
              np.testing.assert_allclose(
                  v_matlab, v_jax[np.isfinite(LY_jax.rB)], rtol=1e-8, atol=1e-8
              )
              continue

            if k in ['zYD', 'rYD']:
              v_matlab = v_matlab[: LY_jax.nA]
              v_jax = v_jax[: LY_jax.nA]
              np.testing.assert_allclose(
                  v_matlab[np.isfinite(v_matlab)],
                  v_jax[np.isfinite(v_matlab)],
                  rtol=1e-8,
                  atol=1e-8,
              )
              continue

            if k in ['PpQg', 'PpQ', 'PQ']:
              # Large values being multipled in multi-domains
              atol = 1e-4
            else:
              atol = 1e-8
            np.testing.assert_allclose(
                v_matlab[np.isfinite(v_matlab)],
                v_jax[np.isfinite(v_jax)],
                rtol=1e-8,
                atol=atol,
            )

          else:
            np.testing.assert_allclose(
                float(v_matlab.item()), float(v_jax), rtol=1e-8, atol=1e-8
            )

    # Compare LY structures
    _compare(LY_jax, LY_matlab)

    # Compare residuals
    np.testing.assert_allclose(
        residuals_jax, residuals_matlab.squeeze(), rtol=1e-8, atol=1e-8
    )

    # Jx
    np.testing.assert_allclose(Jx_jax, Jx_matlab.T, rtol=1e-8, atol=1e-8)

    # Ju
    np.testing.assert_allclose(Ju_jax, Ju_matlab.T, rtol=1e-8, atol=1e-8)

    # Jxdot
    if Jxdot_matlab.shape[0] > 0:
      np.testing.assert_allclose(
          Jxdot_jax, Jxdot_matlab.T, rtol=1e-8, atol=1e-8
      )

    # rowmask
    np.testing.assert_allclose(
        rowmask_jax, rowmask_matlab.squeeze(), rtol=1e-8, atol=1e-8
    )

    fun = lambda x: fgeF_jit(x=x, LYp=LX, LX=LX, xdot=None)[0]
    jac = lambda x: fgeF_jit(x=x, LYp=LX, LX=LX, xdot=None)[2].T
    _, metadata = jax_root_finding.root_newton_raphson(
        fun, x0, custom_jac=jac, use_jax_custom_root=False, tol=1e-13)
    # TODO(adedieu): fix this test
    # self.assertTrue(
    #     metadata.error == 0 and np.linalg.norm(metadata.residual) < 1e-13)


if __name__ == '__main__':
  absltest.main()

# pylint: enable=invalid-name
