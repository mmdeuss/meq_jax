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

"""Implicit Euler dt step for NL system."""
import functools
from typing import Any

import jax
import jax.numpy as jnp

from meq_jax._src import fgeF
from meq_jax._src import jax_root_finding
from meq_jax._src import types


def fgetk_implicit(
    xnlp: jnp.ndarray,
    xnlpdot: jnp.ndarray,
    l: types.StaticData,
    lxt: types.InputData,
    lyp: types.OutputData,
    prec: jnp.ndarray | None,
    agconc: list[types.ConcData] | None,
    cdeconc: list[types.ConcData] | None,
) -> tuple[jnp.ndarray, types.OutputData, dict[str, Any]]:
  """Implicit Euler dt step for NL system. Python/JAX version."""
  del prec  # Preconditioner not used with TORAX Newton solver.

  # F options
  opts = types.Opts(dojacx=True, dopost=True)

  # NL residual operator and initial guess
  assert l.P is not None
  iep = jnp.concatenate([lyp.Ia, lyp.Iu])
  if l.P.algoNL == 'picard':
    raise NotImplementedError('picard not implemented')
  else:  # 'all-nl','all-nl-Fx','Newton-GS'
    # operator
    f_eval = jax.jit(functools.partial(
        fgeF.fgeF,
        L=l,
        LX=lxt,
        opts=opts,
        # Setting the final aux to True: don't compute approximate Jacobian
        aux=[None] * 5 + [True],
        u=None,
        xdot=None,
        agconc=agconc,
        cdeconc=cdeconc,
        LYp=lyp,
    ))

    # initial condition
    xnl0s = []
    init_guesses = l.P.initguess
    if init_guesses is None:
      init_guesses = ['extrap', 'previous']

    for init_mode in init_guesses:
      if init_mode == 'vacuum':
        # initial guess passed via xnl0
        assert (l.ind is not None and
                l.ind.ixa is not None and
                l.ind.ixu is not None)
        ixe = jnp.concatenate([l.ind.ixa, l.ind.ixu])
        nu = l.ind.ixu.shape
        dt = lxt.t - lyp.t

        # linear step of coil-only circuit equation
        iet = jnp.linalg.solve(
            l.Mee + jnp.diag(dt * l.Re),
            jnp.concatenate([dt * lxt.Va, jnp.zeros(nu)]) + l.Mee @ iep)

        xnlv = xnlp.at[ixe - 1].set(iet / l.xscal[ixe-1])  # assign in xnl0
        xnl0s.append(xnlv)
      elif init_mode == 'extrap':
        dt = lxt.t - lyp.t
        xnl0s.append(xnlp + xnlpdot * dt)
      elif init_mode == 'previous':
        xnl0s.append(xnlp)
      elif init_mode == 'LX':
        raise NotImplementedError('LX initguess not implemented')
      else:
        raise ValueError(f'unknown initial guess option {init_mode}')

  # Choose the best initial condition
  if not xnl0s:
    raise ValueError('No initial guess generated.')
  elif len(xnl0s) == 1:
    xnl0 = xnl0s[0]
  else:
    xnl0_stack = jnp.stack(xnl0s)

    def eval_residual(x0):
      res, *_ = f_eval(x=x0)
      return jnp.sum(res**2)

    sqnormfi = jax.vmap(eval_residual)(xnl0_stack)
    ibest = jnp.argmin(sqnormfi)
    xnl0 = xnl0_stack[ibest]

  # Solve the equation F(xnl)=0
  fun = lambda x: f_eval(x=x)[0]
  def jac_fun(x):
    jx = f_eval(x=x)[2]
    assert jx is not None
    return jx.T
  jac = jac_fun
  solve_f = (  # TODO(pfau): jit this once we speed it up
      functools.partial(
          jax_root_finding.root_newton_raphson,
          fun=fun,
          custom_jac=jac,
          use_jax_custom_root=False,
          tol=1e-10,  # l.P.tolF,
          log_iterations=True,
          )
      )
  xnlt, metadata = solve_f(x0=xnl0)

  solverinfo = {
      'res': jnp.linalg.norm(metadata.residual),
      'niter': metadata.iterations,
      'isconverged': metadata.error == 0,
      'nfeval': metadata.iterations + 1,  # niter jac + 1 fun
      'prev_H': [],  # Not used in Newton
  }

  # TODO(pfau): Currently, the two branches return LY structs that may differ
  # in which attributes are none, so JAX fails. For now we assume it is always
  # converged, but in the future this should be fixed.
  # solution_good_enough = solverinfo['res'] < l.P.tolok
  #
  # xnlt, lyt = jax.lax.cond(
  #     jnp.logical_or(solverinfo['isconverged'],
  #                    jnp.logical_and(l.P.keepok, solution_good_enough)),
  #     lambda: (xnlt, f_eval(x=xnlt)[1]),
  #     lambda: (xnlp, lyp),
  # )
  lyt = f_eval(x=xnlt)[1]

  # other specific FGE outputs directly from LX
  lyt.Va = lxt.Va
  lyt.Ini = lxt.Ini
  lyt.IniD = lxt.IniD
  lyt.signeo = lxt.signeo
  lyt.Lp = lxt.Lp
  lyt.Rp = lxt.Rp
  lyt.t = lxt.t

  # Add convergence info
  lyt.res = solverinfo['res']
  lyt.mkryl = 0  # Not using Krylov
  lyt.nfeval = solverinfo['nfeval']
  lyt.niter = solverinfo['niter']
  lyt.isconverged = solverinfo['isconverged']

  return xnlt, lyt, solverinfo
