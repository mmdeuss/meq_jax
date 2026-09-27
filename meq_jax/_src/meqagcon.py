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

"""JAX implementation of the meqagcon.m file."""

import jax
import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import meqagconfun
from meq_jax._src import types


# pylint: disable=invalid-name
def meqagcon(
    L: types.StaticData,
    LX: types.InputData,
    agconc: list[types.ConcData],
    num_domains: int,
    F0: jt.Float[jt.Array, ''],
    F1: jt.Float[jt.Array, ''],
    rA: jt.Float[jt.Array, ''],
    dr2FA: jt.Float[jt.Array, ''],
    dz2FA: jt.Float[jt.Array, ''],
    drzFA: jt.Float[jt.Array, ''],
    ag: jt.Float[jt.Array, 'ng'],
    Fx: jt.Float[jt.Array, 'nry nzy'],
    Opy: jt.Int[jt.Array, 'nry nzy'],
    TpDg: jt.Float[jt.Array, 'nP+nT'],
    ITpDg: jt.Float[jt.Array, 'nP+nT'],
) -> tuple[jt.Float[jt.Array, 'nC'], jt.Float[jt.Array, 'nC']]:
  """Computes residuals for basis function coefficient constraints in MEQ.

  NOTE: This function returns the gradient of the residual with respect to
  the basis function coefficients as it's tricky to mimic the Matlab
  implementation otherwise. Eventually, this should be refactored.

  Args:
      L: StaticData object containing static data for the equilibrium.
      LX: InputData object containing input data for the equilibrium.
      agconc: List of AgConcData objects containing the constraint function
        names, domain ids, and indices.
      num_domains: Number of domains in the equilibrium.
      F0: Flux at the magnetic axis.
      F1: Flux at the plasma boundary.
      rA: Magnetic axis major radius.
      dr2FA: Second derivative of flux with respect to R at the magnetic axis.
      dz2FA: Second derivative of flux with respect to Z at the magnetic axis.
      drzFA: Mixed derivative of flux with respect to R, Z at the magnetic axis.
      ag: Basis function coefficients.
      Fx: Flux on the grid.
      Opy: Plasma domain mask.
      TpDg: Integral of TpDg over the plasma domain.
      ITpDg: Integral of ITpDg over the plasma domain.

  Returns:
      res: The residual of each of the constraints.
      dresdco: The gradient of the residual with respect to co.
  """
  del num_domains
  nC = len(agconc)
  ind = L.ind
  assert ind is not None
  P = L.P
  assert P is not None

  ag_confun_args = (
      L,
      LX,
      F0,
      F1,
      rA,
      dr2FA,
      dz2FA,
      drzFA,
      ag,
      Fx,
      Opy,
      TpDg,
      ITpDg,
  )

  assert LX.Ip is not None
  return jax.lax.cond(
      # TODO(adedieu): nB < L.nD second condition seems enough for fgeF_test.py
      # to pass. We may need to properly support inactive domains later.
      abs(LX.Ip) < P.Ipmin,
      # jnp.logical_or(abs(LX.Ip) < P.Ipmin, nB < L.nD),
      lambda: (
          ag[:nC] / L.xscal[ind.ixg[:nC] - 1],
          jnp.zeros((nC,)),
      ),  # zero indexing
      lambda: _meqagcon(L, LX, ag, agconc, ag_confun_args),
  )


def _meqagcon(
    L: types.StaticData,
    LX: types.InputData,
    ag: jt.Float[jt.Array, 'ng'],
    agconc: list[types.ConcData],
    ag_confun_args,
) -> tuple[jt.Float[jt.Array, 'nC'], jt.Float[jt.Array, 'nC']]:
  """Computes residuals for basis function coefficient constraints in MEQ."""
  results = []

  nB = jnp.sum(ag_confun_args[2] != ag_confun_args[3])

  for i, ag_data in enumerate(agconc):
    try:
      Co = getattr(LX, ag_data.lx_name)
    except AttributeError as e:
      raise ValueError(f'Unknown constraint: {ag_data.fun_name}') from e

    if Co is None or (ag_data.index is not None and not jnp.isscalar(Co)):
      Co_arr = jnp.zeros(0) if Co is None else jnp.atleast_1d(Co)
      if Co_arr.shape[0] == 0:
        # No target. MEQ allows that when the constraint's domain is inactive:
        # meqagcon.m skips the constraint function before it would read
        # LX.(f)(ii), and with no plasma no domain is active, so LX need not
        # carry the targets at all. Here the cond below is traced on both
        # sides, so the read still has to produce something; the branch that
        # uses it is not the one taken.
        Co = jnp.zeros(())
      elif Co_arr.shape[0] == 1:
        Co = Co_arr[0]
      else:
        Co = Co_arr[ag_data.index - 1]

    assert hasattr(L.ind, 'ixg')
    res, grad = jax.lax.cond(
        ag_data.domain_id > nB,
        lambda: (ag[i] / L.xscal[L.ind.ixg[i] - 1], 0.0),
        lambda: _meqagcon_apply(ag_data, Co, ag_confun_args),
    )
    results.append((res, grad))
  return jnp.array([r[0] for r in results]), jnp.array([r[1] for r in results])


def _meqagcon_apply(
    ag_data: types.ConcData,
    Co: float,
    ag_confun_args,
) -> tuple[jt.Float[jt.Array, 'nC'], jt.Float[jt.Array, 'nC']]:
  """Case where we have a single constraint."""
  fun = meqagconfun.meqagconfun_core(ag_data.fun_name)

  def partial_fun(co_val):
    return fun(*ag_confun_args, ag_data.index, co_val)

  res, grad = jax.value_and_grad(partial_fun)(Co)
  return res, grad


# pylint: enable=invalid-name
