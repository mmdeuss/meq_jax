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

"""JAX implementation of the meqprof.m file."""

import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import bfct_router
from meq_jax._src import types


# pylint: disable=invalid-name


def meqprof(
    ag: jt.Float[jt.Array, 'ng'],
    FN: jt.Float[jt.Array, 'nQ'],
    F0: jt.Float[jt.Array, ''],
    F1: jt.Float[jt.Array, ''],
    rBt: jt.Float[jt.Array, ''],
    idsx: jt.Float[jt.Array, ''],
    Bfp: types.BfpData,
    smalldia: bool = False,
) -> tuple[
    jt.Float[jt.Array, 'nQ'],  # PpQ
    jt.Float[jt.Array, 'nQ'],  # TTpQ
    jt.Float[jt.Array, 'nQ'],  # PQ
    jt.Float[jt.Array, 'nQ'],  # TQ
    jt.Float[jt.Array, 'nQ'],  # iTQ
    jt.Float[jt.Array, 'nQ ng'],  # PpQg
    jt.Float[jt.Array, 'nQ ng'],  # TTpQg
]:
  """Computes profiles from basis functions and fitting parameters.

  Args:
      ag: basis function coefficients
      FN: Normalized flux on Q grid = L.pQ^2;
      F0: Axis flux
      F1: boundary flux
      rBt: Vacuum R*Bt
      idsx: 1/(drx*dzx) = 1/(grid element surface)
      Bfp: Bfp data
      nP: Number of P' basis functions.
      nT: Number of TT' basis functions.
      smalldia: use small diamagnetism approximation for TQ computation

  Returns:
      PpQ: pprime
      TTpQ: TTprime
      PQ: Their primitives, assuming P(1)=0
      TQ: Their primitives, assuming T(1)=rBt
      iTQ: 1/TQ
      PpQg: Contributions of each basis function to PpQ
      TTpQg: Contributions of each basis function to TTpQ
  """

  # Computes profiles from basis functions
  # ng = nP + nT
  gQg, IgQg = bfct_router.bfct2(FN=FN, F0=F0, F1=F1, Bfp=Bfp)
  # Same shapes for doublet vs not
  if gQg.ndim == 2:
    assert IgQg.ndim == 2
    gQg = gQg[:, None, :]
    IgQg = IgQg[:, None, :]
  gQg = jnp.transpose(gQg, (2, 1, 0))
  IgQg = jnp.transpose(IgQg, (2, 1, 0))

  aPpg, aTTpg, aPg, ahqTg = bfct_router.bfct3(
      ag=ag, F0=F0, F1=F1, ids=idsx, Bfp=Bfp
  )
  # (ng,), (ng,), (ng,), (ng,)

  assert gQg.shape[-1] == aPpg.shape[0]
  PpQg = gQg * aPpg
  TTpQg = gQg * aTTpg

  PpQ = jnp.sum(PpQg, axis=-1)  # (nQ,nD)
  TTpQ = jnp.sum(TTpQg, axis=-1)  # (nQ,nD)

  PQ = jnp.sum(IgQg * aPg, axis=-1)  # (nQ,nD)
  hqTQ = jnp.sum(IgQg * ahqTg, axis=-1)  # (nQ,nD)

  if smalldia:
    TQ = rBt + hqTQ / rBt  # small diamagnetic approximation
    iTQ = (1 -  hqTQ / (rBt * rBt)) / rBt
  else:
    TQ = jnp.sign(rBt) * jnp.sqrt(2 * hqTQ + rBt * rBt)
    iTQ = 1.0 / TQ

  return PpQ, TTpQ, PQ, TQ, iTQ, PpQg, TTpQg


# pylint: enable=invalid-name
