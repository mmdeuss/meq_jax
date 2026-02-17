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

"""find_contours jax implementation.

This overrides the Matlab implementation in
  third_party/meq/rtciwall.m
"""

import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import bint
from meq_jax._src import find_contours
from meq_jax._src import meq_fw
from meq_jax._src import types


# pylint: disable=invalid-name
def rtciwall(
    L: types.StaticData, LY: types.OutputData
) -> tuple[jt.Float[jt.Array, 'nW nFW'], jt.Float[jt.Array, 'nFW']]:
  """Computes wall gaps from equilibrium in LY.

  Computes wall gaps of the plasma. Internally utilizes find_contours to
  find LCFS.

  Arguments:
    L: standard L structure
    LY: standard LY structure containing converged GS-equilibrium

  Returns:
    aW: Wall gaps
    FW: Flux values at line-of-sight-plasma intersections
  """
  assert L.P is not None
  assert L.G is not None
  assert L.P.nFW is not None

  aW = LY.aW
  if LY.aW is None:
    aW = jnp.zeros((L.P.nFW, L.G.nW))
  else:
    if aW.ndim == 1:
      aW = aW[None]
    assert aW.shape == (L.P.nFW, L.G.nW)

  assert LY.nB is not None
  FW = meq_fw.meq_fw(
      FB=LY.FB[LY.nB - 1],
      FX=LY.FX,
      lX=LY.lX[LY.nB - 1],
      nX=LY.nX,
      nFW=L.P.nFW,
      dimw=L.dimw,  # NOTE: use a static upper bound
  )

  # The Matlab code has the lines below, but we never call it and cannot jit it.
  # if LY.aW is None:
  #   FW = jnp.broadcast_to(L.FN, (L.P.nFW, L.FN.shape[1]))

  if L.nW:
    crW = L.crW
    czW = L.czW
    rO = L.G.rW
    zO = L.G.zW
    # Flux values at wall origin points
    FoW = bint.bintmex(Fx=LY.Fx, k=(L.kxW - 1).astype(jnp.int32), c=L.cWx)
    _, _, aW = find_contours.find_contours(
        L=L,
        LY=LY,
        F=FW,
        aq=aW,
        rO=rO,
        zO=zO,
        crq=crW,
        czq=czW,
        Fo=FoW,
        Opo=jnp.array([0]),
    )
    # clamping to min/max
    aW = jnp.clip(aW, 0, None)
    aW = jnp.minimum(aW, L.G.aW)

  return aW, FW


# pylint: enable=invalid-name
