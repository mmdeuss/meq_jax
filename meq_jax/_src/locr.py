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

"""locRmex JAX implementation.

This overrides the implementation in
  third_party/meq/mexc/locRmex.c
whose pure Matlab implementation is in
  third_party/meq/mexm/locRmexm.m

Note: locRmex.c and locRmexm.m are out-of-sync.
"""

import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import locq

# pylint: disable=invalid-name


def locR(
    aQ: jt.Float[jt.Array, 'nQ'],
    qQ: jt.Float[jt.Array, 'nQ'],
    qR: jt.Float[jt.Array, 'nR'],
    naR: int = 1,
    NF: float = jnp.nan,
) -> jt.Float[jt.Array, 'naR nR']:
  """Location of q surfaces.

  [AR] = LOCRMEX(AQ,QQ,QR,NAR,NF) returns the location of the flux surfaces
  where QR=QQ(AQ==AR)

  Arguments:
    AQ: x data
    QQ: y data
    QR: Requested y values
    NAR: (optional) Number of maximum R values to be found (per QR) for
      nonmonotonic profiles. (default=1: outermost value found)
    NF: (optional) AR = NF signals the q value was not found

  Returns:
    AR: interpolated x data
  """
  assert aQ.ndim == 1, 'aQ must be 1D'
  assert qQ.ndim == 1, 'qQ must be 1D'
  aR = locq.locQ(aQ[None], qQ, qR, naR, NF).squeeze()
  if aR.ndim == 1:
    return aR[None]
  return aR.T


# pylint: enable=invalid-name
