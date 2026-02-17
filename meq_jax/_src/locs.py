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

"""locSmex JAX implementation.

This overrides the implementation in
  third_party/meq/mexc/locSmex.c
whose pure Matlab implementation is in
  third_party/meq/mexm/locSmexm.m
"""

import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import locq

# pylint: disable=invalid-name


def locS(
    aQ: jt.Float[jt.Array, 'nQ nO'],
    qQ: jt.Float[jt.Array, 'nQ'],
    qS: jt.Float[jt.Array, 'nR'],
    NF: float = jnp.nan,
) -> jt.Float[jt.Array, 'naR nR']:
  """Location of q surfaces.

  LOCSMEX  Location of q surfaces
  [AS] = LOCSMEX(AQ,QQ,QS,NF) returns the location of the flux surfaces where
  QR=QQ(AQ==AR)

  Attributes:
    AQ: x data (may be a matrix [nQ x m]
    QQ: y data
    QS: Requested y values
    NF: (optional) AR = NF signals the q value was not found

  Returns:
    AS: interpolated x data
  """
  aS = locq.locQ(aQ.T, qQ, qS, naR=1, NF=NF).squeeze()
  assert aS.ndim == 2, 'locS returned aS with ndim != 2'
  return aS.T


# pylint: enable=invalid-name
