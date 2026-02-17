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

"""fl4p majic implementation."""

import jax
import jax.numpy as jnp
import jaxtyping as jt

from meq_jax._src import bint


# pylint: disable=invalid-name
def fl4pmex(
    Fx: jt.Float[jt.Array, 'nr nz'],
    kl: jt.Integer[jt.Array, 'nl'],
    cl: jt.Float[jt.Array, 'nl 4'],
    FN: jt.Float[jt.Array, ''],
) -> tuple[
    jt.Float[jt.Array, 'nl'],  # FL.
    jt.Float[jt.Array, 'nl'],  # DRFL.
    jt.Float[jt.Array, 'nl'],  # DZFL.
]:
  """Four point interpolation for the flux on the limiter.

  See meq/fl4pmex.m for more details.
  """
  Fl = bint.bintmex(Fx, kl, cl)
  wrapped = jnp.insert(Fl, 0, Fl[-1])
  dFl = jnp.diff(wrapped)
  rolled = jnp.roll(dFl, -1)
  mask = jnp.logical_or(dFl * rolled > 0, (dFl - rolled) * FN > 0)
  Fl = jnp.where(mask, FN, Fl)

  flat = Fx.flatten()
  nz = Fx.shape[1]
  drFl = (flat[kl + nz] - flat[kl]) * (cl[:, 1] + cl[:, 0]) + (
      flat[kl + nz + 1] - flat[kl + 1]
  ) * (cl[:, 3] + cl[:, 2])
  dzFl = (flat[kl + 1] - flat[kl]) * (cl[:, 3] + cl[:, 0]) + (
      flat[kl + nz + 1] - flat[kl + nz]
  ) * (cl[:, 2] + cl[:, 1])

  drFl = jnp.where(mask, 0, drFl)
  dzFl = jnp.where(mask, 0, dzFl)

  drFl = jax.lax.cond(FN == 0, lambda: jnp.zeros_like(drFl), lambda: drFl)
  dzFl = jax.lax.cond(FN == 0, lambda: jnp.zeros_like(dzFl), lambda: dzFl)
  return Fl, drFl, dzFl
