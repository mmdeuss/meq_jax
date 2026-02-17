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

"""JAX implementation of the cizrmexm.m file."""

import jax
import jax.numpy as jnp
import jaxtyping as jt


# pylint: disable=invalid-name
def cizr(
    Fx: jt.Float[jt.Array, 'b1+2 b2+2'],
    Opy: jt.Int[jt.Array, 'b1 b2'],
    Vy: jt.Float[jt.Array, 'b1 b2 nV'],
    dsx: float,
    FNQ: jt.Float[jt.Array, 'nQ'],
    F0: jt.Float[jt.Array, 'nD'],
    F1: jt.Float[jt.Array, 'nD'],
) -> jt.Float[jt.Array, 'nV nD nQ']:
  """JAX implementation of cizrmex.

  NOTE: Vy is flattened in the first two dimensions in the Matlab version.
  Due to the way Matlab represents matrices vs JAX, this means care needs to be
  taken, as the order of elements in the flattened axis will be different
  depending on whether it is flattened in Matlab or JAX.

  Args:
    Fx: The boundary conditions of the system.
    Opy: The active domains.
    Vy: Values of custom grid.
    dsx: RZ grid cell area.
    FNQ: Normalized flux on Q-grid.
    F0: Lower flux limit.
    F1: Upper flux limit.

  Returns:
    The IVQD tensor.
  """
  ivqd = jax.vmap(
      jax.vmap(
          lambda f, o, v: _cizr(f, o, v, FNQ, F0, F1), in_axes=(0, 0, 0)
      ),
      in_axes=(0, 0, 0),
  )(Fx[1:-1, 1:-1], Opy, Vy)
  return dsx * jnp.sum(ivqd, axis=(0, 1))


def _cizr(
    Fx: jt.Float[jt.Array, ''],
    Opy: jt.Int[jt.Array, ''],
    Vy: jt.Float[jt.Array, 'nV'],
    FNQ: jt.Float[jt.Array, 'nQ'],
    F0: jt.Float[jt.Array, 'nD'],
    F1: jt.Float[jt.Array, 'nD'],
) -> jt.Float[jt.Array, 'nV nD nQ']:
  """Helper function for cizr to be used with jax.vmap."""

  # Create masks for active domains
  domain_masks = Opy == (jnp.arange(F0.shape[0]) + 1)

  # Compute FN for each domain
  FN = jnp.nan_to_num((Fx - F0) / (F1 - F0), posinf=0, neginf=0)
  FN = jnp.where(domain_masks, FN, -jnp.inf)  # Inactive domains get -inf

  # Find iQ for each point and domain
  iQ = jnp.sum(FN[:, None] > FNQ[None, :], axis=-1) + 1

  (nQ,) = FNQ.shape
  IVQD = jnp.einsum(
      'd,dq,v->vdq',
      (iQ <= nQ) & domain_masks,  # mask for valid iQ values
      jax.nn.one_hot(iQ - 1, nQ, axis=-1),
      Vy,
  )

  return jnp.cumsum(IVQD, axis=-1)


# pylint: enable=invalid-name
