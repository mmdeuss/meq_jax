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

"""MAJIC wrapper for fsgimex, ported from fsgimexm.m."""

import jax
import jax.numpy as jnp
import jaxtyping as jt


# pylint: disable=invalid-name
def fsgimex(
    M1q: jt.Float[jt.Array, 'n m'],
    M2q: jt.Float[jt.Array, 'n m'],
    rq: jt.Float[jt.Array, 'n m'],
    irq: jt.Float[jt.Array, 'n m'],
    rA: jt.Float[jt.Array, ''],
    FA: jt.Float[jt.Array, ''],
    FB: jt.Float[jt.Array, ''],
    BA: jt.Float[jt.Array, ''],
    lX: jt.Bool[jt.Array, ''],
    rX: jt.Float[jt.Array, ''],
    iTQ: jt.Float[jt.Array, 'n+1'],
    idoq: jt.Float[jt.Array, ''],
) -> tuple[
    jt.Float[jt.Array, 'n+1'],  # Q0Q
    jt.Float[jt.Array, 'n+1'],  # Q1Q
    jt.Float[jt.Array, 'n+1'],  # Q2Q
    jt.Float[jt.Array, 'n+1'],  # Q3Q
    jt.Float[jt.Array, 'n+1'],  # Q4Q
    jt.Float[jt.Array, 'n+1'],  # iqQ
    jt.Float[jt.Array, 'n+1'],  # ItQ
    jt.Float[jt.Array, 'n+1'],  # LpQ
    jt.Float[jt.Array, 'n+1'],  # rbQ
    jt.Float[jt.Array, 'n+1'],  # Q5Q
    jt.Float[jt.Array, 'n+1'],  # SlQ
]:
  """JAX implementation of fsgimexm."""
  mu0 = 4.0 * jnp.pi * 1e-7

  hipi = 1.0 / (2.0 * jnp.pi)
  irA = 1.0 / rA
  FAB = FA - FB
  sIp = jnp.sign(FAB)
  doq = 1.0 / idoq

  CA = hipi * BA
  C3 = 16.0 * FAB**2
  C5 = sIp * hipi * 4.0 * FAB
  Cq = 4.0 * idoq * FAB
  CI = 2.0 * FAB * doq / (jnp.pi * mu0)

  M3q = jnp.sqrt(M2q * M1q)

  s1q = 1.0 / jnp.sum(rq * M1q, axis=1)
  s2q = jnp.sum(irq * M1q, axis=1)
  s3q = jnp.sum(irq * M2q, axis=1)
  s5q = jnp.sum(M3q, axis=1)
  s6q = jnp.sum(rq * M3q, axis=1)

  Q0Q = jnp.concatenate([jnp.array([irA]), s1q * jnp.sum(M1q, axis=1)])
  Q1Q = jnp.concatenate([jnp.array([hipi * CA * irA]), hipi * Cq * s1q])
  Q2Q = jnp.concatenate([jnp.array([irA**2]), s1q * s2q])
  Q3Q = jnp.concatenate([jnp.array([0.0]), C3 * s1q * s3q])
  Q4Q = jnp.concatenate(
      [jnp.array([0.0]), C3 * s1q * jnp.sum(rq * M2q, axis=1)]
  )

  iqQ = jnp.concatenate([jnp.array([CA * rA]), Cq / s2q]) * iTQ

  ItQ = jnp.concatenate([jnp.array([0.0]), CI * s3q])
  LpQ = jnp.concatenate([jnp.array([0.0]), doq * s5q])
  rbQ = jnp.concatenate([jnp.array([rA]), s6q / s5q])
  Q5Q = jnp.concatenate([jnp.array([0.0]), C5 * s1q * s6q])
  SlQ = jnp.concatenate([jnp.array([0.0]), 2.0 * jnp.pi * doq * s6q])

  def diverted_case(Q0Q, Q1Q, Q2Q, Q3Q, Q4Q, iqQ, Q5Q):
    irX = 1.0 / rX
    Q0Q = Q0Q.at[-1].set(irX)
    Q1Q = Q1Q.at[-1].set(0.0)
    Q2Q = Q2Q.at[-1].set(irX**2)
    Q3Q = Q3Q.at[-1].set(0.0)
    Q4Q = Q4Q.at[-1].set(0.0)
    iqQ = iqQ.at[-1].set(0.0)
    Q5Q = Q5Q.at[-1].set(0.0)
    return Q0Q, Q1Q, Q2Q, Q3Q, Q4Q, iqQ, Q5Q

  def non_diverted_case(Q0Q, Q1Q, Q2Q, Q3Q, Q4Q, iqQ, Q5Q):
    return Q0Q, Q1Q, Q2Q, Q3Q, Q4Q, iqQ, Q5Q

  Q0Q, Q1Q, Q2Q, Q3Q, Q4Q, iqQ, Q5Q = jax.lax.cond(
      lX, diverted_case, non_diverted_case, Q0Q, Q1Q, Q2Q, Q3Q, Q4Q, iqQ, Q5Q
  )

  return Q0Q, Q1Q, Q2Q, Q3Q, Q4Q, iqQ, ItQ, LpQ, rbQ, Q5Q, SlQ
# pylint: enable=invalid-name
