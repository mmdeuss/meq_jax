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

"""fbnd majic implementation."""

import jax
import jax.numpy as jnp
import jaxtyping as jt


# pylint: disable=invalid-name
def fbnd(
    Fl: jt.Float[jt.Array, 'nl'],
    rl: jt.Float[jt.Array, 'nl'],
    zl: jt.Float[jt.Array, 'nl'],
    FX: jt.Float[jt.Array, 'nx'],
    rX: jt.Float[jt.Array, 'nx'],
    zX: jt.Float[jt.Array, 'nx'],
    FN: jt.Float[jt.Array, ''],
) -> tuple[
    jt.Float[jt.Array, ''],  # FB
    jt.Float[jt.Array, ''],  # rB
    jt.Float[jt.Array, ''],  # zB
    jt.Bool[jt.Array, ''],  # lB
    jt.Bool[jt.Array, ''],  # lX
    jt.Integer[jt.Array, ''],  # kB
]:
  """Finds the flux and the limiting point of the LCFS."""

  def find_min(fn_scalar, f_arr):
    aug_arr = jnp.concatenate([jnp.array([fn_scalar]), f_arr])
    val = jnp.min(aug_arr)
    idx = jnp.argmin(aug_arr)
    return val, idx

  def find_max(fn_scalar, f_arr):
    aug_arr = jnp.concatenate([jnp.array([fn_scalar]), f_arr])
    val = jnp.max(aug_arr)
    idx = jnp.argmax(aug_arr)
    return val, idx

  def neg_case():
    (fbx, kx) = find_min(FN, FX)
    (fbl, kl) = find_min(FN, Fl)
    lx = fbx < fbl
    return fbx, kx, fbl, kl, lx

  def pos_case():
    (fbx, kx) = find_max(FN, FX)
    (fbl, kl) = find_max(FN, Fl)
    lx = fbx > fbl
    return fbx, kx, fbl, kl, lx

  fbx, kx, fbl, kl, lx = jax.lax.cond(FN > 0, neg_case, pos_case)

  # kx and kl are 0-based indices into augmented arrays.
  # k=0 means FN was selected.
  kx_in_fx = kx > 0
  kl_in_fl = kl > 0

  # Workaround to avoid errors in the case that nL or nX = 0
  rX_ = jnp.concatenate([jnp.array([0.0]), rX])
  zX_ = jnp.concatenate([jnp.array([0.0]), zX])
  rl_ = jnp.concatenate([jnp.array([0.0]), rl])
  zl_ = jnp.concatenate([jnp.array([0.0]), zl])

  def kx_is_limiter_fn():
    # kx is from the outer scope. Indexing is safe due to the condition.
    return fbx, rX_[kx], zX_[kx], jnp.bool_(True), kx

  def kl_is_limiter_fn():
    # kl is from the outer scope.
    def kl_is_limiter_true_fn():
      return fbl, rl_[kl], zl_[kl], jnp.bool_(True), kl

    def kl_is_limiter_false_fn():
      return FN, 0.0, 0.0, jnp.bool_(False), 0

    return jax.lax.cond(kl_in_fl, kl_is_limiter_true_fn, kl_is_limiter_false_fn)

  fb, rb, zb, lb, kb = jax.lax.cond(
      kx_in_fx & lx, kx_is_limiter_fn, kl_is_limiter_fn
  )

  return fb, rb, zb, lb, lx, kb.astype(jnp.int32)
