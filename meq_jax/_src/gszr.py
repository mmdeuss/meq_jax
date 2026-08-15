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

"""JAX implementations of the GSZR MATLAB function.

`gszrjax` is a pure JAX implementation that is faster, but cannot be used with
majic.
`gszrmex` is the version that is compatible with majic.

"""

import functools

import jax
import jax.numpy as jnp
import jaxtyping as jt


@jax.jit
def gszrjax(
    boundary_conditions: jt.Float[jt.Array, '2*nz2+2*nr2+4'],
    filament_currents: jt.Float[jt.Array, 'nr2 nz2'],
    cx: jt.Float[jt.Array, 'nr2'],
    cq: jt.Float[jt.Array, 'nz2 nr2'],
    cr: jt.Float[jt.Array, 'nz2 nr2'],
    cs: jt.Float[jt.Array, 'nz2 nr2'],
    ci: jt.Float[jt.Array, ''],
    co: jt.Float[jt.Array, ''],
    dz: jt.Float[jt.Array, ''],
) -> jt.Float[jt.Array, 'nr2+2 nz2+2']:
  """JAX implementation of the GSZRMEX MATLAB function.

  Solves the Poisson equation:
    Delta'(F(z,r) = -2*pi*mu0*r*(I/dr/dz).
  where F is the flux [Wb]

  NOTE: This function is not compatible with majic, it does not work with
  symbolic shapes as it uses dynamic slices.

  Args:
    boundary_conditions: Flat boundary conditions
    filament_currents: Assumed to be 0 outside of grid [nr2, nz2]
    cx: Coefficients
    cq: Coefficient vector.
    cr: Coefficient vector for the backward loop.
    cs: Coefficient vector for the forward loop.
    ci: Coefficient for the inner boundaries.
    co: Coefficient for the outer boundaries.
    dz: Grid spacing

  Returns:
    [nr2 + 2, nz2 + 2] Solution to the Poisson equation.
  """
  fx = _initialise_grid(boundary_conditions, filament_currents, cx, ci, co)

  nr, nz = fx.shape
  nz1 = nz - 1

  l = nz1 // 2 - 1
  j = jnp.arange(2, nz1, 2)
  i = slice(1, -1)
  jp = j - 1

  vmap_gsp = jax.vmap(_gsp, in_axes=(1, None, None, None), out_axes=1)

  def for_body_fun(l, val):
    p, lo, cq, cr, cs = val
    p_new = jax.lax.cond(
        l % (2 * lo) == (lo - 1) % (2 * lo),
        lambda p: vmap_gsp(p, cq[l, :], cr[l, :], cs[l, :]),
        lambda p: p,
        p)
    return p_new, lo, cq, cr, cs

  # Multigrid V cycle - GSP loop
  p_init = vmap_gsp(2.0 * fx[i, j], cq[l, :], cr[l, :], cs[l, :])
  fx = fx.at[i, j].set(fx[i, j + 1] + fx[i, jp] + p_init)

  p = jnp.zeros((nr - 2, nz - 2))
  p = p.at[:, jp].set(p_init)

  jh = 1
  lo = nz1 // 4
  while lo > 1:
    jd_offset = 2 * jh
    jo_step = 4 * jh
    ji_offset = 3 * jh

    j = jnp.arange(jo_step, nz1, jo_step)

    fx_update_vals = (
        fx[i, j]
        - fx[i, j + jh]
        - fx[i, j - jh]
        + fx[i, j + jd_offset]
        + fx[i, j - jd_offset]
    )

    p_loop = (
        fx[i, j] - fx[i, j + ji_offset] - fx[i, j - ji_offset] + fx_update_vals
    )

    p_loop, *_ = jax.lax.fori_loop(
        lo - 1, nz1, for_body_fun, (p_loop, lo, cq, cr, cs))

    p = p.at[:, j - 1].set(p_loop)
    fx = fx.at[i, j].set(fx_update_vals + p_loop)  # gsu

    jh = jh * 2
    lo = lo // 2

  # Multigrid V cycle - Up loop
  while jh >= 1:
    jo_step = 2 * jh
    j = jnp.arange(jo_step, nz1, 2 * jo_step)
    p_loop = 2 * fx[i, j] + fx[i, j + jo_step] + fx[i, j - jo_step]

    p_loop, *_ = jax.lax.fori_loop(
        lo - 1, nz1, for_body_fun, (p_loop, lo, cq, cr, cs))

    p = p.at[:, j - 1].set(p_loop)
    fx = fx.at[i, j].set(fx[i, j] - fx[i, j + jh] - fx[i, j - jh] + p_loop)
    jh = jh // 2
    lo = lo * 2

  # Multigrid V cycle - Final step
  l = nz1 // 2 - 1
  j = jnp.arange(1, nz1, 2)
  p_final = 2.0 * fx[i, j] + fx[i, j + 1] + fx[i, j - 1]
  fx = fx.at[i, j].set(vmap_gsp(p_final, cq[l, :], cr[l, :], cs[l, :]))

  # Shift solution using dz
  # This takes rows [1, 2, ..., N-1] and appends row N-1 again.
  fx_forward = jnp.concatenate([fx[:, 1:], fx[:, -1:]], axis=1)
  # This takes row 0 and prepends it to rows [0, 1, ..., N-2].
  fx_backward = jnp.concatenate([fx[:, :1], fx[:, :-1]], axis=1)

  # Create the scaling vector: [dz; 0.5*dz; ...; 0.5*dz; dz]
  scaling_vector = jnp.pad(
      jnp.full((nz - 2,), 0.5 * dz),
      (1, 1),
      constant_values=dz,
  )
  return fx + (fx_forward - fx_backward) * scaling_vector[None, :]


def gszrmex(
    boundary_conditions: jt.Float[jt.Array, '2*nz2+2*nr2+4'],
    filament_currents: jt.Float[jt.Array, 'nr2 nz2'],
    cx: jt.Float[jt.Array, 'nr2'],
    cq: jt.Float[jt.Array, 'nz2 nr2'],
    cr: jt.Float[jt.Array, 'nz2 nr2'],
    cs: jt.Float[jt.Array, 'nz2 nr2'],
    ci: jt.Float[jt.Array, ''],
    co: jt.Float[jt.Array, ''],
    dz: jt.Float[jt.Array, ''],
) -> jt.Float[jt.Array, 'nr2+2 nz2+2']:
  """JAX implementation of the GSZRMEX MATLAB function.

  Solves the Poisson equation:
    Delta'(F(z,r) = -2*pi*mu0*r*(I/dr/dz).
  where F is the flux [Wb]
  
  NOTE: This function is compatible with majic as it works with symbolic shapes,
  but is slower than `gszrjax`. It uses jax for loops instead of dynamic slices.

  Args:
    boundary_conditions: [2 nz2 + 2 nr2 + 4] Flat boundary conditions
    filament_currents: Assumed to be 0 outside of grid [nr2, nz2]
    cx: [nr2] Coefficients
    cq: Coefficient vector. [nz2, nr2]
    cr: Coefficient vector for the backward loop. [nz2, nr2]
    cs: Coefficient vector for the forward loop. [nz2, nr2]
    ci: Coefficient for the inner boundaries. []
    co: Coefficient for the outer boundaries. []
    dz: Grid spacing []
  Returns:
    [nr2 + 2, nz2 + 2] Solution to the Poisson equation.
  """
  fx = _initialise_grid(boundary_conditions, filament_currents, cx, ci, co)

  nr, nz = fx.shape
  nz1 = nz - 1

  l = nz1 // 2 - 1
  j = jnp.arange(2, nz1, 2)
  i = slice(1, -1)
  jp = j - 1

  vmap_gsp = jax.vmap(_gsp, in_axes=(1, None, None, None), out_axes=1)

  # Multigrid V cycle - GSP loop
  p_init = vmap_gsp(2.0 * fx[i, j], cq[l, :], cr[l, :], cs[l, :])
  fx = fx.at[i, j].set(fx[i, j + 1] + fx[i, jp] + p_init)

  p = jnp.zeros((nr - 2, nz - 2))
  p = p.at[:, jp].set(p_init)

  # Multigrid V cycle needs to be run log2nz1 - 2 times.
  # NOTE: We assume nz1 is a power of 2.
  log2nz1 = jax.lax.while_loop(
      lambda carry: carry[0] > 0,
      lambda carry: (carry[0] >> 1, carry[1] + 1),
      (nz1, -1),
  )[1]

  # Start of Multigrid V cycle - Down loop
  fx, p = jax.lax.fori_loop(
      0,
      log2nz1 - 2,
      lambda loop_idx, carry: _down_loop_body(loop_idx, *carry, cq, cr, cs),
      (fx, p),
  )

  # Multigrid V cycle - Up loop
  fx, _ = jax.lax.fori_loop(
      0,
      log2nz1 - 1,
      lambda loop_idx, carry: _up_loop_body(loop_idx, *carry, cq, cr, cs),
      (fx, p),
  )

  # Multigrid V cycle - Final step
  j = jnp.arange(1, nz1, 2)
  p_final = 2.0 * fx[i, j] + fx[i, j + 1] + fx[i, j - 1]
  fx = fx.at[i, j].set(vmap_gsp(p_final, cq[l, :], cr[l, :], cs[l, :]))

  # Shift solution using dz
  # This takes rows [1, 2, ..., N-1] and appends row N-1 again.
  fx_forward = jnp.concatenate([fx[:, 1:], fx[:, -1:]], axis=1)
  # This takes row 0 and prepends it to rows [0, 1, ..., N-2].
  fx_backward = jnp.concatenate([fx[:, :1], fx[:, :-1]], axis=1)

  # Create the scaling vector: [dz; 0.5*dz; ...; 0.5*dz; dz]
  scaling_vector = jnp.pad(
      jnp.full((nz - 2,), 0.5 * dz),
      (1, 1),
      constant_values=dz,
  )[None, :]
  return fx + (fx_forward - fx_backward) * scaling_vector


def gszr_operator(
    cx: jt.Float[jt.Array, 'nr2'],
    cq: jt.Float[jt.Array, 'nz2 nr2'],
    cr: jt.Float[jt.Array, 'nz2 nr2'],
    cs: jt.Float[jt.Array, 'nz2 nr2'],
    ci: jt.Float[jt.Array, ''],
    co: jt.Float[jt.Array, ''],
    nr2: int,
    nz2: int,
) -> tuple[
    jt.Float[jt.Array, '2*nz2+2*nr2+4 (nr2+2)*(nz2+2)'],
    jt.Float[jt.Array, 'nr2*nz2 (nr2+2)*(nz2+2)'],
]:
  """Materializes the (linear) gszrjax solve as a pair of dense operators.

  gszrjax (with dz=0) is a linear map from (boundary_conditions,
  filament_currents) to the flux map. On accelerators the cyclic-reduction
  algorithm in gszrjax executes hundreds of tiny sequential kernels, while
  applying the materialized operator is a single matmul. The operators are
  built by pushing basis vectors through gszrjax, so they represent exactly
  the same linear map (applying them differs from gszrjax only by
  floating-point summation order).

  Returns:
    bc_op: operator applied as bc @ bc_op.
    iy_op: operator applied as iy.flatten() @ iy_op.
  """
  nb = 2 * (nz2 + 2 + nr2)
  ny = nr2 * nz2

  def solve(bc, iy_flat):
    return gszrjax(bc, iy_flat.reshape(nr2, nz2), cx, cq, cr, cs, ci, co, 0.0)

  bc_op = jax.vmap(lambda b: solve(b, jnp.zeros(ny)))(jnp.eye(nb))
  iy_op = jax.vmap(lambda y: solve(jnp.zeros(nb), y))(jnp.eye(ny))
  return bc_op.reshape(nb, -1), iy_op.reshape(ny, -1)


def apply_gszr_operator(
    boundary_conditions: jt.Float[jt.Array, '2*nz2+2*nr2+4'],
    filament_currents: jt.Float[jt.Array, 'nr2 nz2'],
    bc_op: jt.Float[jt.Array, '2*nz2+2*nr2+4 nx'],
    iy_op: jt.Float[jt.Array, 'nr2*nz2 nx'],
    dz: jt.Float[jt.Array, ''],
) -> jt.Float[jt.Array, 'nr2+2 nz2+2']:
  """Applies the operators from `gszr_operator`, including the dz shift."""
  nr2, nz2 = filament_currents.shape
  fx = boundary_conditions @ bc_op + filament_currents.reshape(-1) @ iy_op
  fx = fx.reshape(nr2 + 2, nz2 + 2)
  return _dz_shift(fx, dz)


def _dz_shift(
    fx: jt.Float[jt.Array, 'nr nz'], dz: jt.Float[jt.Array, '']
) -> jt.Float[jt.Array, 'nr nz']:
  """Shift solution using dz (identical to the tail of gszrjax/gszrmex)."""
  nz = fx.shape[1]
  fx_forward = jnp.concatenate([fx[:, 1:], fx[:, -1:]], axis=1)
  fx_backward = jnp.concatenate([fx[:, :1], fx[:, :-1]], axis=1)
  scaling_vector = jnp.pad(
      jnp.full((nz - 2,), 0.5 * dz),
      (1, 1),
      constant_values=dz,
  )
  return fx + (fx_forward - fx_backward) * scaling_vector[None, :]


def _initialise_grid(
    boundary_conditions: jt.Float[jt.Array, '2*nz2+2*nr2+4'],
    filament_currents: jt.Float[jt.Array, 'nr2 nz2'],
    cx: jt.Float[jt.Array, 'nr2'],
    ci: jt.Float[jt.Array, ''],
    co: jt.Float[jt.Array, ''],
) -> jnp.ndarray:
  """Create the initial grid.

  Args:
    boundary_conditions: [2 nz2 + 2 nr2 + 4] Flat boundary conditions
    filament_currents: [nr2, nz2] Assumed to be 0 outside of grid
    cx: [nr2] Coefficients
    ci: Coefficient for the inner boundaries. []
    co: Coefficient for the outer boundaries. []

  Returns:
    [nr, nz]
  """
  # Get input dimensions from the 2D Iy matrix
  nr2, nz2 = filament_currents.shape
  nz = nz2 + 2
  nb = 2 * (nz + nr2)

  # ([nz], [2 nr2], [nz])
  top, l_r, bottom = jnp.split(boundary_conditions, (nz, nb - nz), axis=0)
  left, right = l_r[::2, None], l_r[1::2, None]

  fx = jnp.block([
      [top[None, :]],
      [left, cx[:, None] * filament_currents, right],
      [bottom[None, :]],
  ])

  # Inner boundaries with additions
  j = slice(1, -1)
  fx = fx.at[1, j].add(ci * top[j])
  fx = fx.at[-2, j].add(co * bottom[j])

  return fx


def _gsp(
    p: jt.Float[jt.Array, 'nr2'],
    cq: jt.Float[jt.Array, 'nr2'],
    cr: jt.Float[jt.Array, 'nr2'],
    cs: jt.Float[jt.Array, 'nr2'],
) -> jt.Float[jt.Array, 'nr2']:
  """JAX translation of the gsp MATLAB function.

  Unlike the MATLAB function, this function is not batched. We can vmap the gsp
  function in the caller to mimic the original behaviour.

  Args:
    p: The main vector to be updated.
    cq: Coefficient vector.
    cr: Coefficient vector for the backward loop.
    cs: Coefficient vector for the forward loop.

  Returns:
    The p vector
  """

  def backward_scan_body(prev_row, carry):
    new_row = carry[0] + carry[1] * prev_row
    return new_row, new_row

  _, computed_rows_rev = jax.lax.scan(
      backward_scan_body,
      p[-1],
      (p[:-1], cr[:-1]),
      reverse=True,
  )

  p = jnp.concatenate([computed_rows_rev, p[-1:]], axis=0) * cq

  def forward_scan_body(prev_row, carry):
    new_row = carry[0] + carry[1] * prev_row
    return new_row, new_row

  _, computed_rows = jax.lax.scan(forward_scan_body, p[0], (p[1:], cs[1:]))

  return jnp.concatenate([p[:1], computed_rows], axis=0)


def _down_loop_body(idx, fx, p, cq, cr, cs):
  """JAX implementation of the 'down_loop' multigrid restriction step."""
  # NOTE: Unlike the MATLAB implementation, This function uses a for loop for
  # `j`, since JAX is not able to statically determine the values of `jh` and
  # `lo`.
  nz = fx.shape[1]
  jh = 1 << idx  # 2^idx
  lo = (nz - 1) >> (idx + 2)  # nz1 / 2^(idx + 2)
  jd_offset = 2 * jh
  jo_step = 4 * jh
  ji_offset = 3 * jh

  i = slice(1, -1)
  # Calculate the number of coarse rows to iterate over.
  num_iterations = (nz - 2 - jo_step) // jo_step + 1
  num_l_iterations = (nz - 2 - lo) // (2 * lo) + 1

  # Define the body of the loop for a single coarse row
  def body_fun(loop_idx, carry):
    fx, p = carry

    # Calculate the current coarse-grid row index based on the loop index
    j = jo_step * loop_idx

    fx_update_vals = (
        fx[i, j]
        - fx[i, j + jh]
        - fx[i, j - jh]
        + fx[i, j + jd_offset]
        + fx[i, j - jd_offset]
    )

    p_loop = (
        fx[i, j] - fx[i, j + ji_offset] - fx[i, j - ji_offset] + fx_update_vals
    )

    def body_l_fun(loop_idx, p_carry):
      l = lo * (loop_idx * 2 + 1) - 1
      return _gsp(p_carry, cq[l, :], cr[l, :], cs[l, :])

    p_loop = jax.lax.fori_loop(0, num_l_iterations, body_l_fun, p_loop)
    return fx.at[i, j].set(fx_update_vals + p_loop), p.at[:, j - 1].set(p_loop)

  return jax.lax.cond(
      num_iterations > 0,
      functools.partial(jax.lax.fori_loop, 1, num_iterations + 1, body_fun),
      _identity,
      (fx, p),
  )


def _identity(x):
  return x


def _up_loop_body(idx, fx, p, cq, cr, cs):
  """JAX implementation of the 'up_loop' multigrid restriction step."""
  # NOTE: Unlike the MATLAB implementation, This function uses a for loop for
  # `col` and `col_p`, since JAX is not able to statically determine the
  # values of `ih` and `lo`.
  nz = fx.shape[1]
  lo = 1 << idx  # 2^idx
  jh = (nz - 1) >> (idx + 2)  # (nz - 1) / 2^(idx + 2)
  jo_step = 2 * jh

  i = slice(1, -1)
  # Calculate the number of coarse rows to iterate over.
  num_iterations = (nz - 2 - jo_step) // (2 * jo_step) + 1
  num_l_iterations = (nz - 2 - lo) // (2 * lo) + 1

  def body_fun(loop_idx, carry):
    fx, p = carry

    # Calculate the current coarse-grid column index based on the loop index
    j = jo_step * (1 + loop_idx * 2)

    fx_update_vals = fx[i, j] - fx[i, j + jh] - fx[i, j - jh]
    p_loop = 2 * fx[i, j] + fx[i, j + jo_step] + fx[i, j - jo_step]

    def body_l_fun(loop_idx, p_carry):
      l = lo * (loop_idx * 2 + 1) - 1
      return _gsp(p_carry, cq[l, :], cr[l, :], cs[l, :])

    p_loop = jax.lax.fori_loop(0, num_l_iterations, body_l_fun, p_loop)

    return fx.at[i, j].set(fx_update_vals + p_loop), p.at[:, j - 1].set(p_loop)

  return jax.lax.cond(
      num_iterations > 0,
      functools.partial(jax.lax.fori_loop, 0, num_iterations, body_fun),
      _identity,
      (fx, p),
  )
