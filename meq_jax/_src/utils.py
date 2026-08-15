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

"""Utility functions for MEQ."""

import dataclasses
from typing import Any

from absl import logging
import chex
import jax
import jax.numpy as jnp
import jaxtyping as jt
from meqpy import meqpy_impl
import numpy as np
import oct2py
import scipy.sparse

from meq_jax._src import types


def struct_to_dataclass(struct: oct2py.Struct, cls: type[Any]) -> Any:
  """Converts an oct2py.Struct to a dataclass."""
  data_class = cls()
  for field in dataclasses.fields(cls):
    if field.name in struct:
      if field.name == 'P':
        val = struct_to_dataclass(struct[field.name], types.ParameterData)
      elif field.name == 'G':
        val = struct_to_dataclass(struct[field.name], types.GeometryData)
      elif field.name == 'bfp':
        # TODO(pfau, adedieu): Enable BfpData for doublets.
        bfp = struct[field.name].squeeze()
        val = types.BfpData(nP=int(bfp[0]), nT=int(bfp[1]))
      elif field.name == 'ind':
        val = struct_to_dataclass(struct[field.name], types.IndexData)
        for field_ in dataclasses.fields(types.IndexData):
          val_ = getattr(val, field_.name)
          if isinstance(val_, jnp.ndarray):
            setattr(val, field_.name, val_.astype(jnp.int32))
          elif isinstance(val_, float):
            setattr(val, field_.name, int(val_))
      else:
        val = struct[field.name]
        if field.name not in types.STR_FIELDS:
          if isinstance(val, scipy.sparse.csc_array):
            val = val.todense()
          val = jnp.asarray(np.squeeze(val))
          if val.ndim == 2:
            val = val.T
          if field.name == 'TDg' and val.ndim == 1:
            # Special case because the first dimension is the number of domains.
            val = val[None]
          if field.name in ['i95', 'kxl']:
            # Special case for an int array outside of IndexData
            val = val.astype(jnp.int32)
          if val.size == 1:
            val = val.item()
          if field.name in types.INT_FIELDS:
            val = int(val)
          elif field.name in types.BOOL_FIELDS:
            val = bool(val)
        if isinstance(val, oct2py.io.Cell):
          # Special case for initguess
          val = tuple(val.tolist()[0])
      setattr(data_class, field.name, val)
  return data_class


def dataclass_to_octave_shapes(
    output: types.OutputData, ly: oct2py.Struct) -> types.OutputData:
  """Converts a dataclass to an oct2py.Struct."""
  padded_fields = ['FA', 'rA', 'zA', 'FB', 'rB', 'zB', 'FX', 'rX', 'zX',
                   'dr2FA', 'dz2FA', 'drzFA', 'dr2FX', 'dz2FX', 'drzFX']
  reshaped_output = types.OutputData()
  for field in dataclasses.fields(types.OutputData):
    if hasattr(output, field.name):
      val = getattr(output, field.name)
      if val is not None:
        if (isinstance(val, float) or
            isinstance(val, int) or
            (isinstance(val, jnp.ndarray) and val.ndim == 0)):
          # Handle scalar values.
          if ly[field.name].ndim == 0:
            val = jnp.array(val)
          elif ly[field.name].ndim == 1:
            val = jnp.array([val])
          elif ly[field.name].ndim == 2:
            val = jnp.array([[val]])
        if val.ndim == 2:
          val = val.T
        if val.ndim == 3:
          val = val.transpose((2, 1, 0))
          if ly[field.name].ndim != 3:
            # Special case for aq
            val = jnp.squeeze(val)
        if (isinstance(ly[field.name], np.ndarray) and
            ly[field.name].ndim == val.ndim + 1):
          val = jnp.reshape(val, ly[field.name].shape)
        if field.name in padded_fields:
          if field.name.endswith('X'):
            # For some reason, nX is an array, not a scalar, in the oct2py LY.
            if isinstance(output.nX, chex.Array):
              nx = output.nX.item()
            else:
              nx = output.nX
            val = val[:nx, ...]
          elif field.name.endswith('A'):
            val = val[:output.nA, ...]
          elif field.name.endswith('B'):
            val = val[:output.nB, ...]
        if field.name == 'lX':
          # Special case for lX
          val = val[jnp.isfinite(output.rB), ...]
        setattr(reshaped_output, field.name, val)
  return reshaped_output


# TODO(pfau): figure out how to type-check dataclasses
def transpose(data: jt.PyTree) -> jt.PyTree:
  """Recursively transposes 2D JAX arrays in a dataclass."""
  return jax.tree_util.tree_map(_transpose_element, data)


def _transpose_element(val):
  if isinstance(val, chex.Array):
    match val.ndim:
      case 0 | 1:
        return val
      case 2:
        return val.T
      case ndim:
        return val.transpose(range(ndim-1, -1, -1))
  return val


def init_from_octave(meq_instance: meqpy_impl.MeqPy,
                     ly: oct2py.Struct,
                     precompute_gszr_operator: bool = True,
                     clamp_dimw: int | None = None,
                     ) -> tuple[types.StateData,
                                types.StaticData,
                                types.InputData,
                                int,
                                list[types.ConcData],
                                list[types.ConcData]]:
  """Initializes the state from an oct2py instance and an octave LY.

  Args:
    meq_instance: The MeqPy instance to initialize from.
    ly: The LY struct from the octave instance.
    precompute_gszr_operator: If True, materialize the (linear) gszr Poisson
      solve as dense operators (see gszr.gszr_operator). This makes the solve
      a single matmul, which is dramatically faster on accelerators; results
      agree with the sequential gszrjax algorithm up to floating-point
      summation order.
    clamp_dimw: If set, clamp L.dimw (the maximum number of flux extrema
      tracked by asxy/meqpdom; 200 by default in MEQ) to this value. The
      domain-identification loop in meqpdom runs 2*dimw sequential steps, so
      a smaller bound is substantially faster (especially on accelerators).
      Results are bit-identical as long as the actual number of detected
      extrema stays below the bound (typical equilibria have < 10); if it is
      exceeded, asxy reports failure via its stat output, as in MEQ.

  Returns:
    A (state, static, lx, num_steps, agconc, cdeconc) tuple.
  """
  jax_output = struct_to_dataclass(ly, types.OutputData)
  state = types.StateData(
      LYt=jax_output,
      it=1,
      dt=meq_instance.octave_eval('State.dt;', nout=1).item(),
      # Prec is symmetric so no transpose is needed.
      Prec=meq_instance.octave_eval('State.Prec;', nout=1),
      xnl=jnp.squeeze(meq_instance.octave_eval('State.xnl;', nout=1)),
      xnldot=jnp.squeeze(
          meq_instance.octave_eval('State.xnldot;', nout=1)),
      Tstate=jnp.squeeze(
          meq_instance.octave_eval('State.Tstate;', nout=1)),
      PSstate=meq_instance.octave_eval('State.PSstate;', nout=1),
      dstate=jnp.squeeze(jnp.array(
          meq_instance.octave_eval('State.dstate;', nout=1))),
      nnoc=int(meq_instance.octave_eval('State.nnoc;', nout=1).item()),
  )
  meq_instance._strip_function_handles('Lfge', 'Lfge_safe')  # pylint: disable=protected-access
  l_struct = meq_instance.octave_eval('Lfge_safe;', nout=1)
  static = struct_to_dataclass(l_struct, types.StaticData)
  if clamp_dimw is not None and static.dimw is not None:
    static = dataclasses.replace(static, dimw=min(static.dimw, clamp_dimw))
  if precompute_gszr_operator and static.cx is not None:
    from meq_jax._src import gszr  # pylint: disable=g-import-not-at-top

    bc_op, iy_op = gszr.gszr_operator(
        static.cx, static.cq, static.cr, static.cs, static.ci, static.co,
        nr2=static.nry, nz2=static.nzy)
    static = dataclasses.replace(static, gszr_bc_op=bc_op, gszr_iy_op=iy_op)
  lx_struct = meq_instance.octave_eval('LXfge_working;', nout=1)
  lx = struct_to_dataclass(lx_struct, types.InputData)
  num_steps = int(meq_instance.octave_eval('num_steps;', nout=1).item())

  agconc = []
  fun_names = meq_instance.octave_eval(
      'Lfge_safe.agconc(:,3);').squeeze().tolist()
  ids = meq_instance.octave_eval(
      'Lfge_safe.agconc(:,2);').squeeze().tolist()
  iis = (meq_instance.octave_eval(
      'Lfge_safe.agconc(:,4);').squeeze()).tolist()
  for fn_name, id_, ii in zip(fun_names, ids, iis):
    agdata = types.ConcData(
        fun_name=fn_name,
        domain_id=int(id_.item()),
        lx_name=fn_name,
        index=int(ii.item())
    )
    agconc.append(agdata)

  cdeconc = []
  cdec = meq_instance.octave_eval('Lfge_safe.cdec;', nout=1)
  if cdec.size > 0:
    cdeconc.append(
        types.ConcData(
            fun_name=cdec[0, 0],
            domain_id=int(cdec[0, 1].item()),
            lx_name='',
            index=int(cdec[0, 3].item()),
        )
    )
  return state, static, lx, num_steps, agconc, cdeconc


# Fields that are padded with -inf up to a static maximum size (L.dimw etc.)
# in the JAX implementation.
_INF_PADDED_FIELDS = frozenset([
    'FA', 'rA', 'zA', 'FB', 'rB', 'zB', 'FX', 'rX', 'zX',
    'dr2FA', 'dz2FA', 'drzFA', 'dr2FX', 'dz2FX', 'drzFX',
])


def _normalize_leaf(name: str, val: Any, target: Any) -> Any:
  """Converts one field value to the target shape/dtype."""
  if target is None:
    return None
  if not isinstance(target, (jax.ShapeDtypeStruct, jnp.ndarray, np.ndarray)):
    return val  # Non-array field (e.g. static/None): keep as is.
  if val is None:
    fill = -jnp.inf if name in _INF_PADDED_FIELDS else 0
    return jnp.full(target.shape, fill, dtype=target.dtype)
  val = jnp.asarray(val)
  if val.shape == target.shape:
    return val.astype(target.dtype)
  if val.size == np.prod(target.shape):
    return val.reshape(target.shape).astype(target.dtype)
  if val.size > np.prod(target.shape):
    raise ValueError(
        f'Cannot normalize field {name}: value of shape {val.shape} is larger'
        f' than target {target.shape}.')
  if len(target.shape) != 1:
    raise ValueError(
        f'Cannot normalize field {name}: cannot pad shape {val.shape} to'
        f' {target.shape}.')
  fill = -jnp.inf if name in _INF_PADDED_FIELDS else 0
  out = jnp.full(target.shape, fill, dtype=target.dtype)
  return out.at[: val.size].set(val.reshape(-1).astype(target.dtype))


def _normalize_dataclass(value: Any, target: Any) -> Any:
  """Recursively normalizes dataclass fields to the target structure."""
  updates = {}
  for field in dataclasses.fields(value):
    v = getattr(value, field.name)
    t = getattr(target, field.name)
    if dataclasses.is_dataclass(v) and dataclasses.is_dataclass(t):
      updates[field.name] = _normalize_dataclass(v, t)
    else:
      updates[field.name] = _normalize_leaf(field.name, v, t)
  return dataclasses.replace(value, **updates)


def normalize_state(
    state: types.StateData,
    static: types.StaticData,
    lx: types.InputData,
    agconc: list[types.ConcData] | None = None,
    cdeconc: list[types.ConcData] | None = None,
) -> types.StateData:
  """Converts an Octave-initialized state to the steady-state pytree structure.

  A state freshly produced by `init_from_octave` has Python scalars and
  unpadded arrays where the state produced by an environment step has (padded)
  JAX arrays. As a consequence, `jax.jit(fgetk_environment)` compiles twice:
  once for the initial structure and once for the steady structure. This
  function pads/casts the initial state to the steady structure (computed
  abstractly with `jax.eval_shape`, i.e. without compiling anything) so only
  one compilation is ever needed.
  """
  from meq_jax._src import fgetk_environment as _fgetk_env  # pylint: disable=g-import-not-at-top

  assert static.G is not None and static.G.na is not None
  voltages = jax.ShapeDtypeStruct((static.G.na,), jnp.float64)

  def one_step(s, v):
    return _fgetk_env._fgetk_step(  # pylint: disable=protected-access
        s, v, static, lx, agconc, cdeconc, {})[0]

  target = jax.eval_shape(one_step, state, voltages)
  return _normalize_dataclass(state, target)


def compare_octave_and_jax(
    ly: oct2py.Struct,
    ly_jax: types.OutputData,
    skip_fields: list[str] | None = None):
  """Compares the Octave and JAX outputs."""
  # TODO(pfau): dig deeper into numerical discrepancies in Iu and resFx.
  skip_fields = skip_fields or []
  isempty = (lambda x: x is None or
             (isinstance(x, list) and not x) or
             (isinstance(x, chex.Array) and x.size == 0))
  for field in dataclasses.fields(ly_jax):
    if field.name in ly.keys() and field.name not in skip_fields:
      logging.info('Comparing LYs for field %s', field.name)
      jax_field = getattr(ly_jax, field.name)
      if isempty(jax_field) and isempty(ly[field.name]):
        continue
      if (isempty(jax_field) and not isempty(ly[field.name]) or
          not isempty(jax_field) and isempty(ly[field.name])):
        logging.warning('Field %s is empty in one LY but not the other.',
                        field.name)
        # raise ValueError(
        #     f'Field {field.name} is empty in one LY but not the other.')
      elif not isinstance(jax_field, chex.Array):
        if not np.allclose(
            jax_field, ly[field.name], atol=1e-5, rtol=1e-5):
          raise ValueError(
              f'Field {field.name} is not equal in JAX and Matlab LYs.')
      elif jax_field.shape != ly[field.name].shape:
        jax_field = jax_field.reshape(ly[field.name].shape)
      if (not np.allclose(
          jax_field, ly[field.name], atol=1e-4, rtol=1e-5)):
        # Special case for FX, where sign is swapped
        if not np.all(np.logical_and(np.isinf(jax_field),
                                     np.isinf(ly[field.name]))):
          raise ValueError(
              f'Field {field.name} is not equal in JAX and Matlab LYs.')
