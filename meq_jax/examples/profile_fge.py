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

r"""Profiles the MEQ-JAX FGE environment step on whatever backend JAX finds.

Measures what the README claims: compile time, time for one simulation step,
and how per-simulation time scales when the step is `jax.vmap`-ed over a batch.

Initialization needs Octave and MEQ; profiling does not. A run can save the
state so later runs replay it on a machine with only JAX:

    # once, on a machine with MEQ
    python profile_fge.py --save_init=/tmp/fge_init.pkl

    # thereafter, anywhere -- e.g. a GPU box with no Octave
    python profile_fge.py --load_init=/tmp/fge_init.pkl \
        --batch_sizes=1,8,64,256 --trace_dir=/tmp/meqjax_trace

View a trace with `tensorboard --logdir /tmp/meqjax_trace`, or open the
`.trace.json.gz` at https://ui.perfetto.dev.

Reading the numbers:

*   float64 is the default, because parity needs it. Consumer NVIDIA cards run
    it far slower than float32, so `--nox64` gives a float32 comparison -- a
    performance datapoint only, not a correct one.
*   "first call" includes tracing and compilation. "step" is the median of
    `--num_reps` calls, each blocked on. Building the inputs is left out of the
    timed region, so `--donate` is measured fairly.
*   A large interquartile range means a busy machine and an untrustworthy
    median.
"""

from collections.abc import Sequence
import functools
import os
import pickle
import platform
import re
import statistics
import subprocess
import time

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
from meq_jax._src import fgetk_environment
from meq_jax._src import utils
import numpy as np

_TOKAMAK = flags.DEFINE_string('tokamak', 'ana', 'Tokamak to initialize.')
_SHOT = flags.DEFINE_integer('shot', 2, 'Shot number to initialize.')
_TIME = flags.DEFINE_float('time', 0.0, 'Initial time.')
_CONTROL_DT = flags.DEFINE_float('control_timestep', 1e-4, 'Control timestep.')
_SIM_DT = flags.DEFINE_float('simulator_timestep', 1e-4, 'Simulator timestep.')

_BATCH_SIZES = flags.DEFINE_list(
    'batch_sizes', ['1'], 'Batch sizes to vmap the step over.'
)
_NUM_REPS = flags.DEFINE_integer('num_reps', 20, 'Timed calls per batch size.')
_WARMUP = flags.DEFINE_integer('warmup', 3, 'Untimed calls before timing.')

_X64 = flags.DEFINE_bool(
    'x64', True, 'Use float64. Required for parity; slow on most GPUs.'
)
_DONATE = flags.DEFINE_bool(
    'donate', True, 'Donate the state buffers (see README).'
)
_PRECOMPUTE_GSZR = flags.DEFINE_bool(
    'precompute_gszr', True, 'Materialize the gszr Poisson solve as a matmul.'
)
_CLAMP_DIMW = flags.DEFINE_integer(
    'clamp_dimw', 32, 'Clamp L.dimw to this. Negative leaves the MEQ default.'
)
_NORMALIZE_STATE = flags.DEFINE_bool(
    'normalize_state', True, 'Normalize the state so one compile suffices.'
)

_TRACE_DIR = flags.DEFINE_string(
    'trace_dir', None, 'If set, write a jax.profiler trace here.'
)
_CACHE_DIR = flags.DEFINE_string(
    'compilation_cache_dir', None, 'If set, enable the persistent XLA cache.'
)
_SAVE_INIT = flags.DEFINE_string(
    'save_init', None, 'If set, pickle the initialized state here.'
)
_LOAD_INIT = flags.DEFINE_string(
    'load_init', None, 'If set, load a pickled state instead of using Octave.'
)


def _init_from_octave():
  """Initializes from Octave. meqpy is imported lazily so --load_init needs no MEQ."""
  from meqpy import meqpy_impl  # pylint: disable=g-import-not-at-top

  meq_instance = meqpy_impl.MeqPy()
  meq_instance.init_fge(
      _TOKAMAK.value,
      _SHOT.value,
      _TIME.value,
      meqpy_impl.MeqSource.MEQ_DIRECT,
      default_meq_params={'debug': 2},
  )
  ly, _ = meq_instance.init_fgetk_environment(
      control_timestep=_CONTROL_DT.value,
      simulator_timestep=_SIM_DT.value,
  )
  clamp = _CLAMP_DIMW.value if _CLAMP_DIMW.value >= 0 else None
  state, static, lx, num_steps, agconc, cdeconc = utils.init_from_octave(
      meq_instance, ly,
      precompute_gszr_operator=_PRECOMPUTE_GSZR.value,
      clamp_dimw=clamp,
  )
  return (state, static, lx, num_steps, agconc, cdeconc,
          len(meq_instance.get_fgetk_circuit_names()))


def _cpu_name():
  """Best-effort host CPU model, so a result is attributable to hardware."""
  try:
    with open('/proc/cpuinfo') as f:
      for line in f:
        if line.startswith('model name'):
          return line.split(':', 1)[1].strip()
  except OSError:
    pass
  try:
    return subprocess.run(
        ['sysctl', '-n', 'machdep.cpu.brand_string'],
        capture_output=True, text=True, check=True).stdout.strip()
  except (OSError, subprocess.SubprocessError):
    pass
  return platform.processor() or platform.machine()


def _hardware_lines():
  """Two lines naming the accelerator, if any, and the host CPU."""
  lines = []
  devices = jax.devices()
  if devices and devices[0].platform != 'cpu':
    kind = getattr(devices[0], 'device_kind', devices[0].platform)
    lines.append(f'device     : {kind} x{len(devices)}')
  lines.append(f'host cpu   : {_cpu_name()} ({os.cpu_count()} cores)')
  return lines


def _to_host(tree):
  return jax.tree.map(
      lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, tree
  )


def _to_device(tree):
  return jax.tree.map(
      lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, tree
  )


def _replicate(tree, batch_size):
  """Broadcasts every array leaf onto a leading batch axis."""
  return jax.tree.map(
      lambda x: jnp.broadcast_to(jnp.asarray(x), (batch_size,) + jnp.shape(x)),
      tree,
  )


def _time_steady_state(fn, make_args, num_reps, warmup):
  """Medians the per-call time, excluding argument construction.

  Arguments are rebuilt for every call because --donate invalidates the input
  buffers, and are materialized on device before the clock starts so that only
  the step itself is timed.
  """
  for _ in range(warmup):
    jax.block_until_ready(fn(*make_args()))
  samples = []
  for _ in range(num_reps):
    args = jax.block_until_ready(make_args())
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    samples.append(time.perf_counter() - t0)
  samples.sort()
  q1 = samples[len(samples) // 4]
  q3 = samples[(3 * len(samples)) // 4]
  return statistics.median(samples), q3 - q1


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if _CACHE_DIR.value:
    jax.config.update('jax_compilation_cache_dir', _CACHE_DIR.value)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.5)

  backend = jax.default_backend()
  print(f'backend    : {backend}')
  print(f'devices    : {jax.devices()}')
  for _line in _hardware_lines():
    print(_line)
  print(f'jax        : {jax.__version__}')
  print(f'dtype      : {"float64" if _X64.value else "float32"}')
  print(f'donate={_DONATE.value}  precompute_gszr={_PRECOMPUTE_GSZR.value}  '
        f'clamp_dimw={_CLAMP_DIMW.value}  '
        f'normalize_state={_NORMALIZE_STATE.value}')

  print()

  if _LOAD_INIT.value:
    if not os.path.exists(_LOAD_INIT.value):
      raise app.UsageError(
          f'No saved state at {_LOAD_INIT.value}. It is written by a run with'
          ' --save_init, on a machine that has MEQ and Octave. If this machine'
          ' has them, drop --load_init and pass --save_init instead to'
          ' initialize from Octave and save the state for later runs.'
      )
    with open(_LOAD_INIT.value, 'rb') as f:
      p = pickle.load(f)
    if p['x64'] != _X64.value:
      logging.warning('State saved with x64=%s, running with x64=%s; arrays '
                      'will be cast.', p['x64'], _X64.value)
    state, static, lx = (_to_device(p['state']), _to_device(p['static']),
                         _to_device(p['lx']))
    num_steps, num_actions = p['num_steps'], p['num_actions']
    agconc, cdeconc = _to_device(p['agconc']), _to_device(p['cdeconc'])
    print(f'loaded initialized state from {_LOAD_INIT.value}')
  else:
    (state, static, lx, num_steps, agconc, cdeconc,
     num_actions) = _init_from_octave()
    print('initialized from Octave')

  if _SAVE_INIT.value:
    with open(_SAVE_INIT.value, 'wb') as f:
      pickle.dump({'state': _to_host(state), 'static': _to_host(static),
                   'lx': _to_host(lx), 'num_steps': num_steps,
                   'agconc': _to_host(agconc), 'cdeconc': _to_host(cdeconc),
                   'num_actions': num_actions, 'x64': _X64.value}, f)
    print(f'saved initialized state to {_SAVE_INIT.value}')

  if _NORMALIZE_STATE.value:
    state = utils.normalize_state(state, static, lx, agconc, cdeconc)

  step = functools.partial(
      fgetk_environment.fgetk_environment,
      static=static, lx=lx, num_steps=num_steps,
      agconc=agconc, cdeconc=cdeconc,
  )
  voltages = jnp.full((num_actions,), 10.0)
  donate = 'state' if _DONATE.value else ()

  print(f'{"batch":>7}  {"first call s":>12}  {"step ms":>10}  {"IQR ms":>8}  '
        f'{"per-sim ms":>11}  {"speedup":>8}')
  print('-' * 68)

  baseline = None
  for raw in _BATCH_SIZES.value:
    batch_size = int(raw)
    if batch_size == 1:
      fn = jax.jit(step, donate_argnames=donate)
      make_args = lambda: (jax.tree.map(jnp.copy, state), voltages)
    else:
      fn = jax.jit(jax.vmap(step, in_axes=(0, 0)), donate_argnames=donate)
      bstate = _replicate(state, batch_size)
      bvolt = _replicate(voltages, batch_size)
      make_args = lambda _s=bstate, _v=bvolt: (jax.tree.map(jnp.copy, _s), _v)

    # Running out of device memory is an expected outcome of a batch sweep --
    # it is how the ceiling is found -- so report it and carry on rather than
    # discarding the rows already measured.
    try:
      # The first call traces and compiles; time it separately.
      t0 = time.perf_counter()
      jax.block_until_ready(fn(*make_args()))
      first_call_s = time.perf_counter() - t0

      median, iqr = _time_steady_state(
          fn, make_args, num_reps=_NUM_REPS.value, warmup=_WARMUP.value
      )
    except jax.errors.JaxRuntimeError as e:
      if 'RESOURCE_EXHAUSTED' not in str(e):
        raise
      want = re.search(r'allocate ([0-9.]+ ?[KMG]i?B)', str(e))
      note = 'out of memory'
      if want:
        note += f' (a further {want.group(1)} was needed)'
      print(f'{batch_size:>7}  {note:>54}')
      break
    per_sim = median / batch_size
    if baseline is None:
      baseline = per_sim
    print(f'{batch_size:>7}  {first_call_s:>12.2f}  {median * 1e3:>10.3f}  '
          f'{iqr * 1e3:>8.3f}  {per_sim * 1e3:>11.4f}  '
          f'{baseline / per_sim:>7.2f}x')

  if _TRACE_DIR.value:
    batch_size = int(_BATCH_SIZES.value[-1])
    if batch_size == 1:
      fn, args = jax.jit(step), (state, voltages)
    else:
      fn = jax.jit(jax.vmap(step, in_axes=(0, 0)))
      args = (_replicate(state, batch_size), _replicate(voltages, batch_size))
    jax.block_until_ready(fn(*args))  # compile outside the trace
    with jax.profiler.trace(_TRACE_DIR.value):
      for i in range(3):
        with jax.profiler.TraceAnnotation(f'fgetk_environment step {i}'):
          jax.block_until_ready(fn(*args))
    print(f'\ntrace written to {_TRACE_DIR.value}')
    print(f'  tensorboard --logdir {_TRACE_DIR.value}')
    print('  or open the .trace.json.gz at https://ui.perfetto.dev')


if __name__ == '__main__':
  # x64 has to be set before any array is created, which means before main
  # runs but after the flags are parsed.
  flags.FLAGS(__import__('sys').argv, known_only=True)
  jax.config.update('jax_enable_x64', _X64.value)
  app.run(main)
