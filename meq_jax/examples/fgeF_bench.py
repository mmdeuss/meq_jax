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

r"""Benchmarks fgeF in MEQ-JAX against MEQ, on MEQ's own perf axes.

Mirrors MEQ's perf/fgeF_perf.m, which parameterizes a single fgeF evaluation
over

    grid  ny_1000 [32,16] | ny_2000 [32,32] | ny_4000 [64,32]
    shot  circular 1 | double_null 3 | doublet 82
    usecs false | true      (icsint, with ilim 1 or 3)
    jacx  false | true      (optsF dojacx)

and runs each configuration through both implementations from the same
Octave equilibrium, so the columns are comparable.

One deviation: fgeF_perf.m sets up with `fgs`, whose steady-state LX has no
`dt` and no CDE, both of which MEQ-JAX's fgeF needs. This uses the `fbt` plus
`meqxconvert` setup from the tests, over the same axes. That MEQ's fgeF
accepts the `fgs` LX and MEQ-JAX's does not is a parity gap of its own.

    python fgeF_bench.py                       # the full 36-configuration sweep
    python fgeF_bench.py --shots=1 --grids=32x16 --jacx=true

Both stacks must be on one machine for the ratio to mean anything, and MEQ is
required: unlike profile_fge.py this cannot replay a saved state, since Octave
is half the measurement.

Reading the numbers:

*   float64 throughout: parity needs it and MEQ uses it.
*   The JAX column leaves out compile time, reported separately. MEQ pays no
    such cost, so this flatters JAX for one-off runs and is fair for a
    simulation loop.
*   Timings are medians over --num_reps calls with the interquartile range.
    A large IQR relative to the median means a contended machine; treat the
    ratio as indicative only.
*   MEQ-JAX transposes arrays relative to Matlab, so the two do not issue the
    same BLAS calls. This compares the implementations as they are, not
    kernel for kernel.
"""

from collections.abc import Sequence
import functools
import itertools
import os
import platform
from pathlib import Path  # pylint: disable=g-importing-member
import statistics
import subprocess
import time

from absl import app
from absl import flags
import jax
from meq_jax._src import fgeF
from meq_jax._src import fgeF_octave
from meqpy import octave_utils
import numpy as np

_GRIDS = flags.DEFINE_list(
    'grids', ['32x16', '32x32', '64x32'], 'Grids as NRxNZ, per fgeF_perf.m.'
)
_SHOTS = flags.DEFINE_list(
    'shots', ['1', '3', '82'], 'ana shots: 1 circular, 3 double null, 82 doublet.'
)
_USECS = flags.DEFINE_list('usecs', ['false', 'true'], 'icsint settings.')
_JACX = flags.DEFINE_list('jacx', ['false', 'true'], 'optsF dojacx settings.')
_NUM_REPS = flags.DEFINE_integer('num_reps', 10, 'Timed calls per side.')
_WARMUP = flags.DEFINE_integer('warmup', 3, 'Untimed calls before timing.')
_SKIP_OCTAVE = flags.DEFINE_bool(
    'skip_octave', False, 'Time only MEQ-JAX (no comparison column).'
)

def _cpu_name():
  """Best-effort host CPU model. MEQ always runs here, so it is part of the result."""
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


def _median_iqr(samples):
  samples = sorted(samples)
  q1 = samples[len(samples) // 4]
  q3 = samples[(3 * len(samples)) // 4]
  return statistics.median(samples), q3 - q1


def _time_jax(fn, num_reps, warmup):
  for _ in range(warmup):
    jax.block_until_ready(fn())
  samples = []
  for _ in range(num_reps):
    t0 = time.perf_counter()
    jax.block_until_ready(fn())
    samples.append(time.perf_counter() - t0)
  return _median_iqr(samples)


def _time_octave(octave, num_reps, warmup):
  """Times fgeF in Octave, measured inside Octave to exclude oct2py transport."""
  octave.eval(f'for i_=1:{warmup}; [r_,LY_,Jx_,Ju_,Jd_,rm_] = '
              'fgeF(x0,L,LX,LX,opts,aux,u0); end')
  octave.eval(
      f'ts_ = zeros(1,{num_reps});\n'
      f'for i_=1:{num_reps}\n'
      '  t_=tic; [r_,LY_,Jx_,Ju_,Jd_,rm_] = fgeF(x0,L,LX,LX,opts,aux,u0);'
      ' ts_(i_)=toc(t_);\n'
      'end'
  )
  return _median_iqr(
      list(np.atleast_1d(np.squeeze(octave.eval('ts_;', nout=1))))
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print(f'backend : {jax.default_backend()}   devices: {jax.devices()}')
  devices = jax.devices()
  if devices and devices[0].platform != 'cpu':
    kind = getattr(devices[0], 'device_kind', devices[0].platform)
    print(f'device  : {kind} x{len(devices)}')
  print(f'host cpu: {_cpu_name()} ({os.cpu_count()} cores)')
  print(f'jax     : {jax.__version__}   dtype: float64')
  if jax.default_backend() != 'cpu' and not _SKIP_OCTAVE.value:
    print('NOTE: MEQ runs on the CPU regardless, so the ratio compares '
          'different hardware.')
  print()

  octave = octave_utils.create_meq_oct2py_instance()
  octave.addpath(str(Path(__file__).parent))

  hdr = (f'{"grid":>7} {"shot":>5} {"usecs":>6} {"jacx":>6} '
         f'{"jax ms":>9} {"±IQR":>7} {"meq ms":>9} {"±IQR":>7} '
         f'{"ratio":>7} {"compile s":>10}')
  print(hdr)
  print('-' * len(hdr))

  for grid, shot, usecs, jacx in itertools.product(
      _GRIDS.value, _SHOTS.value, _USECS.value, _JACX.value
  ):
    nr, nz = (int(v) for v in grid.lower().split('x'))
    shot = int(shot)
    try:
      octave.eval(fgeF_octave.FBT_SETUP.format(
          shot=shot, icsint='true' if usecs == 'true' else 'false',
          algosNL='all-nl', grid=f",'nr',{nr},'nz',{nz}"))
      octave.eval(f"opts = optsF('dojacx',{jacx},'dojacu',{jacx},"
                  f"'dojacxdot',{jacx},'dopost',true);")
      inputs = fgeF_octave.fgeF_inputs_from_octave(
          octave, shot=shot, test_type='fbt'
      )
    except Exception as e:  # pylint: disable=broad-except
      print(f'{grid:>7} {shot:>5} {usecs:>6} {jacx:>6}   setup failed: '
            f'{type(e).__name__}: {str(e).splitlines()[-1][:40]}')
      continue

    fn = jax.jit(
        functools.partial(
            fgeF.fgeF,
            L=inputs.L,
            opts=inputs.opts,
            aux=inputs.aux,
            agconc=inputs.agconc,
            cdeconc=inputs.cdeconc,
            u=inputs.u0,
        )
    )
    call = lambda _f=fn, _i=inputs: _f(
        x=_i.x0, LX=_i.LX, LYp=_i.LX, xdot=None
    )

    try:
      t0 = time.perf_counter()
      jax.block_until_ready(call())
      compile_s = time.perf_counter() - t0
      jax_med, jax_iqr = _time_jax(call, _NUM_REPS.value, _WARMUP.value)
    except Exception as e:  # pylint: disable=broad-except
      print(f'{grid:>7} {shot:>5} {usecs:>6} {jacx:>6}   jax failed: '
            f'{type(e).__name__}: {str(e).splitlines()[-1][:40]}')
      continue

    if _SKIP_OCTAVE.value:
      print(f'{grid:>7} {shot:>5} {usecs:>6} {jacx:>6} '
            f'{jax_med * 1e3:>9.3f} {jax_iqr * 1e3:>7.3f} '
            f'{"-":>9} {"-":>7} {"-":>7} {compile_s:>10.2f}')
      continue

    meq_med, meq_iqr = _time_octave(octave, _NUM_REPS.value, _WARMUP.value)
    print(f'{grid:>7} {shot:>5} {usecs:>6} {jacx:>6} '
          f'{jax_med * 1e3:>9.3f} {jax_iqr * 1e3:>7.3f} '
          f'{meq_med * 1e3:>9.3f} {meq_iqr * 1e3:>7.3f} '
          f'{meq_med / jax_med:>6.2f}x {compile_s:>10.2f}')

  print('\nratio > 1 means MEQ-JAX is faster on this machine.')


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  app.run(main)
