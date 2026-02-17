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

"""Example run of MEQ-JAX, comparing against MEQ-Octave."""

from collections.abc import Sequence
import functools
from pathlib import Path  # pylint: disable=g-importing-member

from absl import app
from absl import logging
import jax
import matplotlib.pyplot as plt
from meq_jax._src import fgetk_environment
from meq_jax._src import utils
from meqpy import meqpy_impl
import numpy as np

# Turn on float64 for comparison with Matlab implementation
jax.config.update('jax_enable_x64', True)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  # Based off the minimal example in the meqpy README.

  tokamak = 'ana'
  time = 0  # initial time
  source = meqpy_impl.MeqSource.MEQ_DIRECT
  shot = 2

  control_timestep = 1e-4
  simulator_timestep = 1e-4

  # Initialize Octave
  meq_instance = meqpy_impl.MeqPy()
  # Include path for strip_function_handles.m
  meq_instance.octave_eval(f"addpath('{str(Path(__file__).parent)}');")
  meq_instance.init_fge(
      tokamak, shot, time, source, default_meq_params={'debug': 2}
    )

  ly, _ = meq_instance.init_fgetk_environment(
      control_timestep=control_timestep,
      simulator_timestep=simulator_timestep)

  state, static, lx, num_steps, agconc, cdeconc = utils.init_from_octave(
      meq_instance, ly
  )

  jit_fgetk_environment = jax.jit(
      functools.partial(
          fgetk_environment.fgetk_environment,
          static=static,
          lx=lx,
          num_steps=num_steps,
          agconc=agconc,
          cdeconc=cdeconc,
      )
  )

  num_actions = len(meq_instance.get_fgetk_circuit_names())
  actions = np.full((num_actions,), 10)

  # get the r and z coordinates of the flux grid
  rx = meq_instance.get_fge_geometry('rx').reshape(-1)
  zx = meq_instance.get_fge_geometry('zx').reshape(-1)

  for i in range(3):
    # run one step of the Octave environment
    LY_octave, _ = meq_instance.step_fgetk_environment(actions, {})  # pylint: disable=invalid-name

    # run one step of the JAX environment
    LY_jax, _, state, _ = jit_fgetk_environment(  # pylint: disable=invalid-name
        state=state,
        voltages=actions,
    )
    LY_jax = utils.dataclass_to_octave_shapes(LY_jax, LY_octave)  # pylint: disable=invalid-name

    # compare and display
    try:
      utils.compare_octave_and_jax(
          LY_octave,
          LY_jax,
          skip_fields=[
              'niter',
              'nfeval',
              'mkryl',
              'Iu',
              'resFx',
              'Parel',
              'Tarel',
              'Iarel',
          ],
      )
      logging.info('Iteration %d: all fields match.', i + 1)
    except ValueError as e:
      logging.warning('Iteration %d: %s', i + 1, e)

  plt.contour(rx, zx, LY_jax.Fx, LY_jax.FB, colors='k')
  plt.contour(rx, zx, LY_jax.Fx, LY_jax.FB, colors='k')

  plt.axis('equal')
  plt.show()


if __name__ == '__main__':
  app.run(main)
