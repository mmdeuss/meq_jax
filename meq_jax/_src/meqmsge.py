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

"""Python equivalent of meqmsge.m for logging and errors."""

import logging
import sys
import numpy as np

# Configure logger
logger = logging.getLogger('meq')


def _setup_logger():
  if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    # Basic formatter, can be customized
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
  logger.setLevel(logging.INFO)


_setup_logger()


class MeqError(Exception):
  """Base exception for MEQ errors."""


def meqmsge(mode, callfct, tok, t, it, shot, txt, mnem):
  """Displays errors/warnings/information during MEQ run-time.

  Args:
      mode: 'i' (info), 'w' (warning), 'e' (error)
      callfct: Calling function name (str)
      tok: Tokamak name (str)
      t: Time or time range (float or array-like)
      it: Iteration or iteration range (int or array-like)
      shot: Shot number (int)
      txt: Message text (str)
      mnem: Mnemonic for the message ID (str)

  Raises:
      MeqError: If mode is 'e' (error).
      ValueError: If the format of t or it is invalid.
  """
  try:
    it_arr = np.asarray(it)
    if it_arr.ndim == 0:
      it_str = str(it_arr.item())
    elif it_arr.size == 1:
      it_str = str(it_arr.item())
    elif it_arr.size == 2:
      it_str = f'{it_arr[0].item()}.{it_arr[1].item()}'
    else:
      raise ValueError('Unsupported number of elements in it')
  except Exception as e:
    raise ValueError(f'Invalid format for it: {it}') from e

  msg_id = f'{callfct}:{mnem}'

  try:
    t_arr = np.asarray(t)
    if t_arr.ndim == 0:
      msg = f'{tok}#{shot} {t_arr.item():6.4f}s/{it_str}: {txt}'
    elif t_arr.size > 0:
      msg = (
          f'{tok}#{shot} {t_arr[0].item():6.4f}-{t_arr[-1].item():6.4f}s/{it_str}:'
          f' {txt}'
      )
    else:  # Empty array
      msg = f'{tok}#{shot} ??s/{it_str}: {txt}'
  except Exception as e:
    raise ValueError(f'Invalid format for t: {t}') from e

  if mode == 'i':
    logger.info(msg)
  elif mode == 'w':
    logger.warning('%s - %s', msg_id, msg)
  elif mode == 'e':
    raise MeqError(f'{msg_id} - {msg}')
  else:
    logger.error('Unknown meqmsge mode: %s - %s - %s', mode, msg_id, msg)
