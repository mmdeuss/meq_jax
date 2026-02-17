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

import logging
import unittest
from unittest import mock

from meq_jax._src import meqmsge
from meq_jax._src.meqmsge import MeqError
import numpy as np


class MeqmsgeTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # Patch the logger to capture messages
    self.mock_logger = mock.patch.object(
        meqmsge, 'logger', autospec=True
    ).start()
    self.addCleanup(mock.patch.stopall)

  def test_info_mode(self):
    meqmsge.meqmsge(
        'i', 'test_func', 'TOK', 1.23456, 1, 123, 'Info message', 'INFO01'
    )
    self.mock_logger.info.assert_called_once_with(
        'TOK#123 1.2346s/1: Info message'
    )

  def test_warning_mode(self):
    meqmsge.meqmsge(
        'w',
        'test_func',
        'TOK',
        [1.2, 1.3],
        [2, 3],
        123,
        'Warning message',
        'WARN01',
    )
    self.mock_logger.warning.assert_called_once_with(
        '%s - %s', 'test_func:WARN01', 'TOK#123 1.2000-1.3000s/2.3: Warning message'
    )

  def test_error_mode(self):
    with self.assertRaises(MeqError) as context:
      meqmsge.meqmsge(
          'e', 'test_func', 'TOK', 1.5, 5, 123, 'Error message', 'ERR01'
      )
    self.assertEqual(
        str(context.exception),
        'test_func:ERR01 - TOK#123 1.5000s/5: Error message',
    )

  def test_time_formats(self):
    # Scalar time
    meqmsge.meqmsge('i', 'tf', 'T', 1.0, 1, 0, 'm', 'T01')
    self.mock_logger.info.assert_called_with('T#0 1.0000s/1: m')
    # Array time
    meqmsge.meqmsge('i', 'tf', 'T', np.array([1.0, 2.0]), 1, 0, 'm', 'T02')
    self.mock_logger.info.assert_called_with('T#0 1.0000-2.0000s/1: m')
    # Empty time
    meqmsge.meqmsge('i', 'tf', 'T', np.array([]), 1, 0, 'm', 'T03')
    self.mock_logger.info.assert_called_with('T#0 ??s/1: m')

  def test_iteration_formats(self):
    # Scalar int
    meqmsge.meqmsge('i', 'tf', 'T', 1.0, 1, 0, 'm', 'I01')
    self.mock_logger.info.assert_called_with('T#0 1.0000s/1: m')
    # np array scalar
    meqmsge.meqmsge('i', 'tf', 'T', 1.0, np.array(2), 0, 'm', 'I02')
    self.mock_logger.info.assert_called_with('T#0 1.0000s/2: m')
    # np array size 1
    meqmsge.meqmsge('i', 'tf', 'T', 1.0, np.array([3]), 0, 'm', 'I03')
    self.mock_logger.info.assert_called_with('T#0 1.0000s/3: m')
    # np array size 2
    meqmsge.meqmsge('i', 'tf', 'T', 1.0, np.array([4, 5]), 0, 'm', 'I04')
    self.mock_logger.info.assert_called_with('T#0 1.0000s/4.5: m')

  def test_invalid_it_format(self):
    with self.assertRaises(ValueError) as context:
      meqmsge.meqmsge(
          'i', 'tf', 'T', 1.0, np.array([1, 2, 3]), 0, 'm', 'ITERR'
      )
    self.assertIn('Invalid format for it', str(context.exception))

  def test_invalid_t_format(self):
    with self.assertRaises(ValueError) as context:
      meqmsge.meqmsge('i', 'tf', 'T', 'should_be_numeric', 1, 0, 'm', 'TERR')
    self.assertIn('Invalid format for t', str(context.exception))

  def test_unknown_mode(self):
    meqmsge.meqmsge('x', 'tf', 'T', 1.0, 1, 0, 'Unknown mode', 'UNK01')
    self.mock_logger.error.assert_called_once()
    args, _ = self.mock_logger.error.call_args
    self.assertEqual(args[0], 'Unknown meqmsge mode: %s - %s - %s')
    self.assertEqual(args[1], 'x')
    self.assertEqual(args[2], 'tf:UNK01')
    self.assertIn('Unknown mode', args[3])


if __name__ == '__main__':
  unittest.main()
