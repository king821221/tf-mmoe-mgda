# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for Min Norm Solvers
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np

from min_norm_solvers import MinNormSolver
from util import Constants

Constants.TF_PRINT_FLAG = 0

class TestFindMinNormElement(tf.test.TestCase):
  """Tests the find_min_norm_element function"""

  def test_2_tasks(self):
    vec_0=[tf.convert_to_tensor([0.001, -0.52, 0.2]),
           tf.convert_to_tensor([-0.001, 1.52, -0.02])]
    vec_1=[tf.convert_to_tensor([0.00, 0.00, 0.0]),
           tf.convert_to_tensor([0.000, -1.00, 2.00])]
    vecs=[vec_0, vec_1]

    solv_vec, nd = MinNormSolver.find_min_norm_element(vecs)

    with self.test_session() as sess:
      solv_vec_, nd_ = sess.run([solv_vec, nd])

    np.testing.assert_equal(len(solv_vec_), 2)
    np.testing.assert_(np.abs(nd_ - .993595) < 1e-5)
    np.testing.assert_(np.abs(solv_vec_[0] - .610732) < 1e-5)
    np.testing.assert_(np.abs(solv_vec_[1] - .389267) < 1e-5)

  def test_3_tasks(self):
      vec_0 = [tf.convert_to_tensor([0.001, -0.52, 0.2]), tf.convert_to_tensor([-0.001, 1.52, -0.02])]
      vec_1 = [tf.convert_to_tensor([0.00, 0.00, 0.00]), tf.convert_to_tensor([0.000, -1.00, 2.00])]
      vec_2 = [tf.convert_to_tensor([1.00, -200.1, 100.01]),
               tf.convert_to_tensor([1000.000, 12.0, -200.00])]
      vec_0 = [tf.cast(ve, dtype=tf.float64) for ve in vec_0]
      vec_1 = [tf.cast(ve, dtype=tf.float64) for ve in vec_1]
      vec_2 = [tf.cast(ve, dtype=tf.float64) for ve in vec_2]
      vecs = [vec_0, vec_1, vec_2]

      solv_vec, nd = MinNormSolver.find_min_norm_element(vecs)

      with self.test_session() as sess:
          solv_vec_, nd_ = sess.run([solv_vec, nd])

      np.testing.assert_equal(len(solv_vec_), 3)
      np.testing.assert_(np.abs(nd_ - .98862) < 1e-5)
      np.testing.assert_(np.abs(solv_vec_[0] - .60723) < 1e-5)
      np.testing.assert_(np.abs(solv_vec_[1] - .39269) < 1e-5)
      np.testing.assert_(np.abs(solv_vec_[2] - .00006838) < 1e-8)


if __name__ == "__main__":
  tf.test.main()
