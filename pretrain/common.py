# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================
"""Provides flags that are common to scripts.

Common flags from train/eval/vis/export_model.py are collected in this script.
"""
import collections
import copy

import tensorflow as tf

flags = tf.app.flags


# Model dependent flags.

flags.DEFINE_string('model_variant', 'xception_65', 'DeepLab model variant.')

# Defaults to None. Set multi_grid = [1, 2, 4] when using provided
# 'resnet_v1_{50,101}_beta' checkpoints.
flags.DEFINE_multi_integer('multi_grid', [1, 2, 4],
                           'Employ a hierarchy of atrous rates for ResNet.')

flags.DEFINE_float('depth_multiplier', 1.0,
                   'Multiplier for the depth (number of channels) for all '
                   'convolution ops used in MobileNet.')

FLAGS = flags.FLAGS

# Constants
LABELS_CLASS = 'labels_class'
IMAGE = 'image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
LABEL = 'label'
ORIGINAL_IMAGE = 'original_image'

# Test set name.
TEST_SET = 'test'


class ModelOptions(
    collections.namedtuple('ModelOptions', [
        'output_stride',
        'multi_grid',
        'model_variant',
        'depth_multiplier',
    ])):
  """Immutable class to hold model options."""

  __slots__ = ()

  def __new__(cls,
              output_stride=8):
    """Constructor to set default values.

    Args:
      outputs_to_num_classes: A dictionary from output type to the number of
        classes. For example, for the task of semantic segmentation with 21
        semantic classes, we would have outputs_to_num_classes['semantic'] = 21.
      crop_size: A tuple [crop_height, crop_width].
      atrous_rates: A list of atrous convolution rates for ASPP.
      output_stride: The ratio of input to output spatial resolution.

    Returns:
      A new ModelOptions instance.
    """
    return super(ModelOptions, cls).__new__(
        cls, output_stride, FLAGS.multi_grid, FLAGS.model_variant, FLAGS.depth_multiplier)

  def __deepcopy__(self, memo):
    return ModelOptions(copy.deepcopy(self.output_stride)
