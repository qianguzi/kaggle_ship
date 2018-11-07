import collections
import os.path
import tensorflow as tf
import tensorflow.contrib.slim as slim
from deeplab import common
from deeplab import input_preprocess

dataset = slim.dataset
tfexample_decoder = slim.tfexample_decoder
dataset_data_provider = slim.dataset_data_provider


_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'label': ('A class_idx of matches image.'
                     'Its value range from 0 (background) to num_classes.'),
}

# Named tuple to describe the dataset properties.
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    ['splits_to_sizes',   # Splits of the dataset into training, val, and test.
     'num_classes',   
    ]
)

# These number (i.e., 'train'/'test') seems to have to be hard coded
# You are required to figure it out for your training/testing example.
_SHIP_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 7952,  # num of samples in images/training
        'val': 80,  # num of samples in images/validation
    },
    num_classes=2,
)

_DATASETS_INFORMATION = {
    'ship': _SHIP_INFORMATION,
}

# Default file pattern of TFRecord of TensorFlow Example.
_FILE_PATTERN = '%s-*'


def get_dataset(dataset_name, split_name, dataset_dir):
  """Gets an instance of slim Dataset.

  Args:
    dataset_name: Dataset name.
    split_name: A train/val Split name.
    dataset_dir: The directory of the dataset sources.

  Returns:
    An instance of slim Dataset.

  Raises:
    ValueError: if the dataset_name or split_name is not recognized.
  """
  if dataset_name not in _DATASETS_INFORMATION:
    raise ValueError('The specified dataset is not supported yet.')

  splits_to_sizes = _DATASETS_INFORMATION[dataset_name].splits_to_sizes

  if split_name not in splits_to_sizes:
    raise ValueError('data split name %s not recognized' % split_name)

  # Prepare the variables for different datasets.
  num_classes = _DATASETS_INFORMATION[dataset_name].num_classes

  file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Specify how the TF-Examples are decoded.
  keys_to_features = {
      'image/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/filename': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature(
          (), tf.string, default_value='jpeg'),
      'image/height': tf.FixedLenFeature(
          (), tf.int64, default_value=0),
      'image/width': tf.FixedLenFeature(
          (), tf.int64, default_value=0),
      'image/label': tf.FixedLenFeature(
          (), tf.int64, default_value=0),
  }
  items_to_handlers = {
      'image': tfexample_decoder.Image(
          image_key='image/encoded',
          format_key='image/format',
          channels=3),
      'image_name': tfexample_decoder.Tensor('image/filename'),
      'height': tfexample_decoder.Tensor('image/height'),
      'width': tfexample_decoder.Tensor('image/width'),
      'labels_class': tfexample_decoder.Tensor('image/label'),
  }

  decoder = tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  return dataset.Dataset(
      data_sources=file_pattern,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=splits_to_sizes[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=num_classes,
      name=dataset_name,
      multi_label=True)

def _get_data(data_provider, dataset_split):
  """Gets data from data provider.

  Args:
    data_provider: An object of slim.data_provider.
    dataset_split: Dataset split.

  Returns:
    image: Image Tensor.
    label: Label Tensor.
    image_name: Image name.
    height: Image height.
    width: Image width.

  Raises:
    ValueError: Failed to find label.
  """
  if common.LABELS_CLASS not in data_provider.list_items():
    raise ValueError('Failed to find labels.')

  image, height, width = data_provider.get(
      [common.IMAGE, common.HEIGHT, common.WIDTH])

  # Some datasets do not contain image_name.
  if common.IMAGE_NAME in data_provider.list_items():
    image_name, = data_provider.get([common.IMAGE_NAME])
  else:
    image_name = tf.constant('')

  label = None
  if dataset_split != common.TEST_SET:
    label, = data_provider.get([common.LABELS_CLASS])

  return image, label, image_name, height, width

def get(dataset,
        resized_image_size,
        batch_size,
        num_readers=1,
        num_threads=1,
        dataset_split=None,
        is_training=True,
        model_variant=None):
  """Gets the dataset split for semantic segmentation.

  This functions gets the dataset split for semantic segmentation. In
  particular, it is a wrapper of (1) dataset_data_provider which returns the raw
  dataset split, (2) input_preprcess which preprocess the raw data, and (3) the
  Tensorflow operation of batching the preprocessed data. Then, the output could
  be directly used by training, evaluation or visualization.

  Args:
    dataset: An instance of slim Dataset.
    batch_size: Batch size.
    num_readers: Number of readers for data provider.
    num_threads: Number of threads for batching data.
    dataset_split: Dataset split.
    is_training: Is training or not.
    model_variant: Model variant (string) for choosing how to mean-subtract the
      images. See feature_extractor.network_map for supported model variants.

  Returns:
    A dictionary of batched Tensors for semantic segmentation.

  Raises:
    ValueError: dataset_split is None, failed to find labels, or label shape
      is not valid.
  """
  if dataset_split is None:
    raise ValueError('Unknown dataset split.')
  if model_variant is None:
    tf.logging.warning('Please specify a model_variant. See '
                       'feature_extractor.network_map for supported model '
                       'variants.')

  data_provider = dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=num_readers,
      num_epochs=None if is_training else 1,
      shuffle=False)
  image, label, image_name, height, width = _get_data(data_provider,
                                                      dataset_split)
  image = tf.image.resize_images(
        image, resized_image_size, align_corners=True)
  sample = {
      common.IMAGE: image,
      common.IMAGE_NAME: image_name,
      common.HEIGHT: height,
      common.WIDTH: width
  }
  if label is not None:
    sample[common.LABEL] = label

  return tf.train.batch(
      sample,
      batch_size=batch_size,
      num_threads=num_threads,
      capacity=32 * batch_size,
      allow_smaller_final_batch=not is_training,
      dynamic_pad=True)
