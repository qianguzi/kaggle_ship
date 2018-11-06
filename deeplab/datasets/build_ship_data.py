"""Converts Ship data to TFRecord file format with Example protos.

This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
import cv2
import math
import os.path
import sys
import random
import build_data
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset_folder',
                           '/media/jun/data/ship',
                           'Folder containing images.')

tf.app.flags.DEFINE_string('train_folder',
                           '/media/jun/data/ship/train_v2',
                           'Folder containing images.')

tf.app.flags.DEFINE_string('seg_folder',
                           '/media/jun/data/ship/seg_v2',
                           'Folder containing images.')

tf.app.flags.DEFINE_string(
    'output_dir',
    './tfrecord',
    'Path to save converted SSTable of TensorFlow examples.')

tf.app.flags.DEFINE_bool('new_split', True,
                  'Apply new split_term or not.')

_NUM_SHARDS = 4

def rle_to_mask(rle_list, SHAPE):
    tmp_flat = np.zeros(SHAPE[0]*SHAPE[1])
    if len(rle_list) == 1:
        mask = np.reshape(tmp_flat, SHAPE).T
    else:
        strt = rle_list[::2]
        length = rle_list[1::2]
        for i,v in zip(strt,length):
            tmp_flat[(int(i)-1):(int(i)-1)+int(v)] = 1.0
        mask = np.reshape(tmp_flat, SHAPE).T
    return mask

def calc_area_for_rle(rle_str):
    rle_list = [int(x) if x.isdigit() else x for x in str(rle_str).split()]
    if len(rle_list) == 1:
        return 0
    else:
        area = np.sum(rle_list[1::2])
        return area

def write_tfexample(image_name, image_reader, label_reader, train_df, tfrecord_writer):
  # Read the image.
  image_filename = os.path.join(
      FLAGS.train_folder, image_name)
  image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
  height, width = image_reader.read_image_dims(image_data)
  # Read the semantic segmentation annotation.
  mask_list = train_df['EncodedPixels'][train_df['ImageId'] == image_name].tolist()
  seg_mask = np.zeros((768, 768, 1))
  for item in mask_list:
    rle_list = str(item).split()
    tmp_mask = rle_to_mask(rle_list, (768, 768))
    seg_mask[:,:,0] += tmp_mask
  
  seg_filename = os.path.join(FLAGS.seg_folder, image_name[:-4]+'.png')
  cv2.imwrite(seg_filename, seg_mask)
  seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
  seg_height, seg_width = label_reader.read_image_dims(seg_data)

  if height != seg_height or width != seg_width:
    raise RuntimeError('Shape mismatched between image and label.')
  # Convert to tf example.
  example = build_data.image_seg_to_tfexample(
      image_data, image_name[:-4], height, width, seg_data)
  tfrecord_writer.write(example.SerializeToString())

def _convert_train_dataset(train_isship_list, train_nanship_list, train_df):
  """Converts the train dataset to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  dataset = 'train'
  sys.stdout.write('Processing ' + dataset)

  min_num = min(len(train_isship_list),len(train_nanship_list))
  #min_num = 100
  print('Number of train samples: ', 2*min_num)
  train_isship_list = random.sample(train_isship_list, min_num)
  train_nanship_list = random.sample(train_nanship_list, min_num)
  num_per_shard = int(math.ceil(min_num / float(_NUM_SHARDS)))

  image_reader = build_data.ImageReader('jpg', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)
  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        FLAGS.output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, min_num)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting pair of images %d/%d shard %d' % (
            i + 1, min_num, shard_id))
        sys.stdout.flush()
        write_tfexample(train_isship_list[i], image_reader, label_reader, train_df, tfrecord_writer)
        write_tfexample(train_nanship_list[i], image_reader, label_reader, train_df, tfrecord_writer)
        
    sys.stdout.write('\n')
    sys.stdout.flush()

def _convert_val_dataset(val_list, train_df):
  """Converts the train dataset to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  dataset = 'val'
  sys.stdout.write('Processing ' + dataset)

  num_val = len(val_list)
  print('Number of val samples: ', num_val)
  num_per_shard = int(math.ceil(num_val / float(_NUM_SHARDS)))

  image_reader = build_data.ImageReader('jpg', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)
  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        FLAGS.output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_val)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, num_val, shard_id))
        sys.stdout.flush()
        write_tfexample(val_list[i], image_reader, label_reader, train_df, tfrecord_writer)
        
    sys.stdout.write('\n')
    sys.stdout.flush()

def main(unused_argv):
  train_df = pd.read_csv(os.path.join(FLAGS.dataset_folder, 'train_ship_segmentations_v2.csv'))
  train_df = train_df[train_df['ImageId'] != '6384c3e78.jpg']

  def area_isnull(x):
    if x == x:
        return 0
    else:
        return 1

  train_df['isnan'] = train_df['EncodedPixels'].apply(area_isnull)
  train_df = train_df.sort_values('isnan', ascending=False)
  train_df = train_df.iloc[130000:]

  train_df['area'] = train_df['EncodedPixels'].apply(calc_area_for_rle)
  train_gp = train_df.groupby('ImageId').sum()
  train_gp = train_gp.reset_index()

  def calc_class(area):
    area = area / (768*768)
    if area == 0:
        return 0
    elif area < 0.005:
        return 1
    elif area < 0.015:
        return 2
    elif area < 0.025:
        return 3
    elif area < 0.035:
        return 4
    elif area < 0.045:
        return 5
    else:
        return 6
  train_gp['class'] = train_gp['area'].apply(calc_class)
  train, val = train_test_split(train_gp, test_size=0.01, stratify=train_gp['class'].tolist())
  if not FLAGS.new_split:
    train_lotship_list = train['ImageId'][train['class']>1].tolist()
    train_nanship_list = train['ImageId'][train['class']==0].tolist()
    train_nanship_list = random.sample(train_nanship_list, len(train_lotship_list))
    train_fewship_list = train['ImageId'][train['class']==1].tolist()
    split_num = int(len(train_fewship_list)/2)

    train_isship_list = train_lotship_list + train_fewship_list[:split_num]
    train_nanship_list = train_fewship_list[split_num:] + train_nanship_list

    val_list = val['ImageId'].tolist()
    _convert_train_dataset(train_isship_list, train_nanship_list, train_df)
    _convert_val_dataset(val_list, train_df)
  else:
    train_lotship_list = train['ImageId'][train['class']>1].tolist()
    train_fewship_list = train['ImageId'][train['class']<=1].tolist()
    min_num = min(len(train_lotship_list), len(train_fewship_list))
    random.shuffle(train_lotship_list)
    random.shuffle(train_fewship_list)
    train_isship_list = train_lotship_list[:min_num]
    train_nanship_list = train_fewship_list[:min_num]

    other_list = random.sample(train_lotship_list[min_num:] + train_fewship_list[min_num:], 2000)
    val_list = val['ImageId'].tolist() + other_list
    _convert_val_dataset(val_list, train_df)
    _convert_train_dataset(train_isship_list, train_nanship_list, train_df)


if __name__ == '__main__':
  tf.app.run()
