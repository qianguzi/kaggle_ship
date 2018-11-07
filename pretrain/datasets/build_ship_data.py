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

tf.app.flags.DEFINE_string(
    'output_dir',
    './tfrecord/pretrain',
    'Path to save converted SSTable of TensorFlow examples.')

_NUM_SHARDS = 2

def calc_area_for_rle(rle_str):
    rle_list = [int(x) if x.isdigit() else x for x in str(rle_str).split()]
    if len(rle_list) == 1:
        return 0
    else:
        area = np.sum(rle_list[1::2])
        return area

def write_tfexample(image_name, image_reader, train_df, tfrecord_writer, label):
  # Read the image.
  image_filename = os.path.join(
      FLAGS.train_folder, image_name)
  image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
  height, width = image_reader.read_image_dims(image_data)

  # Convert to tf example.
  example = build_data.image_seg_to_tfexample(
      image_data, image_name[:-4], height, width, label)
  tfrecord_writer.write(example.SerializeToString())

def _convert_dataset(dataset, train_isship_list, train_nanship_list, train_df):

  sys.stdout.write('Processing ' + dataset)

  min_num = min(len(train_isship_list),len(train_nanship_list))
  #min_num = 100
  print('Number of train samples: ', 2*min_num)
  train_isship_list = random.sample(train_isship_list, min_num)
  train_nanship_list = random.sample(train_nanship_list, min_num)
  num_per_shard = int(math.ceil(min_num / float(_NUM_SHARDS)))

  image_reader = build_data.ImageReader('jpg', channels=3)
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
        write_tfexample(train_isship_list[i], image_reader, train_df, tfrecord_writer, label=1)
        write_tfexample(train_nanship_list[i], image_reader, train_df, tfrecord_writer, label=0) 
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
    elif area >= 0.015:
        return 1
    else:
        return 2
  train_gp['class'] = train_gp['area'].apply(calc_class)
  train, val = train_test_split(train_gp, test_size=0.01, stratify=train_gp['class'].tolist())
  train_isship_list = train['ImageId'][train['class']==1].tolist()
  train_nanship_list = train['ImageId'][train['class']==0].tolist()
  train_nanship_list = random.sample(train_nanship_list, len(train_isship_list))

  val_isship_list = val['ImageId'][val['class']==1].tolist()
  val_nanship_list = val['ImageId'][val['class']==0].tolist()
  val_nanship_list = random.sample(val_nanship_list, len(val_isship_list))
  _convert_dataset('train',train_isship_list, train_nanship_list, train_df)
  _convert_dataset('val', val_isship_list, val_nanship_list, train_df)


if __name__ == '__main__':
  tf.app.run()
