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
                           '/media/deeplearning/f3cff4c9-1ab9-47f0-8b82-231dedcbd61b/ship',
                           'Folder containing images.')

tf.app.flags.DEFINE_string('train_folder',
                           '/media/deeplearning/f3cff4c9-1ab9-47f0-8b82-231dedcbd61b/ship/train_v2',
                           'Folder containing images.')

tf.app.flags.DEFINE_string('seg_folder',
                           '/media/deeplearning/f3cff4c9-1ab9-47f0-8b82-231dedcbd61b/ship/seg_v2',
                           'Folder containing images.')

tf.app.flags.DEFINE_string(
    'output_dir',
    './tfrecord/deeplab',
    'Path to save converted SSTable of TensorFlow examples.')

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

def aug_write_tfexample(image_name, image_reader, label_reader, train_df, tfrecord_writer):
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

  img = cv2.imread(image_filename)
  M = cv2.getRotationMatrix2D((384,384), 180, 1.0)
  N = cv2.getRotationMatrix2D((384,384), 90, 1.0)
  rotate_imga = cv2.warpAffine(img, M, (768, 768))
  rotate_sega = cv2.warpAffine(seg_mask, M, (768, 768))
  rotate_imgb = cv2.warpAffine(img, N, (768, 768))
  rotate_segb = cv2.warpAffine(seg_mask, N, (768, 768))
  flip_img = cv2.flip(img, 0)
  flip_seg = cv2.flip(seg_mask, 0)

  imga_filename = os.path.join(
      FLAGS.train_folder, image_name[:-4]+'_a.jpg')
  imgb_filename = os.path.join(
      FLAGS.train_folder, image_name[:-4]+'_b.jpg')
  imgf_filename = os.path.join(
      FLAGS.train_folder, image_name[:-4]+'_f.jpg')
  cv2.imwrite(imga_filename, rotate_imga)
  cv2.imwrite(imgb_filename, rotate_imgb)
  cv2.imwrite(imgf_filename, flip_img)
  imga_data = tf.gfile.FastGFile(imga_filename, 'rb').read()
  imgb_data = tf.gfile.FastGFile(imgb_filename, 'rb').read()
  imgf_data = tf.gfile.FastGFile(imgf_filename, 'rb').read()

  sega_filename = os.path.join(
      FLAGS.seg_folder, image_name[:-4]+'_a.png')
  segb_filename = os.path.join(
      FLAGS.seg_folder, image_name[:-4]+'_b.png')
  segf_filename = os.path.join(
      FLAGS.seg_folder, image_name[:-4]+'_f.png')
  cv2.imwrite(sega_filename, rotate_sega)
  cv2.imwrite(segb_filename, rotate_segb)
  cv2.imwrite(segf_filename, flip_seg)
  sega_data = tf.gfile.FastGFile(sega_filename, 'rb').read()
  segb_data = tf.gfile.FastGFile(segb_filename, 'rb').read()
  segf_data = tf.gfile.FastGFile(segf_filename, 'rb').read()
  
  # Convert to tf example.
  example = build_data.image_seg_to_tfexample(
      image_data, image_name[:-4], height, width, seg_data)
  tfrecord_writer.write(example.SerializeToString())
  example_a = build_data.image_seg_to_tfexample(
      imga_data, image_name[:-4]+'_a', height, width, sega_data)
  tfrecord_writer.write(example_a.SerializeToString())
  example_b = build_data.image_seg_to_tfexample(
      imgb_data, image_name[:-4]+'_b', height, width, segb_data)
  tfrecord_writer.write(example_b.SerializeToString())
  example_f = build_data.image_seg_to_tfexample(
      imgf_data, image_name[:-4]+'_f', height, width, segf_data)
  tfrecord_writer.write(example_f.SerializeToString())


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

def convert_dataset(dataset, is_ship_list, nan_ship_list, train_df):
  
  sys.stdout.write('Processing ' + dataset + '\n')
  sys.stdout.flush()
  min_num = min(len(is_ship_list),len(nan_ship_list))
  #min_num = 100
  sys.stdout.write('Number of train samples: %d\n' % (2*min_num))
  sys.stdout.flush()
  is_ship_list = random.sample(is_ship_list, min_num)
  nan_ship_list = random.sample(nan_ship_list, min_num)
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
        if dataset == 'train':
            aug_write_tfexample(is_ship_list[i], image_reader, label_reader, train_df, tfrecord_writer)
        elif dataset == 'val':
            write_tfexample(is_ship_list[i], image_reader, label_reader, train_df, tfrecord_writer)
        else:
            raise RuntimeError('Dataset must be val/train.')
        write_tfexample(nan_ship_list[i], image_reader, label_reader, train_df, tfrecord_writer)
        
    sys.stdout.write('\n')
    sys.stdout.flush()


def split_term(train):
  train_lotship_list = train['ImageId'][train['class']>2].tolist()
  train_fewship_list = train['ImageId'][train['class']==1].tolist()
  train_fewship_list = random.sample(train_fewship_list, len(train_lotship_list))
  train_anyship_list = train['ImageId'][train['class']==2].tolist()
  split_num = int(len(train_anyship_list)/2)

  train_lotship_list = train_lotship_list + train_anyship_list[:split_num]
  train_fewship_list = train_anyship_list[split_num:] + train_fewship_list
  return train_lotship_list, train_fewship_list


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
  train_df = train_df.iloc[100000:]

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

  val_nanship_list = val['ImageId'][val['class']==0].tolist()
  val_isship_list = val['ImageId'][val['class']>0].tolist()

  #val_lotship_list, val_fewship_list = split_term(val)
  convert_dataset('val', val_isship_list, val_nanship_list, train_df)

  train_lotship_list, train_fewship_list = split_term(train)
  convert_dataset('train', train_lotship_list, train_fewship_list, train_df)


if __name__ == '__main__':
  tf.app.run()
