# Abandoned file !
# Abandoned file !
# Abandoned file !
# Abandoned file !
# Abandoned file !
# ==============================================================================
import cv2
import math
import os, sys
import random
import pandas as pd
import numpy as np
import build_data
import build_ship_data
import tensorflow as tf
from sklearn.model_selection import train_test_split

dataset_folder = '/media/deeplearning/f3cff4c9-1ab9-47f0-8b82-231dedcbd61b/ship'
train_folder = '/media/deeplearning/f3cff4c9-1ab9-47f0-8b82-231dedcbd61b/ship/train_v2'
data_format = '.jpg'
seg_format = '.png'

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

def calc_class_a(area):
    area = area / (768*768)
    if area == 0:
        return 0
    elif area < 0.0005:
        return 1
    elif area < 0.001:
        return 2
    elif area < 0.002:
        return 3
    elif area < 0.005:
        return 4
    elif area < 0.025:
        return 5
    else:
        return 6

def calc_class_b(area):
    area = area / (768*768)
    if area == 0:
        return 0
    elif area < 0.0001:
        return 1
    elif area < 0.0002:
        return 2
    elif area < 0.001:
        return 3
    elif area < 0.015:
        return 4
    elif area < 0.025:
        return 5
    else:
        return 6

def img_grid_split(resize_size, crop_num, split_img_size=256):
    half_split_img_size = int(split_img_size / 2)
    dl = (resize_size-split_img_size) / (crop_num-1)
    l = np.arange(half_split_img_size, resize_size-half_split_img_size, dl, np.int32)
    l = np.append(l, resize_size-half_split_img_size)
    l = l[:crop_num]
    centers = np.meshgrid(l, l)
    centers = np.stack(centers, axis=2)
    centers = np.reshape(centers, [-1, 2])
    boxes = np.concatenate((centers - half_split_img_size, centers + half_split_img_size), 1)
    return boxes


def img_sort(resize_size, crop_num, image_name_list, train_df, class_idx, save_num=2):
    sys.stdout.write('Processing train' + class_idx[:-1] + '\n')
    sys.stdout.flush()
    names = []
    boxes = img_grid_split(resize_size, crop_num)
    num_images = len(image_name_list)
    for i, image_name in enumerate(image_name_list):
        sys.stdout.write('\r>> Preprocessing images %d/%d' % (
            i + 1, num_images))
        sys.stdout.flush()
        name = image_name[:-4]+class_idx
        image = cv2.imread(os.path.join(train_folder, image_name))
        image = cv2.resize(image, (resize_size, resize_size))
        mask_list = train_df['EncodedPixels'][train_df['ImageId'] == image_name].tolist()
        mask = np.zeros((768, 768, 1))
        for item in mask_list:
            rle_list = str(item).split()
            tmp_mask = rle_to_mask(rle_list, (768, 768))
            mask[:,:,0] += tmp_mask
        mask = cv2.resize(mask, (resize_size, resize_size))
        areas = []
        split_images = []
        split_masks = []
        for box in boxes:
            split_images.append(image[box[1]:box[3], box[0]:box[2]])
            split_mask = mask[box[1]:box[3], box[0]:box[2]]
            split_masks.append(split_mask)
            areas.append(np.sum(split_mask))
        sort_idx = np.argsort(areas)
        for idx in sort_idx[-save_num:]:
            if areas[idx] >= 64:
                names.append(name+str(idx))
                cv2.imwrite(dataset_folder+'/train/'+name+str(idx)+data_format, split_images[idx])
                cv2.imwrite(dataset_folder+'/seg/'+name+str(idx)+seg_format, split_masks[idx])
    sys.stdout.write('\n')
    sys.stdout.flush()
    return names


def convert_dataset(dataset, names, num_shared=4):
    sys.stdout.write('Processing ' + dataset)
    sys.stdout.write('\n')
    sys.stdout.flush()
    num = len(names)
    num_per_shard = int(math.ceil(num / float(num_shared)))

    image_reader = build_data.ImageReader('jpg', channels=3)
    label_reader = build_data.ImageReader('png', channels=1)
    for shard_id in range(num_shared):
        output_filename = os.path.join(
            dataset_folder, 'tfrecord',
            '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, num_shared))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num)
            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, num, shard_id))
                sys.stdout.flush()
                # Read the image.
                image_filename = os.path.join(
                    dataset_folder,'train', names[i]+data_format)
                image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
                height, width = image_reader.read_image_dims(image_data)
                # Read the semantic segmentation annotation.
  
                seg_filename = os.path.join(
                    dataset_folder,'seg', names[i]+seg_format)
                seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
                seg_height, seg_width = label_reader.read_image_dims(seg_data)

                if height != seg_height or width != seg_width:
                    raise RuntimeError('Shape mismatched between image and label.')
                # Convert to tf example.
                example = build_data.image_seg_to_tfexample(
                    image_data, names[i], height, width, seg_data)
                tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()

def main():
    train_df = pd.read_csv(os.path.join(dataset_folder, 'train_ship_segmentations_v2.csv'))
    train_df = train_df[train_df['ImageId'] != '6384c3e78.jpg']
    train_df = train_df[train_df['ImageId'] != '00021ddc3.jpg']

    def area_isnull(x):
        if x == x:
            return 0
        else:
            return 1

    train_df['isnan'] = train_df['EncodedPixels'].apply(area_isnull)
    train_isship = train_df[train_df['isnan']==0]
    train_nanship = train_df[train_df['isnan']==1]

    train_isship['area'] = train_isship['EncodedPixels'].apply(calc_area_for_rle)
    train_gp = train_isship.groupby('ImageId').sum()
    train_gp = train_gp.reset_index()
    train_gp['class'] = train_gp['area'].apply(calc_class_a)
    train, val = train_test_split(train_gp, test_size=0.05, stratify=train_gp['class'].tolist())

    val_isship = pd.DataFrame()
    for i in val['ImageId']:
        val_isship = pd.concat([val_isship, train_isship[train_isship['ImageId'] == i]], axis=0)
    train_isship.drop(val_isship.index)

    val_isship = val_isship.groupby('ImageId').sum()
    val_isship = val_isship.reset_index()
    val_isship_list = val_isship['ImageId'].tolist()
    val_nanship = train_nanship[:len(val_isship_list)]
    val_nanship_list = val_nanship['ImageId'].tolist()
    build_ship_data.convert_dataset('val', val_isship_list, val_nanship_list, train_df)

    train_isship['class'] = train_isship['area'].apply(calc_class_b)
    train_5 = train_isship[train_isship['class']>=4]
    train_gp5 = train_5.groupby('ImageId').sum()
    train_gp5 = train_gp5.reset_index()
    train_gp5_list = train_gp5['ImageId'].tolist()
    train_4 = train_isship[train_isship['class']>=3]
    train_gp4 = train_4.groupby('ImageId').sum()
    train_gp4 = train_gp4.reset_index()
    train_gp4_list = train_gp4['ImageId'].tolist()
    train_3 = train_isship[(train_isship['class']>=2) & (train_isship['class']<=5)]
    train_gp3 = train_3.groupby('ImageId').sum()
    train_gp3 = train_gp3.reset_index()
    train_gp3_list = train_gp3['ImageId'].tolist()
    train_2 = train_isship[(train_isship['class']>=1) & (train_isship['class']<=4)]
    train_gp2 = train_2.groupby('ImageId').sum()
    train_gp2 = train_gp2.reset_index()
    train_gp2_list = train_gp2['ImageId'].tolist()

    train_nanship_list = train_nanship[-10000:]['ImageId'].tolist()
    names_1 = []
    num_gp1 = len(train_nanship_list)
    sys.stdout.write('Processing train_1\n')
    sys.stdout.flush()
    for i, image_name in enumerate(train_nanship_list):
        sys.stdout.write('\r>> Preprocessing images %d/%d' % (
            i + 1, num_gp1))
        sys.stdout.flush()
        name = image_name[:-4] + '_1'
        names_1.append(name)
        image = cv2.imread(os.path.join(train_folder, image_name))
        image = cv2.resize(image, (256, 256))
        cv2.imwrite(dataset_folder+'/train/'+name+data_format, image)
        mask = np.zeros((256, 256, 1))
        cv2.imwrite(dataset_folder+'/seg/'+name+seg_format, mask)
    sys.stdout.write('\n')
    sys.stdout.flush()
    random.shuffle(names_1)
    convert_dataset('train-1', names_1, 1)

    names_5 = []
    num_gp5 = len(train_gp5_list)
    sys.stdout.write('Processing train_5\n')
    sys.stdout.flush()
    for i, image_name in enumerate(train_gp5_list):
        sys.stdout.write('\r>> Preprocessing images %d/%d' % (
            i + 1, num_gp5))
        sys.stdout.flush()
        name = image_name[:-4]+'_5'
        names_5.append(name)
        mask_list = train_df['EncodedPixels'][train_df['ImageId'] == image_name].tolist()
        seg_mask = np.zeros((768, 768, 1))
        for item in mask_list:
            rle_list = str(item).split()
            tmp_mask = rle_to_mask(rle_list, (768, 768))
            seg_mask[:,:,0] += tmp_mask
        mask = cv2.resize(seg_mask, (256, 256))
        cv2.imwrite(dataset_folder+'/seg/'+name+seg_format, mask)
        image = cv2.imread(os.path.join(train_folder, image_name))
        image = cv2.resize(image, (256, 256))
        cv2.imwrite(dataset_folder+'/train/'+name+data_format, image)
    sys.stdout.write('\n')
    sys.stdout.flush()

    names_4= img_sort(512, 3, train_gp4_list, train_df, '_4_', 1)
    names_3= img_sort(768, 5, train_gp3_list, train_df, '_3_')
    names_2= img_sort(1024, 7, train_gp2_list, train_df, '_2_')

    names = names_2+names_3+names_4
    random.shuffle(names)
    convert_dataset('train', names, 12)
    random.shuffle(names_5)
    convert_dataset('train-5', names_5, 4)
    

if __name__ == '__main__':
    main()