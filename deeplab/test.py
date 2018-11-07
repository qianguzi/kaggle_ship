
import sys
sys.path.append('./')
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from time import time
from skimage.data import imread
from skimage.morphology import label

def multi_rle_encode(img, **kwargs):
    '''
    Encode connected regions as separated masks
    '''
    labels = label(img[0,:,:])
    return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]

  # ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def model_test():
  g = tf.Graph()
  with g.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.FastGFile('./model.pb', 'rb') as f:
        od_graph_def.ParseFromString(f.read())
        img_tensor, seg_pred= tf.import_graph_def(
                od_graph_def,
                return_elements=['ImageTensor:0', 'SemanticPredictions:0'])
    init_op = tf.global_variables_initializer()
    test_img_dir = '/media/deeplearning/f3cff4c9-1ab9-47f0-8b82-231dedcbd61b/ship/test_v2/'

    with tf.Session() as sess:
        sess.run(init_op)
        pred_rows = []
        start_time = time()
        for i, image_name in enumerate(os.listdir(test_img_dir)):
          single_start_time = time()
          test_img = imread(test_img_dir + image_name)
          test_img = np.expand_dims(test_img, 0)
          pred_mask = sess.run(seg_pred, {img_tensor: test_img})
          
          rles = multi_rle_encode(pred_mask)
          if len(rles)>0:
              for rle in rles:
                  pred_rows += [{'ImageId': image_name, 'EncodedPixels': rle}]
          else:
              pred_rows += [{'ImageId': image_name, 'EncodedPixels': None}]
          print('{0}[{1}]: {2} ship(s), time cost: {3}'.format(image_name, i+1, len(rles), time()-single_start_time))
        print('All time cost: %s' % (time()-start_time))          
        submission_df = pd.DataFrame(pred_rows)[['ImageId', 'EncodedPixels']]
        submission_df.to_csv('submission.csv', index=False)
        print('File submission.csv success saved.')

if __name__ == '__main__':
  model_test()