
import sys
sys.path.append('./')
import math
import six
import tensorflow as tf
import tensorflow.contrib.slim as slim
from pretrain import common
from pretrain import model
from pretrain.datasets import input_generator

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('eval_logdir', './val_log/pretrain', 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', './train_log/pretrain', 'Directory of model checkpoints.')

# Settings for evaluating the model.

flags.DEFINE_integer('eval_batch_size', 4,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_multi_integer('resized_image_size', [256, 256],
                           'Image crop size [height, width] for evaluation.')

flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     'How often (in seconds) to run evaluation.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Dataset settings.

flags.DEFINE_string('dataset', 'ship',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset used for evaluation')

flags.DEFINE_string('dataset_dir', './tfrecord/pretrain', 'Where the dataset reside.')

flags.DEFINE_integer('max_number_of_evaluations', 5,
                     'Maximum number of eval iterations. Will loop '
                     'indefinitely upon nonpositive values.')


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  # Get dataset-dependent information.
  dataset = input_generator.get_dataset(
      FLAGS.dataset, FLAGS.eval_split, dataset_dir=FLAGS.dataset_dir)

  tf.gfile.MakeDirs(FLAGS.eval_logdir)
  tf.logging.info('Evaluating on %s set', FLAGS.eval_split)

  with tf.Graph().as_default():
    samples = input_generator.get(
        dataset,
        FLAGS.resized_image_size,
        FLAGS.eval_batch_size,
        dataset_split=FLAGS.eval_split,
        is_training=False,
        model_variant=FLAGS.model_variant)

    model_options = common.ModelOptions(
        output_stride=FLAGS.output_stride)

    logits, end_points = model.get_logits(
      samples[common.IMAGE],
      model_options=model_options,
      num_classes=dataset.num_classes,
      is_training=False,
      fine_tune_batch_norm=False)
    
    predictions = end_points['Predictions']
    predictions = tf.argmax(predictions, axis=1)
    predictions = tf.reshape(predictions, shape=[-1])
    labels = tf.reshape(samples[common.LABEL], shape=[-1])

    # Define the evaluation metric.
    metric_map = {}
    metric_map['accuracy'] = tf.metrics.accuracy(labels, predictions)

    metrics_to_values, metrics_to_updates = (
        tf.contrib.metrics.aggregate_metric_map(metric_map))

    for metric_name, metric_value in six.iteritems(metrics_to_values):
      slim.summaries.add_scalar_summary(
          metric_value, metric_name, print_summary=True)

    num_batches = int(
        math.ceil(dataset.num_samples / float(FLAGS.eval_batch_size)))

    tf.logging.info('Eval num images %d', dataset.num_samples)
    tf.logging.info('Eval batch size %d and num batch %d',
                    FLAGS.eval_batch_size, num_batches)

    num_eval_iters = None
    if FLAGS.max_number_of_evaluations > 0:
      num_eval_iters = FLAGS.max_number_of_evaluations
    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_logdir,
        num_evals=num_batches,
        eval_op=list(metrics_to_updates.values()),
        max_number_of_evaluations=num_eval_iters,
        eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  flags.mark_flag_as_required('eval_logdir')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()