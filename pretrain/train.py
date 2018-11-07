
import sys
sys.path.append('./')
import six
import functools
import tensorflow as tf
import tensorflow.contrib.slim as slim
from deeplab.core import feature_extractor  
from deeplab.utils import train_utils
from pretrain import common
from pretrain import model
from pretrain.datasets import input_generator
from slim_utils.deployment import model_deploy

prefetch_queue = slim.prefetch_queue

flags = tf.app.flags

FLAGS = flags.FLAGS


# Settings for multi-GPUs/multi-replicas training.

flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy.')

flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')

flags.DEFINE_integer('num_replicas', 1, 'Number of worker replicas.')

flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')

flags.DEFINE_integer('num_ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then '
                     'the parameters are handled locally by the worker.')

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

flags.DEFINE_integer('task', 0, 'The task ID.')

# Settings for logging.

flags.DEFINE_string('train_logdir', './train_log/pretrain',
                    'Where the checkpoint and logs are stored.')

flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')

flags.DEFINE_integer('save_interval_secs', 1200,
                     'How often, in seconds, we save the model to disk.')

flags.DEFINE_integer('save_summaries_secs', 600,
                     'How often, in seconds, we compute the summaries.')

flags.DEFINE_boolean('save_summaries_images', False,
                     'Save sample inputs, labels, and semantic predictions as '
                     'images to summary.')

# Settings for training strategy.

flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')

# Use 0.007 when training on PASCAL augmented training set, train_aug. When
# fine-tuning on PASCAL trainval set, use learning rate=0.0001.
flags.DEFINE_float('base_learning_rate', .007,
                   'The base learning rate for model training.')

flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   'The rate to decay the base learning rate.')

flags.DEFINE_integer('learning_rate_decay_step', 2000,
                     'Decay the base learning rate at a fixed step.')

flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')

flags.DEFINE_integer('training_number_of_steps', 100000,
                     'The number of steps used for training')

flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

# When fine_tune_batch_norm=True, use at least batch size larger than 12
# (batch size more than 16 is better). Otherwise, one could use smaller batch
# size and set fine_tune_batch_norm=False.
flags.DEFINE_integer('train_batch_size', 16,
                     'The number of images in each batch during training.')

# For weight_decay, use 0.00004 for MobileNet-V2 or Xcpetion model variants.
# Use 0.0001 for ResNet model variants.
flags.DEFINE_float('weight_decay', 0.00004,
                   'The value of the weight decay for training.')

flags.DEFINE_multi_integer('resized_image_size', [513, 513],
                           'Image crop size [height, width] during training.')

flags.DEFINE_float('last_layer_gradient_multiplier', 1.0,
                   'The gradient multiplier for last layers, which is used to '
                   'boost the gradient of last layers if the value > 1.')

# Settings for fine-tuning the network.

flags.DEFINE_string('tf_initial_checkpoint', './model_zoo/xception_65_coco_pretrained/model.ckpt',
                    'The initial checkpoint in tensorflow format.')

# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean('initialize_last_layer', True,
                     'Initialize the last layer.')

flags.DEFINE_integer('slow_start_step', 0,
                     'Training model with small learning rate for few steps.')

flags.DEFINE_float('slow_start_learning_rate', 1e-4,
                   'Learning rate employed during slow start.')

# Set to True if one wants to fine-tune the batch norm parameters in DeepLabv3.
# Set to False and use small batch size to save GPU memory.
flags.DEFINE_boolean('fine_tune_batch_norm', True,
                     'Fine tune the batch norm parameters or not.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Dataset settings.
flags.DEFINE_string('dataset', 'ship',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')

flags.DEFINE_string('dataset_dir', './tfrecord/pretrain', 'Where the dataset reside.')

def _build_model(inputs_queue, num_classes):

  samples = inputs_queue.dequeue()

  # Add name to input and label nodes so we can add to summary.
  samples[common.IMAGE] = tf.identity(
      samples[common.IMAGE], name=common.IMAGE)
  samples[common.LABEL] = tf.identity(
      samples[common.LABEL], name=common.LABEL)

  model_options = common.ModelOptions(
      output_stride=FLAGS.output_stride)
  logits, end_points = model.get_logits(
      samples[common.IMAGE],
      model_options=model_options,
      num_classes=num_classes,
      weight_decay=FLAGS.weight_decay,
      is_training=True,
      fine_tune_batch_norm=FLAGS.fine_tune_batch_norm)
  labels = slim.one_hot_encoding(samples[common.LABEL], num_classes)
  tf.losses.softmax_cross_entropy(labels, logits)
  return logits

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  # Set up deployment (i.e., multi-GPUs and/or multi-replicas).
  config = model_deploy.DeploymentConfig(
      num_clones=FLAGS.num_clones,
      clone_on_cpu=FLAGS.clone_on_cpu,
      replica_id=FLAGS.task,
      num_replicas=FLAGS.num_replicas,
      num_ps_tasks=FLAGS.num_ps_tasks)

  # Split the batch across GPUs.
  assert FLAGS.train_batch_size % config.num_clones == 0, (
      'Training batch size not divisble by number of clones (GPUs).')

  clone_batch_size = FLAGS.train_batch_size // config.num_clones

  # Get dataset-dependent information.
  dataset = input_generator.get_dataset(
      FLAGS.dataset, FLAGS.train_split, dataset_dir=FLAGS.dataset_dir)

  tf.gfile.MakeDirs(FLAGS.train_logdir)
  tf.logging.info('Training on %s set', FLAGS.train_split)

  with tf.Graph().as_default() as graph:
    with tf.device(config.inputs_device()):
      samples = input_generator.get(
          dataset,
          FLAGS.resized_image_size,
          clone_batch_size,
          dataset_split=FLAGS.train_split,
          is_training=True,
          model_variant=FLAGS.model_variant)
      inputs_queue = prefetch_queue.prefetch_queue(
          samples, capacity=128 * config.num_clones)

    # Create the global step on the device storing the variables.
    with tf.device(config.variables_device()):
      global_step = tf.train.get_or_create_global_step()

      # Define the model and create clones.
      model_fn = _build_model
      model_args = (inputs_queue, dataset.num_classes)
      clones = model_deploy.create_clones(config, model_fn, args=model_args)

      # Gather update_ops from the first clone. These contain, for example,
      # the updates for the batch_norm variables created by model_fn.
      first_clone_scope = config.clone_scope(0)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Add summaries for model variables.
    for model_var in slim.get_model_variables():
      summaries.add(tf.summary.histogram(model_var.op.name, model_var))

    # Add summaries for images, labels, semantic predictions
    if FLAGS.save_summaries_images:
      summary_image = graph.get_tensor_by_name(
          ('%s/%s:0' % (first_clone_scope, common.IMAGE)).strip('/'))
      summaries.add(
          tf.summary.image('samples/%s' % common.IMAGE, summary_image))

      first_clone_label = graph.get_tensor_by_name(
          ('%s/%s:0' % (first_clone_scope, common.LABEL)).strip('/'))
      # Scale up summary image pixel values for better visualization.
      pixel_scaling = max(1, 255 // dataset.num_classes)
      summary_label = tf.cast(first_clone_label * pixel_scaling, tf.uint8)
      summaries.add(
          tf.summary.image('samples/%s' % common.LABEL, summary_label))

      first_clone_output = graph.get_tensor_by_name(
          ('%s/%s:0' % (first_clone_scope, common.OUTPUT_TYPE)).strip('/'))
      predictions = tf.expand_dims(tf.argmax(first_clone_output, 3), -1)

      summary_predictions = tf.cast(predictions * pixel_scaling, tf.uint8)
      summaries.add(
          tf.summary.image(
              'samples/%s' % common.OUTPUT_TYPE, summary_predictions))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

    # Build the optimizer based on the device specification.
    with tf.device(config.optimizer_device()):
      learning_rate = train_utils.get_model_learning_rate(
          FLAGS.learning_policy, FLAGS.base_learning_rate,
          FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
          FLAGS.training_number_of_steps, FLAGS.learning_power,
          FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
      optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
      summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    startup_delay_steps = FLAGS.task * FLAGS.startup_delay_steps
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))

    with tf.device(config.variables_device()):
      total_loss, grads_and_vars = model_deploy.optimize_clones(
          clones, optimizer)
      total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
      summaries.add(tf.summary.scalar('total_loss', total_loss))

      # Modify the gradients for biases and last layer variables.
      grad_mult = train_utils.get_model_gradient_multipliers(
          ['Logits'], FLAGS.last_layer_gradient_multiplier)
      if grad_mult:
        grads_and_vars = slim.learning.multiply_gradients(
            grads_and_vars, grad_mult)

      # Create gradient update op.
      grad_updates = optimizer.apply_gradients(
          grads_and_vars, global_step=global_step)
      update_ops.append(grad_updates)
      update_op = tf.group(*update_ops)
      with tf.control_dependencies([update_op]):
        train_tensor = tf.identity(total_loss, name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(
        tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries))

    # Soft placement allows placing on CPU ops without GPU implementation.
    session_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)

    # Start the training.
    slim.learning.train(
        train_tensor,
        logdir=FLAGS.train_logdir,
        log_every_n_steps=FLAGS.log_steps,
        master=FLAGS.master,
        number_of_steps=FLAGS.training_number_of_steps,
        is_chief=(FLAGS.task == 0),
        session_config=session_config,
        startup_delay_steps=startup_delay_steps,
        init_fn=train_utils.get_model_init_fn(
            FLAGS.train_logdir,
            FLAGS.tf_initial_checkpoint,
            FLAGS.initialize_last_layer,
            last_layers,
            ignore_missing_vars=True),
        summary_op=summary_op,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs)


if __name__ == '__main__':
  flags.mark_flag_as_required('train_logdir')
  flags.mark_flag_as_required('tf_initial_checkpoint')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()