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
"""Utility functions for training."""

import six

import tensorflow as tf
import tensorflow_probability as tfp
from deeplab.core import preprocess_utils

slim = tf.contrib.slim


def add_softmax_cross_entropy_loss_for_each_scale(scales_to_logits,
                                                  labels,
                                                  num_classes,
                                                  ignore_label,
                                                  loss_weight=1.0,
                                                  upsample_logits=True,
                                                  scope=None,
                                                  max_entropy_on_ignore=False,
                                                  train_with_void_class=False):
  """Adds softmax cross entropy loss for logits of each scale.

  Args:
    scales_to_logits: A map from logits names for different scales to logits.
      The logits have shape [batch, logits_height, logits_width, num_classes].
    labels: Groundtruth labels with shape [batch, image_height, image_width, 1].
    num_classes: Integer, number of target classes.
    ignore_label: Integer, label to ignore.
    loss_weight: Float, loss weight.
    upsample_logits: Boolean, upsample logits or not.
    scope: String, the scope for the loss.

  Raises:
    ValueError: Label or logits is None.
  """
  if labels is None:
    raise ValueError('No label for softmax cross entropy loss.')

  for scale, logits in six.iteritems(scales_to_logits):
    loss_scope = None
    if scope:
      loss_scope = '%s_%s' % (scope, scale)

    if upsample_logits:
      # Label is not downsampled, and instead we upsample logits.
      logits = tf.image.resize_bilinear(
          logits,
          preprocess_utils.resolve_shape(labels, 4)[1:3],
          align_corners=True)
      scaled_labels = labels
    else:
      # Label is downsampled to the same size as logits.
      scaled_labels = tf.image.resize_nearest_neighbor(
          labels,
          preprocess_utils.resolve_shape(logits, 4)[1:3],
          align_corners=True)

    scaled_labels = tf.reshape(scaled_labels, shape=[-1])
    one_hot_labels = slim.one_hot_encoding(
        scaled_labels, num_classes, on_value=1.0, off_value=0.0)
    if max_entropy_on_ignore:
      not_ignore_mask = loss_weight
      flat_one_hot = tf.ones_like(one_hot_labels) / num_classes
      one_hot_labels = tf.where(
          tf.equal(scaled_labels, ignore_label),
          flat_one_hot, one_hot_labels)
    elif train_with_void_class:
      not_ignore_mask = loss_weight
      scaled_labels = tf.where(tf.equal(scaled_labels, ignore_label),
                               tf.ones_like(scaled_labels) * (num_classes - 1),
                               scaled_labels)
      one_hot_labels = slim.one_hot_encoding(
          scaled_labels, num_classes, on_value=1.0, off_value=0.0)
    else:
      not_ignore_mask = tf.to_float(tf.not_equal(scaled_labels,
                                                 ignore_label)) * loss_weight
    tf.losses.softmax_cross_entropy(
        one_hot_labels,
        tf.reshape(logits, shape=[-1, num_classes]),
        weights=not_ignore_mask,
        scope=loss_scope)


def safe_mean(losses):
  with tf.name_scope('safe_mean'):
    total_loss = tf.reduce_sum(losses)
    num_elements = tf.to_float(tf.shape(losses)[0])
    return tf.div_no_nan(total_loss, num_elements, name="value")


def add_dirichlet_loss_for_each_scale(scales_to_logits,
                                      labels,
                                      num_classes,
                                      ignore_label,
                                      dirichlet_weight=1.0,
                                      ood_weight=0.,
                                      cross_entropy_weight=0.,
                                      label_smoothing=0.1,
                                      upsample_logits=True,
                                      use_dirichlet_kl=True,
                                      scope=None):
  if labels is None:
    raise ValueError('No label for softmax cross entropy loss.')

  for scale, logits in six.iteritems(scales_to_logits):
    loss_scope = None
    if scope:
      loss_scope = '%s_%s' % (scope, scale)

    if upsample_logits:
      # Label is not downsampled, and instead we upsample logits.
      logits = tf.image.resize_bilinear(
          logits,
          preprocess_utils.resolve_shape(labels, 4)[1:3],
          align_corners=True)
      scaled_labels = labels
    else:
      # Label is downsampled to the same size as logits.
      scaled_labels = tf.image.resize_nearest_neighbor(
          labels,
          preprocess_utils.resolve_shape(logits, 4)[1:3],
          align_corners=True)

    scaled_labels = tf.reshape(scaled_labels, shape=[-1])
    one_hot_labels = slim.one_hot_encoding(
        scaled_labels, num_classes, on_value=1.0, off_value=0.0)
    smooth_labels = (one_hot_labels * (1. - label_smoothing)
                     + label_smoothing / num_classes)
    in_mask = tf.not_equal(scaled_labels, ignore_label)
    logits = tf.reshape(logits, shape=[-1, num_classes])
    logits = tf.check_numerics(logits, 'Logits are inf or nan.')

    pred_in_dist = tfp.distributions.Dirichlet(
        tf.exp(tf.boolean_mask(logits, in_mask)), allow_nan_stats=False)

    if dirichlet_weight > 0:
      if use_dirichlet_kl:
        alpha_0 = 1000
        target_dist = tfp.distributions.Dirichlet(
            tf.boolean_mask(smooth_labels, in_mask) * alpha_0, allow_nan_stats=False)
        target_kl = tfp.distributions.kl_divergence(
           target_dist, pred_in_dist, allow_nan_stats=True)
        # target_kl = tf.Print(target_kl, [tf.reduce_mean(target_kl), tf.reduce_min(target_kl),
                                         # tf.reduce_max(target_kl),
                                         # tf.reduce_sum(tf.to_float(tf.is_finite(target_kl)))])
        target_kl = tf.boolean_mask(target_kl, tf.is_finite(target_kl))
        target_kl = tf.check_numerics(target_kl, 'In-distribution KL is inf or nan.')
        target_loss = dirichlet_weight * safe_mean(target_kl)
        target_loss = tf.identity(target_loss, loss_scope+'/target_kl')
        tf.losses.add_loss(target_loss)
      else:
        nll = -pred_in_dist.log_prob(tf.boolean_mask(smooth_labels, in_mask))
        # nll = tf.Print(nll, [tf.shape(nll), tf.reduce_mean(nll), tf.reduce_min(nll),
                             # tf.reduce_max(nll), tf.reduce_mean(tf.to_float(tf.is_finite(nll)))])
        nll = tf.boolean_mask(nll, tf.is_finite(nll))
        nll = tf.check_numerics(nll, 'In-distribution NLL is inf or nan.')
        nll_loss = dirichlet_weight * safe_mean(nll)
        nll_loss = tf.identity(nll_loss, loss_scope+'/dirichlet_nll')
        tf.losses.add_loss(nll_loss)

    if ood_weight > 0:
      ood_mask = tf.math.logical_not(in_mask)
      pred_ood_dist = tfp.distributions.Dirichlet(
          tf.exp(tf.boolean_mask(logits, ood_mask)), allow_nan_stats=False)
      ood_dist = tfp.distributions.Dirichlet(
          tf.ones(num_classes), allow_nan_stats=False)
      ood_kl = tfp.distributions.kl_divergence(
          ood_dist, pred_ood_dist, allow_nan_stats=True)
      # ood_kl = tf.Print(ood_kl, [tf.reduce_mean(ood_kl), tf.reduce_min(ood_kl),
                                 # tf.reduce_max(ood_kl), tf.reduce_mean(tf.to_float(tf.is_finite(ood_kl)))])
      ood_kl = tf.boolean_mask(ood_kl, tf.is_finite(ood_kl))
      ood_kl = tf.check_numerics(ood_kl, 'OoD KL is inf or nan.')
      ood_loss = safe_mean(ood_weight * ood_kl)
      ood_loss = tf.identity(ood_loss, loss_scope+'/ood_kl')
      tf.losses.add_loss(ood_loss)

    if cross_entropy_weight > 0:
      ce_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
              labels=tf.boolean_mask(one_hot_labels, in_mask),
              logits=tf.boolean_mask(logits, in_mask))
      # ce_loss = tf.boolean_mask(ce_loss, tf.is_finite(ce_loss))
      ce_loss = safe_mean(cross_entropy_weight * ce_loss)
      ce_loss = tf.check_numerics(ce_loss, 'CE loss is inf or nan.')
      ce_loss = tf.identity(ce_loss, loss_scope+'/cross_entropy')
      tf.losses.add_loss(ce_loss)


def get_model_init_fn(train_logdir,
                      tf_initial_checkpoint,
                      initialize_last_layer,
                      last_layers,
                      ignore_missing_vars=False):
  """Gets the function initializing model variables from a checkpoint.

  Args:
    train_logdir: Log directory for training.
    tf_initial_checkpoint: TensorFlow checkpoint for initialization.
    initialize_last_layer: Initialize last layer or not.
    last_layers: Last layers of the model.
    ignore_missing_vars: Ignore missing variables in the checkpoint.

  Returns:
    Initialization function.
  """
  if tf_initial_checkpoint is None:
    tf.logging.info('Not initializing the model from a checkpoint.')
    return None

  if tf.train.latest_checkpoint(train_logdir):
    tf.logging.info('Ignoring initialization; other checkpoint exists')
    return None

  tf.logging.info('Initializing model from path: %s', tf_initial_checkpoint)

  # Variables that will not be restored.
  exclude_list = ['global_step']
  if not initialize_last_layer:
    exclude_list.extend(last_layers)

  variables_to_restore = slim.get_variables_to_restore(exclude=exclude_list)

  if variables_to_restore:
    return slim.assign_from_checkpoint_fn(
        tf_initial_checkpoint,
        variables_to_restore,
        ignore_missing_vars=ignore_missing_vars)
  return None


def get_model_gradient_multipliers(last_layers, last_layer_gradient_multiplier):
  """Gets the gradient multipliers.

  The gradient multipliers will adjust the learning rates for model
  variables. For the task of semantic segmentation, the models are
  usually fine-tuned from the models trained on the task of image
  classification. To fine-tune the models, we usually set larger (e.g.,
  10 times larger) learning rate for the parameters of last layer.

  Args:
    last_layers: Scopes of last layers.
    last_layer_gradient_multiplier: The gradient multiplier for last layers.

  Returns:
    The gradient multiplier map with variables as key, and multipliers as value.
  """
  gradient_multipliers = {}

  for var in slim.get_model_variables():
    # Double the learning rate for biases.
    if 'biases' in var.op.name:
      gradient_multipliers[var.op.name] = 2.

    # Use larger learning rate for last layer variables.
    for layer in last_layers:
      if layer in var.op.name and 'biases' in var.op.name:
        gradient_multipliers[var.op.name] = 2 * last_layer_gradient_multiplier
        break
      elif layer in var.op.name:
        gradient_multipliers[var.op.name] = last_layer_gradient_multiplier
        break

  return gradient_multipliers


def get_model_learning_rate(
    learning_policy, base_learning_rate, learning_rate_decay_step,
    learning_rate_decay_factor, training_number_of_steps, learning_power,
    slow_start_step, slow_start_learning_rate):
  """Gets model's learning rate.

  Computes the model's learning rate for different learning policy.
  Right now, only "step" and "poly" are supported.
  (1) The learning policy for "step" is computed as follows:
    current_learning_rate = base_learning_rate *
      learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)
  See tf.train.exponential_decay for details.
  (2) The learning policy for "poly" is computed as follows:
    current_learning_rate = base_learning_rate *
      (1 - global_step / training_number_of_steps) ^ learning_power

  Args:
    learning_policy: Learning rate policy for training.
    base_learning_rate: The base learning rate for model training.
    learning_rate_decay_step: Decay the base learning rate at a fixed step.
    learning_rate_decay_factor: The rate to decay the base learning rate.
    training_number_of_steps: Number of steps for training.
    learning_power: Power used for 'poly' learning policy.
    slow_start_step: Training model with small learning rate for the first
      few steps.
    slow_start_learning_rate: The learning rate employed during slow start.

  Returns:
    Learning rate for the specified learning policy.

  Raises:
    ValueError: If learning policy is not recognized.
  """
  global_step = tf.train.get_or_create_global_step()
  if learning_policy == 'step':
    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        global_step,
        learning_rate_decay_step,
        learning_rate_decay_factor,
        staircase=True)
  elif learning_policy == 'poly':
    learning_rate = tf.train.polynomial_decay(
        base_learning_rate,
        global_step,
        training_number_of_steps,
        end_learning_rate=0,
        power=learning_power)
  else:
    raise ValueError('Unknown learning policy.')

  # Employ small learning rate at the first few steps for warm start.
  return tf.where(global_step < slow_start_step, slow_start_learning_rate,
                  learning_rate)
