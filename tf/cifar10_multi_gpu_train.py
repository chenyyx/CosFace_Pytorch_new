#!/usr/bin/python
# -*- coding:utf-8 -*-

"""A binary to train CIFAR-10 using multiple GPUs with synchronous updates.
使用具有同步更新的多个 GPU 训练 CIFAR-10 的二进制文件。
Accuracy:
cifar10_multi_gpu_train.py achieves ~86% accuracy after 100K steps (256
epochs of data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
--------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
2 Tesla K20m  | 0.13-0.20              | ~84% at 30K steps  (2.5 hours)
3 Tesla K20m  | 0.13-0.18              | ~84% at 30K steps
4 Tesla K20m  | ~0.10                  | ~84% at 30K steps
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def tower_loss(scope, images, labels):
  """Calculate the total loss on a single tower running the CIFAR model.
     计算运行 CIFAR 模型的单个 tower 上的总损失。
  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    images: Images. 4D tensor of shape [batch_size, height, width, 3].
    labels: Labels. 1D tensor of shape [batch_size].
  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

  # Build inference Graph.
  # 创建 infer Graph
  logits = cifar10.inference(images)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  # 构建 graph 中计算损失的部分。请注意，我们将使用下面的自定义函数组装 total_loss
  _ = cifar10.loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  # 仅汇集当前 tower 的所有损失
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  # 计算当前 tower 的总 loss
  total_loss = tf.add_n(losses, name='total_loss')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  # 附上所有 独立损失 和 总 loss 的标量摘要，对 loss 的平均版本做同样的事
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
    tf.summary.scalar(loss_name, l)
  # 将 total_loss 返回
  return total_loss

# 平均梯度
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  计算所有 tower 中每个共享变量的平均 grad。
  请注意，此功能提供跨所有 tower 的同步点
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:注意，每个 grad_and_vars 就像下面这样：
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      # 将 0 维 添加到 grad 以表示 tower 
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      # 附加在 tower 维度上，我们将在下面平均
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    # 计算 grad 的平均值
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    # 请记住，变量是多余的，因为它们是跨 tower 共享的。所以，我们将返回第一个 tower 的变量指针。
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():
  """Train CIFAR-10 for a number of steps.训练一些步的 CIFAR-10"""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    # 创建一个 variable 来计算 train() 调用的数量。这等于处理 btaches * FLAGS.num_gpus 
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    # 计算 learning rate 的计划
    num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    # 衰减 learning rate
    lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    cifar10.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    # 创建一个执行 梯度下降 的 optim
    opt = tf.train.GradientDescentOptimizer(lr)

    # Get images and labels for CIFAR-10.
    # 获取 CIFAR-10 的 images 和 对应的 labels
    images, labels = cifar10.distorted_inputs()
    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * FLAGS.num_gpus)
    # Calculate the gradients for each model tower.
    # 计算每个 model tower 的 梯度
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
            # Dequeues one batch for the GPU
            # 为 GPU 取出一个队列数据
            image_batch, label_batch = batch_queue.dequeue()
            # Calculate the loss for one tower of the CIFAR model. This function
            # constructs the entire CIFAR model but shares the variables across
            # all towers.
            # 计算 CIFAR 模型的一个 tower 的损失。此函数构造整个 CIFAR 模型，但在所有 tower 中共享变量。
            loss = tower_loss(scope, image_batch, label_batch)

            # Reuse variables for the next tower.
            # 为下一个 tower 重用 variables
            tf.get_variable_scope().reuse_variables()

            # Retain the summaries from the final tower.
            # 保留 final tower 的 summary
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            # Calculate the gradients for the batch of data on this CIFAR tower.
            # 计算在此 CIFAR tower 上的 btach 的梯度
            grads = opt.compute_gradients(loss)

            # Keep track of the gradients across all towers.
            # 跟踪 跨所有 tower 的 grad
            tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    # 我们必须计算每个 grad 的平均值。请注意，这是所有 tower 的同步点。
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    # 添加一个跟踪 lr 的 summary
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    # 添加 grad 的直方图
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    # 应用 grad 来调整这些共享变量
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    # 为训练参数添加直方图
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    # 跟踪所有训练变量的 移动平均
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    # group 所有的更新到 单个 train op 中
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    # 创建一个 saver
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    # 从最后的 tower summaries 构建 summary 操作
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    # 创建一个 初始化 操作
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    # 开始在 Graph 上运行 ops 。必须将 allow_soft_placement 设置为 True 才能在 GPU 上构建 tower，因为某些 ops 没有 GPU 实现。
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    # 开始 queue 的运行
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
      # 每 step 能被 10 整除，就进行 print 
      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))
      # 若 step 能被 100 整除，那么就进行写入 summary
      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      # 保存模型的 checkpoint
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  # 下载 cifar10 数据集
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  # 进行训练
  train()


if __name__ == '__main__':
  tf.app.run()