#!/usr/bin/python
# -*- coding:utf-8 -*-

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# 定义图片的像素，原始图片 32X32
# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
# 分类数量
NUM_CLASSES = 10
# 训练时每个epoch中的数据样本数
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
# eval时每个epoch中的数据样本数
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# 从 CIFAR10 数据文件中读取样本
# filename_queue 一个队列的文件名
def read_cifar10(filename_queue):
  """Reads and parses examples from CIFAR10 data files.
  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.
  从 CIFAR10 中读取并解析样本。如果想要 N 路并行读取，那么请调用此函数 N 次。
  Args:
    filename_queue: A queue of strings with the filenames to read from.
    包含要读取的文件名的字符串队列。
  Returns:
    一个对象表示单个样本，包含以下字段：
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.标量字符串 Tensor ，描述此样本的文件名和记录号
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  # 分类结果的长度，CIFAR-100 长度是 2
  label_bytes = 1  # 2 for CIFAR-100
  result.height = 32
  result.width = 32
  # 3位表示 rgb 颜色，（0-255，0-255，0-255）
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  # 单个记录的总长度 = 分类结果长度 + 图片长度
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  # 读取
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  # 从字符串转换为 uint8 的向量，其长度为 record_bytes
  # 其中的 tf.decode_raw() 是将字符串的 bytes 重新解释为 tensor
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  # 第一位代表 label-图片的正确分类结果，从 uint8 转换为 int32 类型
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  # 分类结果label之后的数据代表图片，我们使用 reshape 进行了调整， 从 [depth * height * width] 改为 [depth, height, width]
  # tf.strided_slice 解释，请看 https://blog.csdn.net/banana1006034246/article/details/75092388
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  # 格式转换，从 [depth, height, width] 转换为 [height, width, depth]，使用 tf 的函数为 tf.transpose
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])
  # 将结果返回
  return result

# 构建一个排列后的一组图片和分类
def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.
     构建一个 images 和 labels 的队列 batch 
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32. 3-D 的 Tensor
    label: 1-D Tensor of type.int32 1-D 的 Tensor
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
      队列中的样本的最小值
    batch_size: Number of images per batch.每个 batch 中的 images 数量。
    shuffle: boolean indicating whether to use a shuffling queue.是否使用 shuffle 队列。
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  # 线程数
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  # 在可视化工具中显示训练图像
  tf.summary.image('images', images)
  # 返回结果
  return images, tf.reshape(label_batch, [batch_size])

# 为 CIFAR-10 评价构建输入
# data_dir 路径
# batch_size 一个batch 中图片数目的大小
def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.
     使用 Reader ops 为CIFAR 训练构建输入。
  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
               for i in xrange(1, 6)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  # 创建包含要读取的文件名称的队列
  # tf.train.string_input_producer() 将字符串（如文件名）输出到输入管道的队列
  filename_queue = tf.train.string_input_producer(filenames)
  # 在 data_augmentation 这个 op 下进行以下操作：
  with tf.name_scope('data_augmentation'):
    # Read examples from files in the filename queue.
    # 从 filename 队列中的 files 中读取 样本
    read_input = read_cifar10(filename_queue)
    # 进行数据格式转换
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    # 图片的长，宽
    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    # 随机裁剪图像的 【长，宽】部分，从原来的 32x32 到 24x24
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Randomly flip the image horizontally.
    # 随机旋转图片
    # tf.image.random_flip_left_right() 以 1/2 的概率沿着图片的 width 进行翻转（从左到右）
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458.
    # 亮度变换
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    # 对比度变换
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    # 标准化
    float_image = tf.image.per_image_standardization(distorted_image)

    # Set the shapes of tensors.
    # 设置 tensors 的 shape
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    # 确保 shuffle 的随机性
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)

# 为 CIFAR10 评价构建输入
# eval_data 使用训练还是评价数据集
# data_dir 路径
# batch_size 一个 batch 的大小
def inputs(eval_data, data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # 如果不是评价数据集，那么就是使用训练数据集
  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
  # 查看 file 是否存在
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)
  #  在 input 这个 op 下操作
  with tf.name_scope('input'):
    # Create a queue that produces the filenames to read.
    # 创建可以读取 filenames 的队列
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    # 在 filename 队列中的 files 中读取 样本
    read_input = read_cifar10(filename_queue)
    # 进行数据类型转换
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    # 图像的长宽
    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    # 图像评价阶段
    # 裁剪图像的中央
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           height, width)

    # Subtract off the mean and divide by the variance of the pixels.
    # 标准化
    float_image = tf.image.per_image_standardization(resized_image)

    # Set the shapes of tensors.
    # 设置 tensors 的 shape
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    # 确保随机 shuffle
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  # 生成 图像和对应 label 的 batch
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)