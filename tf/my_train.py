#!usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

matrix_1 = tf.constant([[3, 3]])
# print('matrix_1', matrix_1)
# print('matrix_1_shape', matrix_1.get_shape)

# a=np.array([[1,2],[4,5]])
# matrix_3 = tf.constant(a)
matrix_2 = tf.constant([[2],[2]])
# print('matrix_3', matrix_3)
# print('matrix_3_shape', matrix_3.get_shape)

product = tf.matmul(matrix_1, matrix_2)
# print('product', product)
# with tf.Session() as sess:
#     for i in matrix_1.eval():
#         print('matrix_1', i)
#     for j in matrix_2.eval():
#         print('matrix_2', j)
#     for z in product.eval():
#         print('zzzz', z)

# sess = tf.Session()
# result = sess.run(product)
# print(result)

# # 任务完成，关闭会话
# sess.close()


with tf.Session() as sess:
    result = sess.run(product)
    print(result)

# 使用 with ... Device 语句来指派特定的 CPU 或 GPU 执行操作
# "/cpu:0" 机器的 cpu "/gpu:0" 机器的第一个gpu，依此类推
# with tf.Session() as sess:
#     with tf.device("/cpu:0"):
#         matrix_1 = tf.constant([[3, 3]])
#         matrix_2 = tf.constant([[2], [2]])
#         product_2 = tf.matmul(matrix_1, matrix_2)


# -------- numpy convert to tensor-----
'''
a = np.array([[1, 2],[3, 4]])
print('aaaaa', a)
b = tf.constant(a)
print('bbbbb', b)
print('bbbb_shape', b.get_shape)

with tf.Session() as sess:
    print('session_b', b)
    for i in b.eval():
        print('iiiiiii', i)
    
    print('a is a array', a)

    tensor_a = tf.convert_to_tensor(a)
    print('now it is tensor...', tensor_a)
    '''

'''
# 创建一个变量, 初始化为标量 0.
state = tf.Variable(0, name="counter")

# 创建一个 op, 其作用是使 state 增加 1

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
init_op = tf.global_variables_initializer()

# 启动图, 运行 op
with tf.Session() as sess:
  # 运行 'init' op
  sess.run(init_op)
  # 打印 'state' 的初始值
  print sess.run(state)
  # 运行 op, 更新 'state', 并打印 'state'
  for _ in range(3):
    sess.run(update)
    print(sess.run(state))
'''

# input1 = tf.constant(3.0)
# input2 = tf.constant(2.0)
# input3 = tf.constant(5.0)
# intermed = tf.add(input2, input3)
# mul = tf.multiply(input1, intermed)

# with tf.Session() as sess:
#   result = sess.run([mul, intermed])
#   print(result)

