import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# 卷积层1
x_image = tf.reshape(x, shape=[-1, 28, 28, 1])

kernel_1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1, dtype=tf.float32), name='kernel_1')

conv1_value = tf.nn.conv2d(input=x_image, filter=kernel_1, strides=[1, 1, 1, 1], padding='SAME')

bias_value_1 = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='bias_value_1')

bias1_output = tf.nn.bias_add(conv1_value, bias_value_1)

conv1_output = tf.nn.relu(bias1_output)

pool1_output = tf.nn.max_pool(conv1_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 卷积层2

kernel_2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1, dtype=tf.float32), name='kernel_2')

conv2_value = tf.nn.conv2d(input=pool1_output, filter=kernel_2, strides=[1, 1, 1, 1], padding='SAME')

bias_value_2 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='bias_value_2')

bias2_output = tf.nn.bias_add(conv2_value, bias_value_2)

conv2_output = tf.nn.relu(bias2_output)

pool2_output = tf.nn.max_pool(conv2_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 全连接层1
w_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], dtype=tf.float32, stddev=0.1), name='w_fcl')

b_fc1 = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32))

pool2_output_flat = tf.reshape(pool2_output, shape=[-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(pool2_output_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)

h_fcl_drop = tf.nn.dropout(h_fc1, keep_prob)

# 全连接层2
w_fc2 = tf.Variable(tf.truncated_normal([1024, 10], dtype=tf.float32, stddev=0.1), name='w_fc2')

b_fc2 = tf.Variable(tf.constant(0.0, shape=[10], dtype=tf.float32))

#
y_conv = tf.matmul(h_fcl_drop, w_fc2) + b_fc2

# tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))是tensorflow提供的
# logit是展平后的特征举证乘以权重加上偏置logit = w * x + b
# logit作为softmax的参数，输出该目标属于各个类别的概率，
# 在取出最大的概率类别作为预测label与实际label做交叉熵
# 最后tf.reduce_mean求所有交叉熵的平均值，目标是越来越小
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

# adam一种优化器算法，寻找全局最优
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# argmax返回one-hot编码向量中最大的值的索引，用equal进行比对
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

# 去所有比对结果的平均值(tf.cast将correct_prediction转为tf.float32)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# InteractiveSession()允许在运行计算图的时候额外插入一些计算图
# sess = tf.InteractiveSession()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# sess.run(tensor),tensor.run(),tensor.eval()  三种方式均是运行计算图
# 其中tensor.run(),tensor.eval()属于额外插入计算图进行求目标tensor的值,所以需要使用tf.InteractiveSession()创建会话
# 三者都会运行这个计算图，但eval只会计算目标tensor
for i in range(2000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        # train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d,training accuracy %g' % (i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print('test accuracy %g'%sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))