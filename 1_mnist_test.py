from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc 
import os
import tensorflow as tf 


# mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

# #查看训练数据大小
# print(mnist.train.images.shape)     #(55000, 784)
# print(mnist.train.labels.shape)     #(55000, 10)

# #查看验证数据的大小
# print(mnist.validation.images.shape)        #(5000, 784)
# print(mnist.validation.images.shape)        #(5000, 784)

# #查看测试数据大小
# print(mnist.test.images.shape)      #(10000, 784)
# print(mnist.test.images.shape)      #(10000, 784)

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# save_dir = 'MNIST_data/raw/'

# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)


# for i in range(20):
#     image_array = mnist.train.images[i,:]
#     image_array = image_array.reshape(28,28)
#     filename = save_dir + 'mnist_train_{}.jpg'.format(i)
#     scipy.misc.toimage(image_array,cmin=0.0,cmax=1.0).save(filename)



####################################################################################
# softmax回归模型
####################################################################################
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# softmax计算出的输出label
y = tf.nn.softmax(tf.matmul(x, w)+b)
# 实际label
y_ = tf.placeholder(tf.float32, [None, 10])

# 根据y和y_构造交叉熵损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
# 梯度下降优化器，0.01：学习率
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
    # 获取训练集的100条数据
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # feed batch_xs给计算图中的x节点，计算softmax后的输出，和y_节点的真实结果做比较，进行梯度下降，更新权值w和偏置b
    sess.run(train_step,feed_dict={x: batch_xs, y_: batch_ys})

# 正确的预测结果
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
# 计算预测准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
# feed 测试集的图片给计算图的x节点，计算softmax输出的label和测试集的真实label做批量比较，计算准确率
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))