import tensorflow as tf 

# g1 = tf.Graph()
# with g1.as_default():
#     v = tf.get_variable('v',initializer=tf.zeros_initializer(shape=[1]))

# g2 = tf.Graph()

# with g2.as_default():
#     v = tf.get_variable('v',initializer=tf.ones_initializer(shape=[1]))

# with tf.Session(graph=g1) as sess:
#     tf.initialize_all_variables().run()
#     with tf.variable_scope('',reuse=True):
#         print(sess.run(tf.get_variable('v')))

# with tf.Session(graph=g2) as sess:
#     tf.initialize_all_variables().run()
#     with tf.variable_scope('',reuse=True):
#         print(sess.run(tf.get_variable('v')))

#=========================================================================================

a = tf.constant([1.0,2.0],name='a')
b = tf.constant([2.0,3.0],name='b')

result = tf.add(a,b,name='add')
result2 = result + a
result3 = result - b
# print(result3)

#=========================================================================================
# sess = tf.Session()
# with sess.as_default():
#     print(result.eval())


# InteractiveSession    默认会话函数
# ConfigProto   配置会话
#=========================================================================================
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=True)
sess = tf.InteractiveSession(config=config)
print(result.eval())
sess.close()

