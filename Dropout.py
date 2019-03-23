'''
这个代码里有奇怪的问题
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/MNIST_data/", one_hot=True)
batch_size = 100
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)      # 设置drop-out时有百分之多少的神经元是工作的

w1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1))
biases1 = tf.Variable(tf.zeros([2000]) + 0.1)

layer1 = tf.nn.tanh(tf.matmul(x, w1) + biases1)
layer1_drop = tf.nn.dropout(layer1, keep_prob)

w2 = tf.Variable(tf.truncated_normal([2000, 10], stddev=0.1))
biases2 = tf.Variable(tf.zeros([10]) + 0.1)


prediction = tf.nn.softmax(tf.matmul(layer1_drop, w2) + biases2)


# prediction = tf.nn.softmax(tf.matmul(x, w) + biases)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

train_step = tf.train.AdamOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.7})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print("在第%d次迭代后,测试集精确度为%f" % (epoch, acc), end=", ")
