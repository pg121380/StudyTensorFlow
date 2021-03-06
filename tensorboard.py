'''
如何使用tensorboard
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/MNIST_data/", one_hot=True)

batch_size = 100

n_batch = mnist.train.num_examples // batch_size

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

w = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1))
b = tf.Variable(tf.truncated_normal([200], stddev=0.1))
layer1 = tf.nn.relu(tf.matmul(x, w) + b)

w2 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
b2 = tf.Variable(tf.truncated_normal([10], stddev=0.1))

prediction = tf.nn.softmax(tf.matmul(layer1, w2) + b2)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(1):
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("%d次迭代后,在测试集上的准确率为:%f" % (epoch, acc))
