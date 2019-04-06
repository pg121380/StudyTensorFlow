import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/MNIST_data/", one_hot=True)

batch_size = 100

n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

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

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

    saver.restore(sess, 'net/my_net.ckpt')
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
