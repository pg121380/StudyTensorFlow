import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/MNIST_data/", one_hot=True)

batch_size = 1
n_batch = mnist.test.num_examples // batch_size


# 初始化权值
def weight_variable(shape):
    weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    return weight


# 初始化偏置值
def biases_variable(shape):
    biases = tf.constant(0.1, shape=shape)
    return tf.Variable(biases)


# 卷积层
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
def max_poling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

w_conv1 = weight_variable([5, 5, 1, 16])
b_conv1 = biases_variable([16])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_poling_2x2(h_conv1)

w_conv2 = weight_variable([5, 5, 16, 32])
b_conv2 = biases_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_poling_2x2(h_conv2)

w_fc1 = weight_variable([7 * 7 * 32, 100])
b_fc1 = biases_variable([100])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weight_variable([100, 10])
b_fc2 = biases_variable([10])

prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob:0.7})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y:mnist.test.labels, keep_prob: 1.0})
        print("在%d此迭代后,准确率为%f"% (epoch, acc))

