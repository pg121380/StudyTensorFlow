import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

x = tf.placeholder(tf.float32, [None, 1], name="x")
y = tf.placeholder(tf.float32, [None, 1], name="y")

w_1 = tf.Variable(tf.random_normal([1, 10]))
biases_1 = tf.Variable(tf.zeros([1, 10]))

layer_1 = tf.nn.tanh(tf.matmul(x, w_1) + biases_1)

w_2 = tf.Variable(tf.random_normal([10, 1]))
biases_2 = tf.Variable(tf.random_normal([1, 1]))

predict = tf.nn.tanh(tf.matmul(layer_1, w_2) + biases_2)

loss = tf.reduce_mean(tf.square(predict - y_data))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})
        if i % 100 == 0:
            print("after %d training step(s) the loss is %f" % (i, sess.run(loss, feed_dict={x: x_data, y: y_data})))

    predict_value = sess.run(predict, feed_dict={x: x_data, y: y_data})
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, predict_value, 'r-')
    plt.show()
