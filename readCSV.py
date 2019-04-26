import tensorflow as tf

filename_queue = tf.train.string_input_producer(["rrrrr.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1.0], [1.0], [1.0], [1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]]
col1, col2, col3, col4,col5, col6, col7, col8, col9,col10,y,col11 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11])
# x = tf.placeholder(tf.float32, [1, 10], name="x")
# y = tf.placeholder(tf.float32, [1, 1], name="y")
#
# w_1 = tf.Variable(tf.random_normal([10, 10]))
# biases_1 = tf.Variable(tf.zeros([1, 10]))
#
# layer_1 = tf.nn.tanh(tf.matmul(x, w_1) + biases_1)
#
# w_2 = tf.Variable(tf.random_normal([10, 1]))
# biases_2 = tf.Variable(tf.random_normal([1, 1]))
#
# predict = tf.nn.tanh(tf.matmul(layer_1, w_2) + biases_2)
#
# loss = tf.reduce_mean(tf.square(predict - y))
#
# train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


with tf.Session() as sess:
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(2970):
    example,label = sess.run([features, y])
    print(sess.run(example))

    # sess.run(train_step, feed_dict={x:example, y:label})
  coord.request_stop()
  coord.join(threads)