import tensorflow as tf
from numpy.random import RandomState
'''
自定义损失函数
'''
batch_size = 8

# 两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')

y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

# 定义一个单层神经网络的前向传播过程
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))

y = tf.matmul(x, w1)

loss_less = 10
loss_more = 1

loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))

learning_rate = 0.001
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 设置回归的正确值为两个输入的和加上一个随机量， 加上一个随机量是为了加入不可预测的噪音，否则不同损失函数的意义就不大了
# 因为不同损失函数都会在能完全预测正确的时候最低，一般来说噪音为一个均值为0的小量
Y = [[x1 + x2 + rdm.rand() / 10 - 0.05] for (x1, x2) in X]

# 训练神经网络
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
    print(sess.run(w1))


