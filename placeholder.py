import tensorflow as tf
from numpy.random import RandomState
'''
    为了减少计算图中的节点数，tensorflow提供了placeholder机制用于提供输入数据
    placeholder相当于定义了一个位置，这个位置中的数据在程序运行时再指定
'''

# 定义训练数据的大小
batch_size = 8

# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

# 定义placeholder作为存放输入数据的地方，这里维度也不一定要定义，
# 但如果维度是确定的，那么给出维度可以降低出错的概率

x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播算法步骤
learning_rate = 0.001
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# 随机生成模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

# 定义样本标签
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# 创建会话
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))

    STEPS = 5000

    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("在%d次训练后,在所有数据上的损失函数为%g"%(i, total_cross_entropy))
    print(sess.run(w1))
    print(sess.run(w2))