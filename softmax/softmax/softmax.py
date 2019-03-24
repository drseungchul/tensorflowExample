
#tensorboard --logdir==training:logs --host=127.0.0.1:6006

import tensorflow as tf
import numpy as np

learning_rate = 0.1
learning_iteration = 2001
n_classes = 3


filename_queue = tf.train.string_input_producer(['data1.csv','data2.csv'], shuffle=False, name='filename_queue')
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
record_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)
train_x_batch, train_y_batch = tf.train.batch([xy[0:4], xy[4:]], batch_size=10)

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, n_classes])

with tf.name_scope('layer1'):
     W1 = tf.Variable(tf.random_normal([4, 4]), name='weight1')
     b1 = tf.Variable(tf.random_normal([4]), name='bias')
     L1 = tf.add(tf.matmul(X, W1), b1)

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_normal([4, n_classes]), name='weight2')
    b2 = tf.Variable(tf.random_normal([n_classes]), name='bias2')

with tf.name_scope('optimizer'):
    logits = tf.matmul(L1, W2) + b2
    hypothesis = tf.nn.softmax(logits)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
for step in range(learning_iteration):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_batch, Y: y_batch})
    if step % 400 == 0:
        print(step, cost_val)
    
    
print('--------------')
# Testing & One-hot encoding
a = sess.run(logits, feed_dict={X: [[1, 11, 7, 9]]})
print(a, sess.run(tf.argmax(a, 1)))

print('--------------')
b = sess.run(logits, feed_dict={X: [[1, 3, 4, 3]]})
print(b, sess.run(tf.argmax(b, 1)))

print('--------------')
c = sess.run(logits, feed_dict={X: [[1, 1, 0, 1]]})
print(c, sess.run(tf.argmax(c, 1)))


coord.request_stop()
coord.join(threads)
