#tensorboard  --logdir==training:logs --host=127.0.0.1:6006
# linear regression

import tensorflow as tf
import numpy as np

learning_rate = 0.01
filename_queue = tf.train.string_input_producer(['data-01-test-score.csv','data-02-test-score.csv'], shuffle=False, name='filename_queue')
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# collect batches of csv in
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

with tf.name_scope('layer1'):
     W1 = tf.Variable(tf.random_normal([3, 10]), name='weight1')
     b1 = tf.Variable(tf.random_normal([10]), name='bias')
     L1 = tf.add(tf.matmul(X, W1), b1)

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_normal([10, 1]), name='weight2')
    b2 = tf.Variable(tf.random_normal([1]), name='bias2')

with tf.name_scope('optimizer'):
    hypothesis = tf.add(tf.matmul(L1, W2), b2)
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(cost)
    tf.summary.scalar('cost', cost)

summary = tf.summary.merge_all()

# Launch the graph in a session.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

writer = tf.summary.FileWriter('./logs')
writer.add_graph(sess.graph)

for step in range(2000):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 500 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
    s, _ = sess.run([summary, train], feed_dict={X: x_batch, Y: y_batch})
    writer.add_summary(s, global_step=None)

coord.request_stop()
coord.join(threads)

# Ask my score
print("Your score will be ",
      sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ",
      sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))

