import tensorflow as tf
import numpy as np
tf.set_random_seed(777) 

filename_queue = tf.train.string_input_producer(['data-03-diabetes.csv','data-04-diabetes.csv'], shuffle=False, name='filename_queue')
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
record_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

with tf.name_scope('layer1'):
     W1 = tf.Variable(tf.random_normal([8, 16]), name='weight1')
     b1 = tf.Variable(tf.random_normal([16]), name='bias')
     L1 = tf.add(tf.matmul(X, W1), b1)

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_normal([16, 1]), name='weight2')
    b2 = tf.Variable(tf.random_normal([1]), name='bias2')

with tf.name_scope('optimizer'):
    hypothesis = tf.sigmoid(tf.add(tf.matmul(L1, W2), b2))
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
for step in range(3001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, _ = sess.run([cost, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 1000 == 0:
        print(step, cost_val)

h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_batch, Y: y_batch})
print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

# Ask 
h1, p1 = sess.run([hypothesis, predicted], feed_dict={X: [[0,0.3,0.1,-0.2,0, -0.3, -0.7, -0.8]]})
print("Your diabetes will be ", h1, p1)
     

coord.request_stop()
coord.join(threads)
