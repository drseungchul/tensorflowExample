
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint
import matplotlib.pyplot as plt

input_dim = 4
seq_length = 5
hidden_dim = 20
num_hidden_layers = 2
output_dim = 1
learning_rate = 0.2
iterations = 5001

xy = np.loadtxt('data2.csv', delimiter=',')
xy = xy[::-1]   # reversed

train_size = int(len(xy) * 0.7)
train_set = xy[0:train_size]
test_set = xy[train_size - seq_length:] 

def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length, :]
        _y = time_series[i + seq_length, [-1]]  # Next price       
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)

trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)

X = tf.placeholder(tf.float32, [None, seq_length, input_dim])
Y = tf.placeholder(tf.float32, [None, output_dim])

cells = []
for _ in range(0, num_hidden_layers):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim)
    cells.append(cell)                 
cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)
loss = tf.reduce_sum(tf.square(Y_pred - Y))
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(iterations):
    _, l= sess.run([train, loss], feed_dict={X:trainX, Y:trainY})
    if i % 500==0:
        print(i, l)

price = sess.run(Y_pred, feed_dict={X:testX})
plt.plot(testY)
plt.plot(price)
plt.show()