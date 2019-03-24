
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint
import matplotlib.pyplot as plt

def MinMaxScaler(data):   
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

input_dim = 4
seq_length = 5
hidden_dim = 100
num_hidden_layers = 2
output_dim = 1
learning_rate = 0.01
iterations = 2501

xy = np.loadtxt('./data/data2.csv', delimiter=',', skiprows=1)
xy = xy[::-1]   # reversed
dataX = []
dataY = []
for i in range(0, len(xy)-seq_length):
    _x = xy[i:i+seq_length,:]
    _y = xy[i+seq_length,[-1]]
    dataX.append(_x)
    dataY.append(_y)
dataX = MinMaxScaler(dataX)
dataY = MinMaxScaler(dataY)

X = tf.placeholder(tf.float32, [None, seq_length, input_dim])
Y = tf.placeholder(tf.float32, [None, 1])

#for softmax 
#W = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
#b = tf.Variable(tf.random_normal([output_dim]))

#dropout
#cell1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
#cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
#cell2 = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
#multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

cells = []
for _ in range(0, num_hidden_layers):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim)
    cells.append(cell)                 
multi_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

outputs, _states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)
loss = tf.reduce_sum(tf.square(Y_pred - Y))
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

#for softmax
#outputs = tf.transpose(outputs, [1, 0, 2])
#outputs = outputs[-1]
#model = tf.matmul(outputs, W) + b
#sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
#loss = tf.reduce_mean(sequence_loss) 
#train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#prediction = tf.argmax(outputs, axis=2)

#################################################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(iterations):
    _, l= sess.run([train, loss], feed_dict={X:dataX, Y:dataY})
    if i % 500==0:
        print(i," loss = ", l)

price = sess.run(Y_pred, feed_dict={X:dataX})
plt.plot(dataY)
plt.plot(price)
plt.show()
