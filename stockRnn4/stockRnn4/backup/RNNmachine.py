import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import util

class RNNmachine(object):
    def __init__(self, id, input_dim, seq_length, hidden_dim, num_hidden_layers, output_dim, learning_rate, num_iterations):        
        self.id = id
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        print("new RNN machine created ",self.id)

    def doTrain(self, datax, datat):
        print("training ....")        
        tf.reset_default_graph()
        self.dataX = datax
        self.dataT = datat

        self.X = tf.placeholder(tf.float32, [None, self.seq_length, self.input_dim])
        self.T = tf.placeholder(tf.float32, [None, self.output_dim])
        self.cells = []
        for _ in range(0, self.num_hidden_layers):
            self.cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_dim)
            self.cells.append(self.cell)                 
        self.multi_cell = tf.contrib.rnn.MultiRNNCell(self.cells, state_is_tuple=True)
        self.outputs, _states = tf.nn.dynamic_rnn(self.multi_cell, self.X, dtype=tf.float32)
        self.Y = tf.contrib.layers.fully_connected(self.outputs[:, -1], self.output_dim, activation_fn=None)
        self.loss = tf.reduce_sum(tf.square(self.Y - self.T))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train = self.optimizer.minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        for step in range(self.num_iterations):
            _, l= self.sess.run([self.train, self.loss], feed_dict={self.X:self.dataX, self.T:self.dataT})
            if step % 500==0:
                print(step," loss = ", l)

    def doPredict(self, datax):
        return self.sess.run(self.Y, feed_dict={self.X:datax})
