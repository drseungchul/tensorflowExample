import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint
import matplotlib.pyplot as plt

char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# {'a': 0, 'b': 1, 'c': 2, ..., 'j': 9, 'k', 10, ...}
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']

dataX = []
dataY = []
for seq in seq_data:
    # [22, 14, 17] [22, 14, 14] [3, 4, 4] [3, 8, 21] ...
    input = [num_dic[n] for n in seq[:-1]]
    target = num_dic[seq[-1]]
    # one-hot 인코딩을 합니다.
    # if input is [0, 1, 2]:
    # [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    #  [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]]
    dataX.append(np.eye(dic_len)[input])
    # 지금까지 손실함수로 사용하던 softmax_cross_entropy_with_logits 함수는
    # label 값을 one-hot 인코딩으로 넘겨줘야 하지만,
    # 이 예제에서 사용할 손실 함수인 sparse_softmax_cross_entropy_with_logits 는
    # one-hot 인코딩을 사용하지 않으므로 index 를 그냥 넘겨주면 됩니다.
    dataY.append(target)

input_dim = dic_len
seq_length = 3
hidden_dim = 128
output_dim = dic_len
learning_rate = 0.01
iterations = 501

X = tf.placeholder(tf.float32, [None, seq_length, input_dim])
Y = tf.placeholder(tf.int32, [None])
W = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
b = tf.Variable(tf.random_normal([output_dim]))

cell1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
# 최종 결과는 one-hot 인코딩 형식으로 만듭니다
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


######
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(iterations):
    _, loss = sess.run([optimizer, cost], feed_dict={X:dataX, Y:dataY})
    if i % 100==0:
        print(i, 'cost =', loss)


prediction = tf.cast(tf.argmax(model, 1), tf.int32)
prediction_check = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))
predict, accuracy_val = sess.run([prediction, accuracy], feed_dict={X: dataX, Y:dataY})
print('입력값:', [w[:3] + ' ' for w in seq_data])
for idx, val in enumerate(seq_data):
    print(char_arr[predict[idx]])



