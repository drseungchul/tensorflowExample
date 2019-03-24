#tensorboard --logdir==training:logs --host=127.0.0.1:6006
import tensorflow as tf
import numpy as np

n_classes = 7  # 0 ~ 6

xy = np.loadtxt('data-041.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])  
Y_one_hot = tf.one_hot(Y, n_classes)  # one hot
Y_one_hot = tf.reshape(Y_one_hot, [-1, n_classes])

with tf.name_scope('layer1'):
     W1 = tf.Variable(tf.random_normal([16, 32]), name='weight1')
     b1 = tf.Variable(tf.random_normal([32]), name='bias')
     L1 = tf.add(tf.matmul(X, W1), b1)

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_normal([32, n_classes]), name='weight2')
    b2 = tf.Variable(tf.random_normal([n_classes]), name='bias2')

with tf.name_scope('optimizer'):
    logits = tf.matmul(L1, W2) + b2
    hypothesis = tf.nn.softmax(logits)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y_one_hot))                                                              
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={X: x_data, Y: y_data})
                                        
        if step % 400 == 0:
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
