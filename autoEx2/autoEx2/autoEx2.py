import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

all_images = np.loadtxt('./fashionmnist/fashion-mnist_train.csv',delimiter=',', skiprows=1)[:,1:]
#print(all_images.shape)
#plt.imshow(all_images[0].reshape(28,28),  cmap='Greys')
#plt.show()

n_nodes_inpl = 784  #encoder
n_nodes_hl1  = 32  #encoder
n_nodes_hl2  = 32  #decoder
n_nodes_outl = 784  #decoder
learn_rate = 0.1
batch_size = 100  # how many images to use together for training
hm_epochs =10    # how many times to go through the entire dataset
tot_images = 60000

hidden_1_layer_vals = {
'weights':tf.Variable(tf.random_normal([n_nodes_inpl,n_nodes_hl1])),
'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))  }
hidden_2_layer_vals = {
'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))  }
output_layer_vals = {
'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_outl])),
'biases':tf.Variable(tf.random_normal([n_nodes_outl])) }

input_layer = tf.placeholder('float', [None, 784])
layer_1 = tf.nn.sigmoid(tf.matmul(input_layer,hidden_1_layer_vals['weights']) + hidden_1_layer_vals['biases'])
layer_2 = tf.nn.sigmoid(tf.matmul(layer_1,hidden_2_layer_vals['weights']) + hidden_2_layer_vals['biases'])
output_layer = tf.matmul(layer_2,output_layer_vals['weights']) + output_layer_vals['biases']
output_true = tf.placeholder('float', [None, 784])

cost = tf.reduce_mean(tf.square(output_layer - output_true))
optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(hm_epochs):
    epoch_loss = 0    # initializing error as 0
    for i in range(int(tot_images/batch_size)):
        epoch_x = all_images[ i*batch_size : (i+1)*batch_size ]
        _, c = sess.run([optimizer, cost], feed_dict={input_layer: epoch_x, output_true: epoch_x})
        epoch_loss += c
print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)

any_image = all_images[218]
output_any_image = sess.run(output_layer,feed_dict={input_layer:[any_image]})
encoded_any_image = sess.run(layer_1, feed_dict={input_layer:[any_image]})
plt.imshow(all_images[218].reshape(28,28),  cmap='Greys')
plt.show()
print(encoded_any_image)

