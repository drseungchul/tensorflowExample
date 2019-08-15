
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


tf.set_random_seed(32608)
learning_rate = 0.01
num_iterations = 50
num_iterations2 = 2
noise_size = 128
data_size = 41 #24
filename = './data/traindata.csv'
num_day = 7
num_location = 8
num_direction = 2


def get_one_hot(target, nb_classes):
    t =np.array(target).reshape(-1)
    res = np.eye(nb_classes)[np.array(t).reshape(-1)]
    return res.reshape(list(t.shape)+[nb_classes])

def load_data(filename):
    print("training with", filename)     
    
    idx2day = ["일", "월", "화","수", "목","금","토"]
    day2idx = {c: i for i, c in enumerate(idx2day)}
    idx2location = ["A-01", "A-02", "A-03","A-04", "A-06","A-07","A-08","A-10"]
    location2idx = {c: i for i, c in enumerate(idx2location)}
    idx2direction = ["유입", "유출"]
    direction2idx = {c: i for i, c in enumerate(idx2direction)}        

    loaded_data = np.loadtxt(filename, delimiter=',', dtype=str, skiprows=1)
    data = []
    for i in range(len(loaded_data)):
        l = []
        tmp = get_one_hot(day2idx[loaded_data[i,0]], num_day)      
        l.append(list(tmp.flat))
        tmp = get_one_hot(location2idx[loaded_data[i,2]], num_location)
        l.append(list(tmp.flat))           
        tmp = get_one_hot(direction2idx[loaded_data[i,3]], num_direction)
        l.append(list(tmp.flat))
        l.append(loaded_data[i,5:])
        flatlist = [y for x in l for y in x]
        data.append(flatlist)
    
    return data

def get_noise(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def generator(Z, hsize=[16, 16],reuse=False):
    with tf.variable_scope("Generator",reuse=reuse):
        #Z = tf.concat([Z, target],1)
        h1 = tf.layers.dense(Z,hsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h2,data_size)
    return out

def discriminator(X, hsize=[16, 16],reuse=False):
    with tf.variable_scope("Discriminator",reuse=reuse):
        #X = tf.concat([X, target],1)
        h1 = tf.layers.dense(X,hsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2,2*data_size)
        out = tf.layers.dense(h3,data_size)
    return out

tf.reset_default_graph()
X = tf.placeholder(tf.float32,[None,data_size])
Z = tf.placeholder(tf.float32,[None,noise_size])

G = generator(Z)
real = discriminator(X)
fake = discriminator(G,reuse=True)
loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real,labels=tf.ones_like(real)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=fake,labels=tf.zeros_like(fake)))
loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake,labels=tf.ones_like(fake)))
var_G = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Generator")
var_D = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Discriminator")
train_G = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_G,var_list = var_G) 
train_D = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_D,var_list = var_D) 





sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
data = load_data(filename)
for i in range(num_iterations):
    X_batch = data
    Z_batch = get_noise(len(data), noise_size)

    for _ in range(num_iterations2):
        _, dloss = sess.run([train_D, loss_D], feed_dict={X: X_batch, Z: Z_batch})
    for _ in range(num_iterations2):
        _, gloss = sess.run([train_G, loss_G], feed_dict={Z: Z_batch})

    if i%10 ==0:
        print ("Iterations:", i,"Discriminator loss:", dloss, " Generator loss:", gloss)
