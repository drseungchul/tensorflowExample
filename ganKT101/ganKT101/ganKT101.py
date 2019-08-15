
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.cluster import KMeans


tf.set_random_seed(32608)
learning_rate = 0.1
num_iterations = 2000
num_iterations2 = 15
noise_size = 256
data_size = 24 
target_size = 17
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
    list_data, list_target = [], []
    for i in range(len(loaded_data)):
        l = []
        tmp = get_one_hot(day2idx[loaded_data[i,0]], num_day)      
        l.append(list(tmp.flat))
        tmp = get_one_hot(location2idx[loaded_data[i,2]], num_location)
        l.append(list(tmp.flat))           
        tmp = get_one_hot(direction2idx[loaded_data[i,3]], num_direction)
        l.append(list(tmp.flat))
        flatlist = [y for x in l for y in x]
        list_target.append(flatlist)
        list_data.append(loaded_data[i,5:])   
    return list_data, list_target

def get_targets(sample_size, target):
    list = []
    for _ in range(sample_size):
        list.append(target)
    return list


def get_noise(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

# generator
def generator(Z, target, hsize=[16, 16],reuse=False):
    with tf.variable_scope("Generator",reuse=reuse):
        Z = tf.concat([Z, target],1)
        h1 = tf.layers.dense(Z,hsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h2,data_size)
    return out

# discriminator
def discriminator(X, target, hsize=[16, 16],reuse=False):
    with tf.variable_scope("Discriminator",reuse=reuse):
        X = tf.concat([X, target],1)
        h1 = tf.layers.dense(X,hsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2,2)
        out = tf.layers.dense(h3,1)
    return out

tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, data_size])
Z = tf.placeholder(tf.float32, [None, noise_size])
T = tf.placeholder(tf.float32, [None, target_size])

G = generator(Z,T)
real = discriminator(X,T)
fake = discriminator(G,T,reuse=True)
loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real,labels=tf.ones_like(real)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=fake,labels=tf.zeros_like(fake)))
loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake,labels=tf.ones_like(fake)))
var_G = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Generator")
var_D = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Discriminator")
train_G = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_G,var_list = var_G) 
train_D = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_D,var_list = var_D) 


sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
x, t = load_data(filename)
z = get_noise(len(x), noise_size)
for i in range(num_iterations):
    #X_batch, T_batch = load_data(filename)
    #Z_batch = get_noise(len(X_batch), noise_size)
    X_batch, T_batch, Z_batch = x, t, z

    for _ in range(num_iterations2):
            _, dloss = sess.run([train_D, loss_D], feed_dict={X: X_batch, T:T_batch, Z: Z_batch})        
    for _ in range(num_iterations2):
            _, gloss = sess.run([train_G, loss_G], feed_dict={T:T_batch, Z: Z_batch})

    if i%1000 ==0:
        print ("Iterations:", i,"Discriminator loss:", dloss, " Generator loss:", gloss)

sample_size = 10
target = [0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0, 1,0]
targets = get_targets(sample_size, target)
noise = get_noise(sample_size, noise_size)
samples = sess.run(G, feed_dict={Z: noise, T:targets})

color = iter(cm.rainbow(np.linspace(0,1,sample_size)))
for i in range(sample_size):
    c = next(color)
    plt.plot(samples[i], color=c, marker='.', label=i)

means = []
for i in range(data_size):
    kmeans = KMeans(n_clusters=1).fit(samples[:,i].reshape(-1,1))
    C= kmeans.cluster_centers_
    means.extend(C)
plt.plot(means, color='b', marker='o', label='m')

plt.grid()
plt.show()




