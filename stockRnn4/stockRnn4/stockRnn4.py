# made by dr.seungchul@gmail.com

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import util
from RNNmachine import *

# parameters
seq_length = 5
hidden_dim = 10
num_hidden_layers = 10
learning_rate = 0.01
num_iterations = 10000
trainFile = './data/traindata.csv'
testFile = './data/data.csv'

input_dim = 4
output_dim = 1

loaded_data = np.loadtxt(trainFile, delimiter=',', skiprows=1)
loaded_data = loaded_data[::-1]   # reverse

# normalize   
col0 = loaded_data[:,0]
col1 = loaded_data[:,1]
col2 = loaded_data[:,2]
col3 = loaded_data[:,3]
xmax = col3.max()
xmin = col3.min()
col0 = util.minmax_normalize(col0)
col1 = util.minmax_normalize(col1)
col2 = util.minmax_normalize(col2)
col3 = util.minmax_normalize(col3)
for i in range(0,len(loaded_data)):
    loaded_data[i,0] = col0[i]
    loaded_data[i,1] = col1[i]
    loaded_data[i,2] = col2[i]
    loaded_data[i,3] = col3[i]    

dataX = []
dataT = []
for i in range(0, len(loaded_data)-seq_length):
    _x = loaded_data[i:i+seq_length,:]
    _t = loaded_data[i+seq_length,[-1]]
    dataX.append(_x)
    dataT.append(_t)

m0 = RNNmachine(0, input_dim, seq_length, hidden_dim, num_hidden_layers, output_dim, learning_rate, num_iterations)
m0.doTrain(dataX, dataT)


loaded_data2 = np.loadtxt(testFile, delimiter=',', skiprows=1)
loaded_data2 = loaded_data2[::-1]   # reverse
col0 = loaded_data2[:,0]
col1 = loaded_data2[:,1]
col2 = loaded_data2[:,2]
col3 = loaded_data2[:,3]
col0 = util.minmax_normalize(col0)
col1 = util.minmax_normalize(col1)
col2 = util.minmax_normalize(col2)
col3 = util.minmax_normalize(col3)
for i in range(0,len(loaded_data2)):
    loaded_data2[i,0] = col0[i]
    loaded_data2[i,1] = col1[i]
    loaded_data2[i,2] = col2[i]
    loaded_data2[i,3] = col3[i]    

data = []
for i in range(0, len(loaded_data2)-seq_length):
    _x = loaded_data2[i:i+seq_length,:]
    data.append(_x)

pred = m0.doPredict(data)
for i in range(0, len(pred)):
    pred_denorm = util.minmax_get_denorm(pred[i], xmax, xmin)
    print(pred_denorm)

plt.plot(dataT, color='b', marker='o')
plt.plot(pred, color='r', marker='x')
plt.grid()
plt.show()



