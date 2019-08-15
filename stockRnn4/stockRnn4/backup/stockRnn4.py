import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import util
from RNNmachine import *

input_dim = 4
seq_length = 5
hidden_dim = 5
num_hidden_layers = 2
output_dim = 1
learning_rate = 0.01
num_iterations = 500
filename = './data/data3.csv'

loaded_data = np.loadtxt(filename, delimiter=',', skiprows=1)
loaded_data = loaded_data[::-1]   # reverse

# normalize   
col0 = loaded_data[:,0]
col1 = loaded_data[:,1]
col2 = loaded_data[:,2]
col3 = loaded_data[:,3]
xmax = col3.max()
xmin = col3.min()
print(xmax, xmin)
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

data = [[ [0.0, 0.7, 0.3, 0.4],
         [0.5, 0.77, 0.3, 0.2],
         [0.2, 0.5, 0.6, 0],
         [0.9, 0.4, 0.8, 0.1],
         [0.5, 0.3, 0.9, 0.6] ]]

pred = m0.doPredict(data)
pred_denorm = util.minmax_get_denorm(pred, xmax, xmin)
print(pred_denorm)

plt.plot(dataT, color='b', marker='o')
plt.plot(pred, color='r', marker='x')
plt.grid()
plt.show()



