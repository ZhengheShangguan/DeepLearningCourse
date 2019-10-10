import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import copy
from random import randint
#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()
#Implementation of stochastic gradient descent algorithm
#number of inputs
num_inputs = 28*28
#number of hidden layer units
num_hiddens = 50
#number of outputs
num_outputs = 10
model = {}
model['W1'] = np.random.randn(num_hiddens,num_inputs) / np.sqrt(num_inputs)
model['b1'] = np.zeros((num_hiddens,1))
model['C'] = np.random.randn(num_outputs,num_hiddens) / np.sqrt(num_hiddens)
model['b2'] = np.zeros((num_outputs,1))
def indicator_function(y, num_output):
    vec = np.zeros((num_output, 1))
    vec[y] = 1
    return vec
def softmax_function(z):
    z_max = np.max(z)
    z_new = z - z_max
    ZZ = np.exp(z_new) / np.sum(np.exp(z_new), axis = 0, keepdims = True)
    return ZZ
def forward(x, model):
    x_new = x.reshape(-1, 1)
    Z1 = np.dot(model['W1'], x_new) + model['b1']
    H1 = np.tanh(Z1)
    U = np.dot(model['C'], H1) + model['b2']
    p = softmax_function(U)

    # Also store the cache for backprop usage
    cache = {"Z1": Z1,
            "H1": H1,
            "U": U,
            "p": p}

    return cache
def backward(x, y, cache, model):
    # Retrive some model parameter values
    W1 = model['W1']
    C = model['C']

    # Retrive some cache values
    H1 = cache["H1"]
    p = cache["p"]

    # Backward prop calculations
    dU = p - indicator_function(y, num_outputs)
    dC = np.dot(dU, H1.T)
    db2 = dU
    dZ1 = np.dot(C.T, dU) * (1 - np.power(H1, 2))
    dW1 = np.dot(dZ1, x.reshape(1, -1))
    db1 = dZ1

    model_grads = {"dW1": dW1,
                    "db1": db1,
                    "dC": dC,
                    "db2": db2}

    return model_grads

import time
time1 = time.time()
LR = .01
num_epochs = 20
prev_train_acc = 0.0
list_train_acc = []
list_test_acc = []
epoch_sizes = []
for epochs in range(num_epochs):
    #Learning rate schedule
    if (epochs > 5):
        LR = 0.001
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001
    total_correct = 0
    for n in range( len(x_train)):
        n_random = randint(0,len(x_train)-1 )
        y = y_train[n_random]
        x = x_train[n_random][:]
        cache = forward(x, model)
        prediction = np.argmax(cache["p"])
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x, y, cache, model)
        model['W1'] = model['W1'] - LR*model_grads["dW1"]
        model['b1'] = model['b1'] - LR*model_grads["db1"]
        model['C'] = model['C'] - LR*model_grads["dC"]
        model['b2'] = model['b2'] - LR*model_grads["db2"]
    curr_train_acc = total_correct/np.float(len(x_train) )
    print(curr_train_acc)
    if (abs(curr_train_acc - prev_train_acc) < 0.0001):
        print("Two consecutive loss is too close(< 0.01%), thus terminate the SGD")
        break
    prev_train_acc = curr_train_acc

    # for learning curve plot
    epoch_sizes.append(epochs + 1)
    # after each epoch, save the accuracy of training data
    list_train_acc.append(curr_train_acc)
    # for learning curve plot
    # after each epoch, save the accuracy of testing data
    total_curr_correct = 0
    for n in range( len(x_test)):
        y = y_test[n]
        x = x_test[n][:]
        cache = forward(x, model)
        prediction = np.argmax(cache["p"])
        if (prediction == y):
            total_curr_correct += 1
    list_test_acc.append(total_curr_correct/np.float(len(x_test) ) )

#time for training
time2 = time.time()
print(time2-time1)

#test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    cache = forward(x, model)
    prediction = np.argmax(cache["p"])
    if (prediction == y):
        total_correct += 1
print(total_correct/np.float(len(x_test) ) )

#plot the learning curve
plt.plot(epoch_sizes, list_train_acc, '-', color='b',  label="Training Acc")
plt.plot(epoch_sizes, list_test_acc, '--', color='r', label="Testing Acc")
# Create plot
plt.title("Learning Curve")
plt.xlabel("Epoch"), plt.ylabel("Accuracy for Training and Testing along different Epochs"), plt.legend(loc="best")
plt.tight_layout()
plt.show()