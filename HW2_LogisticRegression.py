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
num_xdimx = 28
num_xdimy = 28
#dimension of filters
num_k = 3
num_k = 3
num_c = 4
#number of outputs
num_outputs = 10
#initialize the model parameters with Xivier Initialization
model = {}
model['K'] = np.random.randn(num_k,num_k,num_c)
for i in range(num_c):
    model['K'][:,:,i] /= np.sqrt(num_k * num_k)
model['W'] = np.random.randn(num_outputs, num_xdimy - num_k + 1, num_xdimx - num_k + 1, num_c)
for i in range(num_outputs):
    model['W'][i,:,:,:] /= np.sqrt((num_xdimy - num_k + 1)*(num_xdimx - num_k + 1)*num_c)
model['b'] = np.zeros((num_outputs, 1))

def conv_single_layer(x_new, K_filter):
    # Retrieve the dim
    (n_x_dimy, n_x_dimx) = x_new.shape
    (n_k_dimy, n_k_dimx, n_k_c) = K_filter.shape

    # initialize variables
    Z = np.zeros((n_x_dimy - n_k_dimy + 1, n_x_dimx - n_k_dimx + 1, n_k_c))

    # use stack to save some reused memory of x_new
    stack = np.zeros((n_x_dimy - n_k_dimy + 1, n_x_dimx - n_k_dimx + 1, n_k_dimy, n_k_dimx))
    for h in range(n_x_dimy - n_k_dimy + 1):
        for w in range(n_x_dimx - n_k_dimx + 1):
            stack[h, w, :, :] = x_new[h:(h+n_k_dimy), w:(w+n_k_dimx)]

    # run the convolution for multi-channels
    # PS!!!: use np.tensordot to calculate the single layer within one stroke
    for n_c in range(n_k_c):
        Z[:, :, n_c] = np.tensordot(stack, K_filter[:, :, n_c], axes=([2, 3], [0, 1]))
    H = sigmoid(Z)
    return Z, H
def conv_single_layer_single_channel(x_new, K_filter):
    # Retrieve the dim
    (n_x_dimy, n_x_dimx) = x_new.shape
    (n_k_dimy, n_k_dimx) = K_filter.shape
    stack = np.zeros((n_x_dimy - n_k_dimy + 1, n_x_dimx - n_k_dimx + 1, n_k_dimy, n_k_dimx))
    for h in range(n_x_dimy - n_k_dimy + 1):
        for w in range(n_x_dimx - n_k_dimx + 1):
            stack[h, w, :, :] = x_new[h:(h+n_k_dimy), w:(w+n_k_dimx)]
    Z = np.tensordot(stack, K_filter, axes=([2, 3], [0, 1]))
    return Z
def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm
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
    x_new = x.reshape(num_xdimy, num_xdimx)
    # conv for x_new for the first layer
    Z, H = conv_single_layer(x_new, model['K'])
    # calculate U
    U = np.zeros((num_outputs, 1))
    for n_out in range(num_outputs):
        U[n_out] = np.sum(np.multiply(model['W'][n_out,:,:,:], H)) + model['b'][n_out]
    # calculate p
    p = softmax_function(U)
    # Also store the cache for backprop usage
    cache = {"Z": Z,
             "H": H,
             "U": U,
             "p": p}
    return cache
def backward(x, y, cache, model):
    # Retrive some model and cache values
    K = model['K']
    W = model['W']
    b = model['b']
    H = cache["H"]
    Z = cache["Z"]
    p = cache["p"]

    # Backward prop calculations, where W and H should be reshaped for the last layer
    dU = p - indicator_function(y, num_outputs)
    dW = np.zeros((num_outputs, num_xdimy - num_k + 1, num_xdimx - num_k + 1, num_c))
    for i in range(num_outputs):
        dW[i,:,:,:] = np.multiply(dU[i], H)
    db = dU
    dH = np.tensordot(np.squeeze(dU), W, axes=([0],[0]))
    # calc dK
    dK = np.zeros((num_k, num_k, num_c))
    x_new = x.reshape(num_xdimy, num_xdimx)
    val_tmp = np.multiply(sigmoid(Z, True), dH)
    for p in range(num_c):
        dK[:,:,p] = conv_single_layer_single_channel(x_new, val_tmp[:,:,p])

    model_grads = {"dW": dW,
                   "db": db,
                   "dK": dK}

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
        model['W'] = model['W'] - LR*model_grads["dW"]
        model['b'] = model['b'] - LR*model_grads["db"]
        model['K'] = model['K'] - LR*model_grads["dK"]
    curr_train_acc = total_correct/np.float(len(x_train) )
    print(f"Epoch: {epochs} LR: {LR} Training Acc: {curr_train_acc}")
    if (abs(curr_train_acc - prev_train_acc) < 0.001):
        print("Two consecutive loss is too close(< 0.1%), thus terminate the SGD")
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
time_tol = time2 - time1
print(f"Training Time: {time_tol}")

#test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    cache = forward(x, model)
    prediction = np.argmax(cache["p"])
    if (prediction == y):
        total_correct += 1
test_res = total_correct/np.float(len(x_test) )
print(f"Test Acc: {test_res}" )

#plot the learning curve
plt.plot(epoch_sizes, list_train_acc, '-', color='b',  label="Training Acc")
plt.plot(epoch_sizes, list_test_acc, '--', color='r', label="Testing Acc")
# Create plot
plt.title("Learning Curve")
plt.xlabel("Epoch"), plt.ylabel("Accuracy for Training and Testing along different Epochs"), plt.legend(loc="best")
plt.tight_layout()
plt.show()