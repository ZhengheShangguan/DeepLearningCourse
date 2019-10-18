import time
import matplotlib.pyplot as plt

# Input the data
list_test_acc_Node0 = [0.2233, 0.3481, 0.4293, 0.5041, 0.5322, 0.5323, 0.573, 0.542, 0.5849, 0.579, 0.5742, 0.6759,
                        0.6813, 0.6855, 0.6906, 0.6874, 0.6915, 0.6931, 0.6939, 0.6927, 0.6978, 0.6995, 
                        0.6987, 0.7004, 0.7009, 0.6995, 0.7012, 0.7003, 0.6986, 0.7002]
list_test_acc_Node1 = [0.2232,0.3449,0.4277,0.5069,0.5333,0.5298,0.5744,0.5409,0.5851,0.5784,0.5719,0.6762,
                        0.6824,0.6864,0.6927,0.6857,0.6913,0.693,0.692,0.694,0.6967,
                        0.7011,0.6989,0.7,0.7002,0.7002,0.7014,0.6975,0.6995,0.6987]
list_train_acc_Node0 = [0.1168, 0.2847, 0.4101, 0.4931, 0.5432, 0.5797, 0.6107, 0.6328, 0.6532, 0.6735, 0.6869, 0.7708, 
                        0.7982, 0.8123, 0.8244, 0.8380, 0.8467, 0.8588, 0.8648, 0.8736, 0.8826, 0.8963, 0.8991, 0.9024, 
                        0.9027, 0.9063, 0.9056, 0.9079, 0.9091, 0.9086]
list_train_acc_Node1 = [0.11628,0.28298,0.41478,0.49168,0.54734,0.5843,0.61116,0.63414,0.65402,0.67124,0.68276,
                        0.77092,0.8,0.81458,0.82516,0.83352,0.84922,0.85586,0.86656,0.87618,0.87982,
                        0.89498,0.89928,0.9013,0.90384,0.90606,0.90762,0.90634,0.9081,0.90892]

# Take the average data
list_test_acc = []
list_train_acc = []
for i in range(30):
    list_test_acc.append((list_test_acc_Node0[i] + list_test_acc_Node1[i]) / 2)
    list_train_acc.append((list_train_acc_Node0[i] + list_train_acc_Node1[i]) / 2)

# Plot for the average of Node0 and Node1
epochs = range(1, len(list_test_acc) + 1)
#plot the accuracy learning curve
plt.figure(figsize=(10,6))
plt.plot(epochs, list_train_acc, '-o', color='b',  label="Training Accuracy")
plt.plot(epochs, list_test_acc, '-o', color='r', label="Testing Accuracy")
plt.title("Synchronous ResNet-18 for CIFAR100::Learning Curve for Accuracy")
plt.xlabel("Epoch"), plt.ylabel("Accuracy for Training and Testing along different Epochs")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
