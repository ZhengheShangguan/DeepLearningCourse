import torch
import torch.utils.data
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
import os
import subprocess
from mpi4py import MPI
import torch.utils.data
import torch.optim as optim
import time
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import csv



# Code for initialization pytorch distributed 
cmd = "/sbin/ifconfig"
out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
    stderr=subprocess.PIPE).communicate()
ip = str(out).split("inet addr:")[1].split()[0]

name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_nodes = int(comm.Get_size())

ip = comm.gather(ip)

if rank != 0:
  ip = None

ip = comm.bcast(ip, root=0)

os.environ['MASTER_ADDR'] = ip[0]
os.environ['MASTER_PORT'] = '2222'

backend = 'mpi'
dist.init_process_group(backend, rank=rank, world_size=num_nodes)

dtype = torch.FloatTensor



# Setup dataset and loader
num_epochs = 30
batch_size = 128
# torch.manual_seed(0)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR100(root='~/scratch/', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR100(root='~/scratch/', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)




# Basic Block for ResNet
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet
    """
    def __init__(self, block, layers, num_classes=100, zero_init_residual=False):
        super(ResNet, self).__init__()

        self.in_channels = 32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.20)

        self.conv2_x = self._make_layer(block, 32, layers[0])
        self.conv3_x = self._make_layer(block, 64, layers[1], stride=2)
        self.conv4_x = self._make_layer(block, 128, layers[2], stride=2)
        self.conv5_x = self._make_layer(block, 256, layers[3], stride=2)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(256 * 2 * 2 * block.expansion, num_classes)


    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels,
                        out_channels=out_channels * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        # For the rest blocks, the stride is 1 according to the ppt
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)

        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def train(model, criterion, optimizer, scheduler, num_epochs):
    # lists as storage
    list_train_acc = []
    list_test_acc = []
    # prev_train_acc = 0.0
    # prev_test_acc = 0.0

    for epoch in range(1, 1 + num_epochs):
        total_correct = 0
        total = 0
        # Train the model
        model.train()
        for i, data in enumerate(trainloader, 0):
            # GPU data copy
            X_train_batch = Variable(data[0]).cuda()
            Y_train_batch = Variable(data[1]).cuda()

            # Forward pass
            outputs = model(X_train_batch)
            # Get loss
            loss = criterion(outputs, Y_train_batch)
            # Statistics
            _, predicted = torch.max(outputs.data, 1)
            total += Y_train_batch.size(0)
            total_correct += float(predicted.eq(Y_train_batch.data).sum() )

            # zero out grad
            optimizer.zero_grad()
            # Backward pass
            loss.backward()

            for param in model.parameters():
                #print(param.grad.data)
                tensor0 = param.grad.data.cpu()
                dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
                tensor0 /= float(num_nodes)
                param.grad.data = tensor0.cuda()

            if epoch > 16:
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        if state['step'] >= 1024:
                            state['step'] = 1000

            # Update the optimizer
            optimizer.step()

        # Update list_train_acc data
        curr_train_acc = total_correct / total
        list_train_acc.append(curr_train_acc)

        # Test the model
        model.eval()
        total_correct = 0
        total = 0

        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader, 0):
            # GPU data copy
            X_test_batch, Y_test_batch= Variable(X_test_batch, volatile=True).cuda(),Variable(Y_test_batch, volatile=True).cuda()

            # Forward pass
            outputs = model(X_test_batch)

            # Statistics
            _, predicted = torch.max(outputs.data, 1)
            total += Y_test_batch.size(0)
            total_correct += float(predicted.eq(Y_test_batch.data).sum() )

        # Update list_test_acc data
        curr_test_acc = total_correct / total
        list_test_acc.append(curr_test_acc)

        # Update learning rate
        scheduler.step()

        # Print
        print('Node: {} | Epoch: {} | Train Acc: {}% | Test Acc: {}%'.format(rank, epoch, curr_train_acc*100, curr_test_acc*100))

        # Save the model, only the model parameters
        if epoch % num_epochs == 0:
            print("===> Saving model...")
            torch.save(model.state_dict(), 'epoch-{}.ckpt'.format(epoch))

        # # Terminate Condition
        # if (curr_test_acc < prev_test_acc and prev_train_acc - prev_test_acc + 0.005 < curr_train_acc - curr_test_acc):
        #     print("The Test Acc begins to drop and the gap between Train and Test starts to increase (overfitting), thus terminate the Adam Optimizer")
        #     break
        # prev_train_acc = curr_train_acc
        # prev_test_acc = curr_test_acc

    return list_train_acc, list_test_acc




# Create ResNet
model = ResNet(BasicBlock, (2, 4, 4, 2))

#Make sure that all nodes have the same model
for param in model.parameters():
    tensor0 = param.data
    dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
    param.data = tensor0/np.sqrt(np.float(num_nodes))

model = model.cuda()
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))



# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


time1 = time.time()
# Train the model
list_train_acc, list_test_acc = train(model, criterion, optimizer, scheduler, num_epochs)
time2 = time.time()
print('Total Training Time: {}'.format(time2 - time1))
print("Finish Training and Testing")


# save the data to csv
if rank == 0:
    with open('list_train_acc-Data-Node0', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(list_train_acc)
    with open('list_test_acc-Data-Node0', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(list_test_acc)

if rank == 1:
    with open('list_train_acc-Data-Node1', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(list_train_acc)
    with open('list_test_acc-Data-Node1', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(list_test_acc)
