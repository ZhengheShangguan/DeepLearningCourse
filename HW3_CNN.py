import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Define configuration parameters
dataroot = './'
batch_size_train = 128
batch_size_test = 64
LR = 0.001
wd = 0.0005
scheduler_step_size = 8
scheduler_gamma = 0.1
num_epochs = 30

# Define data augumentation
# For train set, random flip, rotate, crop and change brightness and contrast
transforms_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.RandomCrop(32, padding = 4),
    # transforms.ColorJitter(brightness=0.05, contrast=0.05),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
# For test set, only normalize the dataset without augumentation
transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load datasets
trainset = torchvision.datasets.CIFAR10(
    root=dataroot, train=True, download=True, transform=transforms_train
)
testset = torchvision.datasets.CIFAR10(
    root=dataroot, train=False, download=True, transform=transforms_test
)
trainloader = torch.utils.data.DataLoader(
    dataset=trainset, batch_size=batch_size_train, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    dataset=testset, batch_size=batch_size_test, shuffle=False
)

# Define CNN with given configurations in Lecture 6
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(
            
            # Conv Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.Dropout2d(p=0.05),

            # Conv Block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.Dropout2d(p=0.05),

            # Conv Block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=2),

        )

        self.fc_layer = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        """Forword propagation"""
        # conv
        x = self.conv_layer(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc
        x = self.fc_layer(x)
        return x



# Define Loss Function, optimizer, step scheduler
model = CNN()
model = model.cuda()
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=wd)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

# Train the model, while also print the current test accuracy
import time
import matplotlib.pyplot as plt
time1 = time.time()
prev_train_acc = 0.0
list_train_acc = []
list_test_acc = []
epoch_sizes = []
for epochs in range(num_epochs):
    total_correct = 0
    total = 0
    # Train an epoch
    for i, data in enumerate(trainloader, 0):
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        # Backward and Optimize
        optimizer.zero_grad()
        loss.backward()

        if epochs > 16:
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if state['step'] >= 1024:
                        state['step'] = 1000

        optimizer.step()
    
    scheduler.step()

    curr_train_acc = total_correct / total

    # Current Test Acc
    with torch.no_grad():
        total_correct = 0
        total = 0
        for i, data in enumerate(testloader, 0):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    curr_test_acc = total_correct / total

    # Print
    print(f"Epoch: {epochs} | Loss: {loss.item()} | Train Acc: {curr_train_acc} | Test Acc: {curr_test_acc}")

    # Save the model, only the model parameters
    if (epochs + 1) % (num_epochs + 1) == 0:
        print("===> Saving model...")
        torch.save(model.state_dict(), 'epoch-{}.ckpt'.format(epochs))

    # Terminate Condition
    if (abs(curr_train_acc - prev_train_acc) < 0.001):
        print("Two consecutive loss is too close(< 0.1%), thus terminate the Adam Optimizer")
        break
    prev_train_acc = curr_train_acc    

    # for learning curve plot
    epoch_sizes.append(epochs + 1)
    list_train_acc.append(curr_train_acc)
    list_test_acc.append(curr_test_acc)


#Time for training
time2 = time.time()
time_tol = time2 - time1
print(f"Total Training Time: {time_tol}")
print("Finished Training...")

#plot the learning curve
plt.plot(epoch_sizes, list_train_acc, '-', color='b',  label="Training Acc")
plt.plot(epoch_sizes, list_test_acc, '--', color='r', label="Testing Acc")
# Create plot
plt.title("Learning Curve")
plt.xlabel("Epoch"), plt.ylabel("Accuracy for Training and Testing along different Epochs"), plt.legend(loc="best")
plt.tight_layout()
plt.show()



