import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
import torch.utils.data
import torch.optim as optim
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
# Loading the data
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


# trainset = torchvision.datasets.CIFAR100(root='~/scratch/', train=True, download=False, transform=transform_train)
trainset = torchvision.datasets.CIFAR100(root='./data/', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

# testset = torchvision.datasets.CIFAR100(root='~/scratch/', train=False, download=False, transform=transform_test)
testset = torchvision.datasets.CIFAR100(root='./data/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Pretrained ResNet-18
def resnet18(pretrained=True):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18'],model_dir='~/scratch/'))
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'],model_dir='./pretrained'))
    return model



# Helper function from Google Colab: https://colab.research.google.com/drive/1gJAAN3UI9005ecVmxPun5ZLCGu4YBtLo#scrollTo=9QAuxIQvoSDV
class AverageBase(object):
    def __init__(self, value=0):
        self.value = float(value) if value is not None else None
    
    def __str__(self):
        return str(round(self.value, 4))
    
    def __repr__(self):
        return self.value
    
    def __format__(self, fmt):
        return self.value.__format__(fmt)
    
    def __float__(self):
        return self.value

class RunTrainLoss(AverageBase):
    """
    Tracking the training loss with exponentially-decaying moving average.EMA
    """
    def __init__(self, alpha=0.99):
        super(RunTrainLoss, self).__init__(None)
        self.alpha = alpha

    def update(self, value):
        if self.value is None:
            self.value = float(value)
        else:
            self.value = (1 - self.alpha) * float(value) + self.alpha * self.value
        return self.value

class RunTestLoss(AverageBase):
    """
    Tracking the testing loss with cumulative moving average.CMA
    """
    def __init__(self, value=0, count=0):
        super(RunTestLoss, self).__init__(value)
        self.count = count
    
    def update(self, value):
        self.value = (self.value * self.count + float(value))
        self.count += 1
        self.value /= self.count
        return self.value


def train(model, criterion, optimizer, scheduler, num_epochs):
    # lists as storage
    list_train_loss = []
    list_test_loss = []
    list_train_acc = []
    list_test_acc = []
    # prev_train_acc = 0.0
    # prev_test_acc = 0.0

    # Upsample the input data for pretrained-model
    upsample = nn.Upsample(scale_factor=7, mode='bilinear')

    for epoch in range(1, 1 + num_epochs):
        total_correct = 0
        total = 0
        # Train the model
        model.train()
        train_loss = RunTrainLoss()
        for i, data in enumerate(trainloader, 0):
            # GPU data copy
            X_train_batch = data[0].to(device)
            Y_train_batch = data[1].to(device)

            # Upsample the input data for pretrained-model
            X_train_batch = upsample(X_train_batch)

            # Forward pass
            outputs = model(X_train_batch)
            # Get loss
            loss = criterion(outputs, Y_train_batch)
            # Statistics
            _, predicted = torch.max(outputs.data, 1)
            total += Y_train_batch.size(0)
            total_correct += (predicted == Y_train_batch).sum().item()

            # zero out grad
            optimizer.zero_grad()
            # Backward pass
            loss.backward()

            if epoch > 16:
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        if state['step'] >= 1024:
                            state['step'] = 1000

            # Update the optimizer
            optimizer.step()

            # update model
            train_loss.update(loss)

        # Update list_train_acc data and list_train_loss data
        curr_train_acc = total_correct / total
        list_train_acc.append(curr_train_acc*100)
        list_train_loss.append(train_loss.value)
        
        # Test the model
        model.eval()
        total_correct = 0
        total = 0
        test_loss = RunTestLoss()
        with torch.no_grad():
            for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader, 0):
                # GPU data copy
                X_test_batch, Y_test_batch= X_test_batch.to(device),Y_test_batch.to(device)

                # Upsample the input data for pretrained-model
                X_test_batch = upsample(X_test_batch)

                # Forward pass
                outputs = model(X_test_batch)

                # Get loss
                loss = criterion(outputs, Y_test_batch)
                # update loss
                test_loss.update(loss)

                # Statistics
                _, predicted = torch.max(outputs.data, 1)
                total += Y_test_batch.size(0)
                total_correct += (predicted == Y_test_batch).sum().item()

        # Update list_test_acc data and list_test_loss data
        curr_test_acc = total_correct / total
        list_test_acc.append(curr_test_acc*100)
        list_test_loss.append(test_loss.value)

        # Update learning rate
        scheduler.step()

        # Print
        print(f"Epoch: {epoch} | Train Loss: {train_loss.value} | Test Loss: {test_loss.value} | Train Acc: {curr_train_acc*100}% | Test Acc: {curr_test_acc*100}%")

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

    return list_train_loss, list_test_loss, list_train_acc, list_test_acc





# Create ResNet
model = resnet18(pretrained=True)
# If you just need to fine-tune the last layer, comment out the code below.
# for param in model.parameters():
#     param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 100)

model.to(device)

# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

time1 = time.time()
# Train the model
list_train_loss, list_test_loss, list_train_acc, list_test_acc = train(model, criterion, optimizer, scheduler, num_epochs)
time2 = time.time()
print(f"Pretrained ResNet-18 for CIFAR100::Total Training Time: {time2 - time1} seconds")
print("Finish Training and Testing")

# Plot
epochs = range(1, len(list_test_acc) + 1)
#plot the loss learning curve
plt.figure(figsize=(10,6))
plt.plot(epochs, list_train_loss, '-o', color='b',  label="Training Loss")
plt.plot(epochs, list_test_loss, '-o', color='r', label="Testing Loss")
plt.title("Pretrained ResNet-18 for CIFAR100::Learning Curve for Loss")
plt.xlabel("Epoch"), plt.ylabel("Loss for Training and Testing along different Epochs")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
#plot the accuracy learning curve
plt.figure(figsize=(10,6))
plt.plot(epochs, list_train_acc, '-o', color='b',  label="Training Accuracy(%)")
plt.plot(epochs, list_test_acc, '-o', color='r', label="Testing Accuracy(%)")
plt.title("Pretrained ResNet-18 for CIFAR100::Learning Curve for Accuracy")
plt.xlabel("Epoch"), plt.ylabel("Accuracy(%) for Training and Testing along different Epochs")
plt.legend(loc="best")
plt.tight_layout()
plt.show()


