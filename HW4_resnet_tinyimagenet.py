import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


num_epochs = 60
batch_size = 128

transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
])

def create_val_folder(val_dir):
    """
    This method is responsible for separating validation images into separate sub folders
    """
    path = os.path.join(val_dir, 'images')  # path where validation data is present now
    filename = os.path.join(val_dir, 'val_annotations.txt')  # file where image2class mapping is present
    fp = open(filename, "r")  # open file in read mode
    data = fp.readlines()  # read line by line

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()
    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath):  # check if folder exists
            os.makedirs(newpath)
        if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
            os.rename(os.path.join(path, img), os.path.join(newpath, img))
    return

# Your own directory to the train folder of tiyimagenet
#train_dir = '/u/training/instr030/scratch/tiny-imagenet-200/train/'
train_dir = './data/tiny-imagenet-200/train/'
train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
# To check the index for each classes
# print(train_dataset.class_to_idx)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
# Your own directory to the validation folder of tiyimagenet
#val_dir = '/u/training/instr030/scratch/tiny-imagenet-200/val/'
val_dir = './data/tiny-imagenet-200/val/'

if 'val_' in os.listdir(val_dir+'images/')[0]:
    create_val_folder(val_dir)
    val_dir = val_dir+'images/'
else:
    val_dir = val_dir+'images/'


val_dataset = datasets.ImageFolder(val_dir, transform=transforms.ToTensor())
# To check the index for each classes
# print(val_dataset.class_to_idx)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)


# YOUR CODE GOES HERE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ResNet
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
    def __init__(self, block, layers, num_classes=200, zero_init_residual=False):
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
        self.fc = nn.Linear(256 * 4 * 4 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)      
        

        # Zero-initialize the last BN in each residual branch, improves 0.2~0.3% 
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


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

class RunValLoss(AverageBase):
    """
    Tracking the Validating loss with cumulative moving average.CMA
    """
    def __init__(self, value=0, count=0):
        super(RunValLoss, self).__init__(value)
        self.count = count
    
    def update(self, value):
        self.value = (self.value * self.count + float(value))
        self.count += 1
        self.value /= self.count
        return self.value


def train(model, criterion, optimizer, scheduler, num_epochs):
    # lists as storage
    list_train_loss = []
    list_val_loss = []
    list_train_acc = []
    list_val_acc = []
    # prev_train_acc = 0.0
    # prev_val_acc = 0.0

    for epoch in range(1, 1 + num_epochs):
        total_correct = 0
        total = 0
        # Train the model
        model.train()
        train_loss = RunTrainLoss()
        for i, data in enumerate(train_loader, 0):
            # GPU data copy
            X_train_batch = data[0].to(device)
            Y_train_batch = data[1].to(device)

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
        list_train_acc.append(curr_train_acc)
        list_train_loss.append(train_loss.value)
        
        # Validate the model
        model.eval()
        total_correct = 0
        total = 0
        val_loss = RunValLoss()
        with torch.no_grad():
            for batch_idx, (X_val_batch, Y_val_batch) in enumerate(val_loader, 0):
                # GPU data copy
                X_val_batch, Y_val_batch= X_val_batch.to(device),Y_val_batch.to(device)

                # Forward pass
                outputs = model(X_val_batch)

                # Get loss
                loss = criterion(outputs, Y_val_batch)
                # update loss
                val_loss.update(loss)

                # Statistics
                _, predicted = torch.max(outputs.data, 1)
                total += Y_val_batch.size(0)
                total_correct += (predicted == Y_val_batch).sum().item()

        # Update list_val_acc data and list_val_loss data
        curr_val_acc = total_correct / total
        list_val_acc.append(curr_val_acc)
        list_val_loss.append(val_loss.value)

        # Update learning rate
        scheduler.step()

        # Print
        print(f"Epoch: {epoch} | Train Loss: {train_loss.value} | Validate Loss: {val_loss.value} | Train Acc: {curr_train_acc*100}% | Validate Acc: {curr_val_acc*100}%")

        # Save the model, only the model parameters
        if epoch % num_epochs == 0:
            print("===> Saving model...")
            torch.save(model.state_dict(), 'epoch-{}.ckpt'.format(epoch))

        # # Terminate Condition
        # if (curr_val_acc < prev_val_acc and prev_train_acc - prev_val_acc + 0.005 < curr_train_acc - curr_val_acc):
        #     print("The Validate Acc begins to drop and the gap between Train and Validate starts to increase (overfitting), thus terminate the Adam Optimizer")
        #     break
        # prev_train_acc = curr_train_acc
        # prev_val_acc = curr_val_acc

    return list_train_loss, list_val_loss, list_train_acc, list_val_acc





# Create ResNet
model = ResNet(BasicBlock, (2, 4, 4, 2))
model.to(device)

# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

time1 = time.time()
# Train the model
list_train_loss, list_val_loss, list_train_acc, list_val_acc = train(model, criterion, optimizer, scheduler, num_epochs)
time2 = time.time()
print(f"Tiny-ImageNet::Total Training Time: {time2 - time1}")
print("Tiny-ImageNet::Finish Training and Validating")

# Plot
epochs = range(1, len(list_val_acc) + 1)
#plot the loss learning curve
plt.figure(figsize=(10,6))
plt.plot(epochs, list_train_loss, '-o', color='b',  label="Training Loss")
plt.plot(epochs, list_val_loss, '-o', color='r', label="Validating Loss")
plt.title("Tiny-ImageNet::Learning Curve for Loss")
plt.xlabel("Epoch"), plt.ylabel("Loss for Training and Validating along different Epochs")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
#plot the accuracy learning curve
plt.figure(figsize=(10,6))
plt.plot(epochs, list_train_acc, '-o', color='b',  label="Training Accuracy")
plt.plot(epochs, list_val_acc, '-o', color='r', label="Validating Accuracy")
plt.title("Tiny-ImageNet::Learning Curve for Accuracy")
plt.xlabel("Epoch"), plt.ylabel("Accuracy for Training and Validating along different Epochs")
plt.legend(loc="best")
plt.tight_layout()
plt.show()


