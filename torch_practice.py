import torch
import torchvision
import time
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torchvision import transforms

def make_dataset(batch_size, num_workers):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = len(classes)
    train_size, test_size = len(trainset), len(testset)
    
    return trainloader, train_size, testloader, test_size, classes, num_classes
    
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 1) #(6, 32, 32)
        self.pool1 = nn.MaxPool2d(2, 2) #(6, 16, 16)
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 1) #(16, 16, 16)
        self.pool2 = nn.MaxPool2d(2, 2) #(16, 8, 8)
        self.fc1 = nn.Linear(16 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2( F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
if __name__ == '__main__':  
    
    epochs = 20
    batch_size = 16
    num_workers = 0
    learning_rate = 0.001
    
    trainloader, train_size, \
        testloader, test_size, classes, num_classes = make_dataset(batch_size, num_workers)
    
    net = Net(num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    net.to(device)
    

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        
        start_time = time.time()
        
        train_loss = 0.0
        train_acc = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            
            outputs = net(inputs)
            batch_loss = loss(outputs, labels)
            batch_loss.backward()
            optimizer.step()
            
            train_loss += batch_loss.item()
            train_acc += np.sum(np.argmax(outputs.cpu().data.numpy(), axis=1) == data[1].numpy())
        print(f'[{epoch + 1:2d}/{epochs:2d}] {time.time()-start_time:.2f} sec(s) ' + \
                f'loss: {train_loss/train_size:.4f} accuracy: {100 * train_acc/train_size:.2f}')
        
    print('training finish')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
            
