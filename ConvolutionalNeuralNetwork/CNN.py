import numpy as np
from PIL import Image
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from keras.src.metrics.accuracy_metrics import accuracy

print(f"Model file exists: {os.path.exists('trained_net.pth')}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


train_data = torchvision.datasets.CIFAR10(root='./data',train=True,transform=transform,download=True)
test_data = torchvision.datasets.CIFAR10(root='./data',train=False,transform=transform,download=True)

train_loader = torch.utils.data.DataLoader(train_data,batch_size=32,shuffle=True,num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True, num_workers=2)

print(f"Number of training samples: {len(train_data)}")
print(f"Number of test samples: {len(test_data)}")

image,label = train_data[0]
image.size()
torch.Size([3,32,32])
class_name = ["plane","car","bird","cat","deer","dog","frog","horse","ship","truck"]

class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,12,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(12,24,5)
        self.fcl1 = nn.Linear(24*5*5,120)
        self.fcl2 = nn.Linear(120,80)
        self.fcl3 = nn.Linear(80,10)


    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fcl1(x))
        x = F.relu(self.fcl2(x))
        x = self.fcl3(x)
        return x


if __name__ == "__main__":
    net = NeuralNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)

    for epoch in range(50):
        print(f"Training epoch {epoch}....")
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Loss: {running_loss / len(train_loader):.4f}")

    torch.save(net.state_dict(), 'trained_net.pth')
    net.load_state_dict(torch.load('trained_net.pth'))
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            images,labels = data
            outputs = net(images)
            _,predicted = torch.max(outputs,1)
            total+=labels.size(0)
            correct+=(predicted == labels).sum().item()

    accuracy = 100* correct/total
    print(f"Accuracy : {accuracy}%")

    new_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])


    def load_image(image_path):
        image = Image.open(image_path)
        image = new_transform(image)
        image = image.unsqueeze(0)
        return image

    image_paths = ["1.jpeg","2.jpeg","3.jpeg","4.jpeg"]
    images = [load_image(img) for img in image_paths]
    net.eval()
    with torch.no_grad():
        for image in images:
            output = net(image)
            _,predicted = torch.max(output,1)
            print(f"Prediction : {class_name[predicted.item()]}")
