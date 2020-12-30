
import os
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=[196, 32, 32]),
            nn.LeakyReLU(0.02, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm(normalized_shape=[196, 16, 16]),
            nn.LeakyReLU(0.02, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=[196, 16, 16]),
            nn.LeakyReLU(0.02, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm(normalized_shape=[196, 8, 8]),
            nn.LeakyReLU(0.02, inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=[196, 8, 8]),
            nn.LeakyReLU(0.02, inplace=True)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=[196, 8, 8]),
            nn.LeakyReLU(0.02, inplace=True)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=[196, 8, 8]),
            nn.LeakyReLU(0.02, inplace=True)
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm(normalized_shape=[196, 4, 4]),
            nn.LeakyReLU(0.02, inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=4)

        self.fc1 = nn.Linear(196, 1)

        self.fc10 = nn.Linear(196, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool(x)
        temp = x.size()
        x = x.view(-1, temp[1] * temp[2] * temp[3])
        out1 = self.fc1(x)
        out2 = self.fc10(x)

        return out1, out2


def load(batch_size):
    print("start data loading")

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
        transforms.ColorJitter(
            brightness=torch.abs(0.1 * torch.randn(1)).item(),
            contrast=torch.abs(0.1 * torch.randn(1)).item(),
            saturation=torch.abs(0.1 * torch.randn(1)).item(),
            hue=torch.abs(0.1 * torch.randn(1)).item()
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("data loading done")

    return trainloader, testloader


def StartTrain(ClassMD, Epoch, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("you are using", device)

    trainloader, testloader = load(batch_size)

    model = ClassMD().to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(Epoch):

        if (epoch == 50):
            for param_group in optimizer.param_groups:
                learning_rate = param_group['lr']
                # print(learning_rate)
                param_group['lr'] = learning_rate / 10.0
                # print(param_group['lr'])
        if (epoch == 75):
            for param_group in optimizer.param_groups:
                learning_rate = param_group['lr']
                param_group['lr'] = learning_rate / 100.0

        for step, (input, label) in enumerate(trainloader):

            input, labels = input.to(device), label.to(device)
            _, output = model(input)
            # print(output.shape)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch > 10):
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        if ('step' in state and state['step'] >= 1024):
                            state['step'] = 1000

            optimizer.step()
            running_loss = loss.item()
            if (step % 100 == 99):
                print(epoch, "Epoch,loss:", running_loss)

    torch.save(model, 'cifar10.model.ckpt')


if __name__ == '__main__':
    StartTrain(ClassMD=Discriminator, Epoch=100, batch_size=128)
    # trainloader, testloader = load(128)




