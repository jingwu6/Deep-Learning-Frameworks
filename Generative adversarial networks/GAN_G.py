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
import matplotlib
import numpy as np
import time
from time import *


matplotlib.use('Agg')
import matplotlib.gridspec as gridspec


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()

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


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(100, 196 * 4 * 4),
            nn.BatchNorm1d(196 * 4 * 4),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=196, out_channels=196, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=196, out_channels=196, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=196, out_channels=196, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True))
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh())

    def forward(self, x):
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = x.view(-1, 196, 4, 4)
        # print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        return x


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


def calc_gradient_penalty(netD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    alpha = alpha.cuda()

    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    # fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig


def StartTrain(ClassMD, ClassMG, Epoch, batch_size, gen_train):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("you are using", device)

    trainloader, testloader = load(batch_size)

    aD = ClassMD().to(device)

    aG = ClassMG().to(device)

    # model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer_g = torch.optim.Adam(aG.parameters(), lr=0.0001, betas=(0, 0.9))
    optimizer_d = torch.optim.Adam(aD.parameters(), lr=0.0001, betas=(0, 0.9))
    n_z = 100
    n_classes = 10
    np.random.seed(352)
    label = np.asarray(list(range(10)) * 10)
    noise = np.random.normal(0, 1, (100, n_z))
    label_onehot = np.zeros((100, n_classes))
    label_onehot[np.arange(100), label] = 1
    noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
    noise = noise.astype(np.float32)

    save_noise = torch.from_numpy(noise)
    save_noise = Variable(save_noise).to(device)

    start_time = time()

    for epoch in range(Epoch):

        aG.train()
        aD.train()

        for step, (input, label) in enumerate(trainloader):
            # train G
            if ((step % gen_train) == 0):
                for p in aD.parameters():
                    p.requires_grad_(False)

                aG.zero_grad()

                label = np.random.randint(0, n_classes, batch_size)
                noise = np.random.normal(0, 1, (batch_size, n_z))
                label_onehot = np.zeros((batch_size, n_classes))
                label_onehot[np.arange(batch_size), label] = 1
                noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
                noise = noise.astype(np.float32)
                noise = torch.from_numpy(noise)
                noise = Variable(noise).to(device)
                # print("n",noise.shape)
                # fake_label = Variable(torch.from_numpy(label.astype(np.float32)).type(torch.FloatTensor)).to(device)
                fake_label = Variable(torch.from_numpy(label).type(torch.long)).to(device)
                # print(fake_label)
                # fake_label = torch.empty(batch_size, dtype=torch.long).random_(10).to(device)

                fake_data = aG(noise)
                gen_source, gen_class = aD(fake_data)

                gen_source = gen_source.mean()
                print(gen_class.type(),fake_label.type(torch.LongTensor))
                print(label.shape)

                gen_class = criterion(gen_class, label)
                gen_cost = -gen_source + gen_class
                gen_cost.backward()
                optimizer_g.step()




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("you are using", device)


aG=generator().to(device)
aD=discriminator().to(device)

trainloader, testloader = load(128)

criterion = nn.CrossEntropyLoss()
optimizer_g = torch.optim.Adam(aG.parameters(), lr=0.0001, betas=(0, 0.9))

Epoch=1
n_classes=10
n_z=100
batch_size=128
for epoch in range(Epoch):

    aG.train()


    for step, (input, label) in enumerate(trainloader):
        # train G

        input, labels = input.to(device), label.to(device)
        label = np.random.randint(0, n_classes, batch_size)
        noise = np.random.normal(0, 1, (batch_size, n_z))
        label_onehot = np.zeros((batch_size, n_classes))
        label_onehot[np.arange(batch_size), label] = 1
        noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
        noise = noise.astype(np.float32)
        noise = torch.from_numpy(noise)
        noise = Variable(noise).to(device)
        fake_data = aG(noise)
        fake_label = Variable(torch.from_numpy(label).type(torch.LongTensor)).to(device)
        gen_source, gen_class = aD(fake_data)

        print(labels.type() ,fake_label.type())
        print(labels.shape ,fake_label.shape)
        # fake_label= fake_label.type(torch.LongTensor)
        # gen_class = criterion(gen_class, labels)
        gen_class = criterion(gen_class, fake_label)



        break