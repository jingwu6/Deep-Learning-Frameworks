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
import matplotlib.gridspec as gridspec
import numpy as np
import time
from time import *
from torch import autograd
from torch.autograd import Variable

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

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

batch_size = 128
testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
testloader = enumerate(testloader)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=[196, 32, 32]),
            nn.LeakyReLU(0.02, inplace=True)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 2, 1),
            nn.LayerNorm(normalized_shape=[196, 16, 16]),
            nn.LeakyReLU(0.02, inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 1, 1),
            nn.LayerNorm(normalized_shape=[196, 16, 16]),
            nn.LeakyReLU(0.02, inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 2, 1),
            nn.LayerNorm(normalized_shape=[196, 8, 8]),
            nn.LeakyReLU(0.02, inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 1, 1),
            nn.LayerNorm(normalized_shape=[196, 8, 8]),
            nn.LeakyReLU(0.02, inplace=True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 1, 1),
            nn.LayerNorm(normalized_shape=[196, 8, 8]),
            nn.LeakyReLU(0.02, inplace=True))
        self.conv7 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 1, 1),
            nn.LayerNorm(normalized_shape=[196, 8, 8]),
            nn.LeakyReLU(0.02, inplace=True))
        self.conv8 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 2, 1),
            nn.LayerNorm(normalized_shape=[196, 4, 4]),
            nn.LeakyReLU(0.02, inplace=True))
        self.pool = nn.MaxPool2d(kernel_size=4)
        self.fc1 = nn.Sequential(
            nn.Linear(196, 1))
        self.fc10 = nn.Sequential(
            nn.Linear(196, 10)
            )
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
        x = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])
        fc1_out = self.fc1(x)
        fc10_out = self.fc10(x)
        return fc1_out, fc10_out

model = torch.load('cifar10.model')
model.cuda()
model.eval()


batch_idx, (X_batch, Y_batch) = testloader.__next__()
X_batch = Variable(X_batch,requires_grad=True).cuda()
# print (Y_batch)
Y_batch_alternate = (Y_batch + 1)%10
# print (Y_batch_alternate )
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()


## save real images
samples = X_batch.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/real_images.png', bbox_inches='tight')
plt.close(fig)


_, output = model(X_batch)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print(accuracy)



## slightly jitter all input images
criterion = nn.CrossEntropyLoss(reduce=False)
loss = criterion(output, Y_batch_alternate)

gradients = torch.autograd.grad(outputs=loss, inputs=X_batch,
                          grad_outputs=torch.ones(loss.size()).cuda(),
                          create_graph=True, retain_graph=False, only_inputs=True)[0]

# save gradient jitter
gradient_image = gradients.data.cpu().numpy()
gradient_image = (gradient_image - np.min(gradient_image))/(np.max(gradient_image)-np.min(gradient_image))
gradient_image = gradient_image.transpose(0,2,3,1)
fig = plot(gradient_image[0:100])
plt.savefig('visualization/gradient_image.png', bbox_inches='tight')
plt.close(fig)


# jitter input image
gradients[gradients>0.0] = 1.0
gradients[gradients<0.0] = -1.0

gain = 8.0
X_batch_modified = X_batch - gain*0.007843137*gradients
X_batch_modified[X_batch_modified>1.0] = 1.0
X_batch_modified[X_batch_modified<-1.0] = -1.0

## evaluate new fake images
_, output = model(X_batch_modified)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print(accuracy)

## save fake images
samples = X_batch_modified.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/jittered_images.png', bbox_inches='tight')
plt.close(fig)
