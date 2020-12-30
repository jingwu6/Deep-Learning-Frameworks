import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("you are using", device)


class self_CNN(torch.nn.Module):

    def __init__(self):
        super(self_CNN, self).__init__()
        ### (32,32,3)--(3,64,1)--(32,32,64)--(31,31,64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),

        )
        ### (31,31,64)--(31,31,128)--(15,15,128)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

        )
        ### (15,15,128)--(9,9,256)--(8,8,256)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        ### (8,8,256)--(5,5,512)--(4,4,512)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(4 * 4 * 512, 2000),
            nn.BatchNorm1d(num_features=2000),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2000, 500),
            nn.BatchNorm1d(num_features=500),
            nn.ReLU(),
            nn.Dropout()
        )
        self.out = nn.Sequential(
            nn.Linear(500, 10),
            nn.BatchNorm1d(num_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)

        x = x.view(-1, 4 * 4 * 512)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = self.fc2(x)
        output = self.out(x)
        # print(output.shape)

        return output

    def train(self, net, LR, Epoch):

        criteria = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)

        for epoch in range(Epoch):
            print("traninig in the %i Epoch" % (epoch + 1))
            running_loss = 0
            for step, (input, label) in enumerate(self.trainloader):
                # print(input.shape)
                # print(label.shape)

                input, labels = input.to(device), label.to(device)
                output = net(input)
                loss = criteria(output, labels)
                optimizer.zero_grad()
                loss.backward()

                # if (epoch >2):
                #     for group in optimizer.param_groups:
                #         for p in group['params']:
                #             state = optimizer.state[p]
                #             if (state['step'] >= 1024):
                #                 state['step'] = 1000
                optimizer.step()
                running_loss = loss.item()
                if (step % 100 == 99):
                    print(epoch, "Epoch,loss:", running_loss)
                # if (step % 100 == 99):  # print every 2000 mini-batches
                #
                #     print("Epoch: ", epoch + 1, ", ", i + 1, " mini-batches, loss: %.3f" % (running_loss / 100.),
                #           ", Using %.3f s" % (time() - start_time))
                #     running_loss = 0.
        print("training Done")

        torch.save(net, 'net2.pkl')

    def MC(self, net, sample):
        print("start MC Process")
        temp = 0
        for i in range(sample):
            count = 0
            total = 0

            with torch.no_grad():
                for (input, label) in self.trainloader:
                    input, labels = input.to(device), label.to(device)
                    output = net(input)
                    pred_y = torch.max(output, 1)[1].to(device)
                    total += labels.shape[0]
                    # print(labels.shape[0])
                    count += ((pred_y == labels).sum().item())
                # print("Monte Calo accuracy:", float(count / total))
            temp += float(count / total)
        print("Monte Calo accuracy:", temp / 50)

    def test(self, net):
        print("start testing")
        count = 0
        total = 0

        with torch.no_grad():
            for (input, label) in self.testloader:
                input, labels = input.to(device), label.to(device)
                output = net(input)
                pred_y = torch.max(output, 1)[1].to(device)
                total += labels.shape[0]
                # print(labels.shape[0])
                count += ((pred_y == labels).sum().item())

        print("test accuracy:", float(count / total))

        # return float(count/total)

    def load(self, BatchSize):

        print("start data augmentation")
        transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomVerticalFlip(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()])
        transform1 = transforms.Compose([
            transforms.ToTensor()])

        train_dataset = torchvision.datasets.CIFAR10(root="./cifar10", train=True, transform=transform, download=True)
        self.trainloader = Data.DataLoader(train_dataset, batch_size=BatchSize, shuffle=True, num_workers=0)

        test_dataset = torchvision.datasets.CIFAR10(root="./cifar10", train=False, transform=transform1, download=True)
        self.testloader = Data.DataLoader(test_dataset, batch_size=BatchSize, shuffle=False, num_workers=0)
        print("data loading done")


if __name__ == "__main__":
    net = self_CNN().to(device)
    net.load(BatchSize=128)
    net.train(net=net, LR=0.001, Epoch=200)
    ### heuristic method

    net.test(net)

    ### MC method for 50 samples
    net.MC(net, sample=50)
