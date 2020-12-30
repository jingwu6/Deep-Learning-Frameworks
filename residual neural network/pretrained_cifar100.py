import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
# Loading the data





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("you are using",device)



class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        print("start pre-trained model loading")
        def resnet18(pretrained=True):
            model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
            if pretrained:
                model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='E:\CS547 DeepLearning\hw4'))
            return model
        self.upsample=nn.Upsample(scale_factor=7, mode='bilinear')
        self.model = resnet18(pretrained=True)

        # print(model)
        # If you just need to fine-tune the last layer, comment out the code below.
        # for param in model.parameters():
        #     param.requires_grad = False

        self.model.fc = nn.Linear(self.model.fc.in_features, 100)


    def forward(self,x):
        x=self.upsample(x)
        x=self.model(x)
        return x

    def load(self,BatchSize):

        print("start data augmentation")
        transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomVerticalFlip(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()])

        train_dataset=torchvision.datasets.CIFAR100(root="./cifar100",train=True,transform=transform,download=True)
        self.trainloader=Data.DataLoader(train_dataset,batch_size=BatchSize,shuffle=True,num_workers=0)

        test_dataset=torchvision.datasets.CIFAR100(root="./cifar100",train=False,transform=transform,download=True)
        self.testloader=Data.DataLoader(test_dataset,batch_size=BatchSize,shuffle=False,num_workers=0)
        print("data loading done")



    def train(self, net, LR, Epoch):

        criteria = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
        self.Acc1=[]
        self.Acc2=[]
        for epoch in range(Epoch):
            print("traninig in the %i Epoch" % (epoch + 1))
            # running_loss = 0
            for step, (input, label) in enumerate(self.trainloader):
                # print(input.shape)
                # print(label)
                # if step==10:
                #     break

                input, labels = input.to(device), label.to(device)
                output = net(input)
                print(output.type(),labels)
                loss = criteria(output, labels)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                running_loss = loss.item()
                if (step % 100 == 99):
                    print(epoch, "Epoch,loss:", running_loss)
                    # break
            self.Acc1.append(self.trainAC(net))
            self.Acc2.append(self.test(net))

        print("training Done")


    def test(self,net):
        print("Accuracy of test set")
        count = 0
        total= 0

        with torch.no_grad():
            for (input,label) in self.testloader:
                input, labels=input.to(device),label.to(device)
                output=net(input)

                pred_y = torch.max(output, 1)[1].to(device)
                # print(labels,pred_y)
                total+=labels.shape[0]

                # print(labels.shape[0])
                count +=((pred_y==labels).sum().item())
                # print(count)
                # print(total)
                # break
        print(float(count/total))

        return float(count/total)


    def trainAC(self,net):
        print("Accuracy of training set")
        count = 0
        total= 0

        with torch.no_grad():
            for (input,label) in self.trainloader:
                input, labels=input.to(device),label.to(device)
                output=net(input)

                pred_y = torch.max(output, 1)[1].to(device)
                # print(pred_y)
                # print(labels)
                # print(labels,pred_y)
                total+=labels.shape[0]

                # print(labels.shape[0])
                count +=((pred_y==labels).sum().item())
                # print(count)
                # print(total)
                # break
        print(float(count/total))

        return float(count/total)

    def plotF(self, Epoch):
        x_raw = []
        for i in range(Epoch):
            x_raw.append(i)

        x = x_raw
        print(x)
        # y = L1
        y1 = self.Acc1
        y2 = self.Acc2

        lines = plt.plot(x, y1, x, y2)
        plt.setp(lines[0], linewidth=2)
        plt.setp(lines[1], markersize=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(('Train Set Accuracy', 'Test Set Accuracy'),
                   loc='upper right')
        plt.title('Accuracy about the Model')
        # plt.ylim(0.3, 1)  # return the current ylim
        # plt.show()
        # plt.plot(x_raw,L, 'bo')
        plt.savefig('Cifar_trained.jpg', bbox_inches='tight')





if __name__== "__main__":

    net=ResNet().to(device)
    net.load(BatchSize=128)
    net.train(net=net,LR=0.001,Epoch=25)
    # b=net.test(net)
    net.plotF(Epoch=25)
    # print(net)
