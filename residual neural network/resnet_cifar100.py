import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("you are using",device)

class BasicBlock(nn.Module):
    def __init__(self,inCHN,outCHN,stride):
        super(BasicBlock,self).__init__()
        # self.inCHN= inCHN
        # self.outCHN= outCHN
        self.conv1= nn.Sequential(
            nn.Conv2d(inCHN, outCHN, kernel_size=3 ,stride=stride , padding= 1),
            nn.BatchNorm2d(num_features= outCHN),
            nn.ReLU(),
            nn.Conv2d(outCHN, outCHN, kernel_size=3,stride=1 , padding=1),
            nn.BatchNorm2d(num_features=outCHN),
        )
        self.add= nn.Sequential()
        if (stride !=1 or inCHN != outCHN):
            self.add = nn.Sequential(
                nn.Conv2d(inCHN,outCHN,kernel_size=1 ,stride=stride, bias=False ),
                nn.BatchNorm2d(outCHN)
            )

    def forward(self,x):
            # print()
            output = self.conv1(x)
            output = output+ self.add(x)
            output = F.relu(output)
            return output

class BasicLayer(nn.Module):
    def __init__(self,basicblock,blockN,inCHN,outCHN,stride):
        super(BasicLayer,self).__init__()

        self.md=[]
        self.md.append(basicblock(inCHN,outCHN,stride))

        for i in range(blockN-1):
            # print(i)
            self.md.append(basicblock(outCHN,outCHN,stride=1))
        self.layers=nn.Sequential(*self.md)

    def forward(self,x):
        # print("layers")
        return  self.layers(x)





class ResNet(nn.Module):
    def __init__(self,layer,block):
        super(ResNet,self).__init__()
        ##(32,32,3)--(32,,32,32)
        self.conv1= nn.Sequential(
            nn.Conv2d(3,32,3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        ##(32,32,32)--(32,,32,32)
        self.block1=layer(basicblock=block,blockN=2,inCHN=32,outCHN=32,stride=1)
        self.drop0 = nn.Dropout2d(0.05)
        ##(32,32,32)--(16.5,16.5,64)
        self.block2=layer(basicblock=block,blockN=4,inCHN=32,outCHN=64,stride=2)
        self.drop1 = nn.Dropout2d(0.05)
        ##(16, 16, 64)--(8.5,8.5,128)
        self.block3=layer(basicblock=block,blockN=4,inCHN=64,outCHN=128,stride=2)
        self.drop2 = nn.Dropout2d(0.01)
        ##(8,8,128)--(4,4,256)
        self.block4=layer(basicblock=block,blockN=2,inCHN=128,outCHN=256,stride=2)
        self.drop3 = nn.Dropout2d(0.01)
        ##(4,4,256)--(2,2,256)
        self.MP=nn.MaxPool2d(kernel_size=3,stride=1)

        self.fc= nn.Sequential(
            nn.Linear(4*256,100)
        )


    def forward(self,x):
        # print("input",x.shape)
        x=self.conv1(x)
        # print("C1",x.shape)
        x=self.block1(x)
        # x=self.drop0(x)
        # print("B1",x.shape)
        x=self.block2(x)
        # x=self.drop1(x)
        # print("B2".shape)
        x=self.block3(x)
        # x=self.drop2(x)
        # print(x.shape)
        x=self.block4(x)
        # x=self.drop3(x)
        # print(x.shape)
        x=self.MP(x)
        # x=F.dropout(x,p=0.2)
        x=self.fc(x.view(-1,4*256))

        return x

    def load(self,BatchSize):

        print("start data augmentation")
        transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=2),
             transforms.RandomVerticalFlip(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()])

        transform1 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            #    transforms.RandomVerticalFlip(),
            transforms.RandomCrop(size=32),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        #    transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
        transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        transform22 = transforms.Compose(
            transforms.ToTensor())


        train_dataset=torchvision.datasets.CIFAR100(root="./cifar100",train=True,transform=transform1,download=True)
        self.trainloader=Data.DataLoader(train_dataset,batch_size=BatchSize,shuffle=True,num_workers=0)

        test_dataset=torchvision.datasets.CIFAR100(root="./cifar100",train=False,transform=transform2 ,download=True)
        self.testloader=Data.DataLoader(test_dataset,batch_size=BatchSize,shuffle=False,num_workers=0)
        print("data loading done")

    def trainS(self, net, LR, Epoch):
        net.train()
        self.Acc1=[]
        self.Acc2=[]

        criteria = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=5e-4)

        for epoch in range(Epoch):
            net.train()
            print("traninig in the %i Epoch" % (epoch + 1))
            if epoch == 20:
                optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)

            if epoch == 30:
                optimizer = torch.optim.Adam(net.parameters(), lr=LR / 30, weight_decay=5e-4)
            for step, (input, label) in enumerate(self.trainloader):
                # print(input.shape)
                # print(label)
                # if step==10:
                #     break

                input, labels = input.to(device), label.to(device)
                output = net(input)
                loss = criteria(output, labels)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                running_loss = loss.item()
                if (step % 100 == 99):
                    print(epoch, "Epoch,loss:", running_loss)
            self.Acc1.append(self.trainAC(net))
            self.Acc2.append(self.test(net))

        print("training Done")

    def trainAC(self,net):
        net.train()
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

    def test(self,net):
        net.eval()
        print("Testing Accuracy")
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
        plt.savefig('Cifar100.jpg', bbox_inches='tight')





if __name__== "__main__":

    net=ResNet(layer=BasicLayer,block=BasicBlock).to(device)
    net.load(BatchSize=128)

    net.trainS(net=net,LR=0.001,Epoch=50)
    # b=net.test(net)
    net.plotF(Epoch=50)