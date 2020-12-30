import  numpy as np
import h5py
from time import *
from random import randint


class SelfNet():
    def __init__(self,LR,batchN,HiddenN):
        super(SelfNet, self).__init__()
        self.LR=LR
        self.batchN=batchN
        self.HiddenN=HiddenN

    def initialization(self):

        self.w1=np.random.randn(self.HiddenN,784)* np.sqrt(2 / 784)
        self.b1=np.zeros((self.HiddenN,1))
        self.w2=np.random.randn(10,self.HiddenN)* np.sqrt(2 / self.HiddenN)
        self.b2=np.zeros((10,1))
        print("initialization done")
    def load(self):
        Data=h5py.File("MNISTdata.hdf5","r")
        self.x_train = np.float32(Data['x_train'][:])
        self.y_train = np.int32(np.array(Data['y_train'][:,0]))
        self.x_test = np.float32(Data['x_test'][:])
        self.y_test = np.int32(np.array(Data['y_test'][:,0]))
        Data.close
        print("data loading done")

    def sigmoid(self,x):
        return (1./(1+np.exp(-x)))

    def softmax(self,x):
        return np.exp(x)/np.sum(np.exp(x))

    def fowardpropagation(self,input):
        z1=np.dot(self.w1,input)+self.b1
        A1=self.sigmoid(z1)
        z2=np.dot(self.w2,A1)+self.b2
        A2=self.softmax(z2)
        cache={"z1":z1,"z2":z2,"A1":A1,"A2":A2}
        return cache

    def loss(self,A2,label):
        """
        log loss in logistic regression
        A2 is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        	Note that y is not one-hot encoded vector.
        	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = label.shape[0]

        log_likelihood = -np.log(A2[label,range(m)])
        loss = np.sum(log_likelihood) / m
        return loss

    def backpropagration(self,cache,X,Y,batchN):

        dz2=cache["A2"]
        dz2[Y]-=1
        dw2=np.matmul(dz2,cache["A1"].T)
        db2= dz2

        dA1=np.matmul((self.w2).T,dz2)
        dz1=dA1*(cache["A1"]*(1-cache["A1"]))
        dw1=np.matmul(dz1,X.T)
        db1=dz1

        self.w1 -= self.LR * dw1
        self.b1 -= self.LR * db1
        self.w2 -= self.LR * dw2
        self.b2 -= self.LR * db2


    def train(self,iteration,batchN):
        print("start our training")
        start=time()
        for i in range(iteration):
            if i==50:
                self.LR=self.LR/5
            if i==80:
                self.LR = self.LR / 10
            FT= 0
            BT= 0
            CT= 0
            for j in range(60000):
                C1=time()
                index = randint(0, 59999)
                C2=time()
                CT+=(C2-C1)
                input=self.x_train[index].reshape((-1,1))
                Y=np.array(self.y_train[index])
                F1=time()
                cache=self.fowardpropagation(input)
                F2=time()
                FT+=(F2-F1)
                B1=time()
                self.backpropagration(cache,input,Y,batchN)
                B2=time()
                BT+=(B2-B1)

            y_pred=self.fowardpropagation(self.x_train.T)["A2"]
            print ("Epoch %i: "%(i+1), ", Loss: ", self.loss(y_pred, self.y_train), ", Train Accuracy: ", self.accuracy(self.x_train, self.y_train))

            print("Training time in this iteration Use %.2f seconds" % (time() - start))

            print ("Forward Time %.2f seconds"%(FT))
            print ("Backward Time %.2f seconds"%(BT))
            #print("Choosing Time %.2f seconds"%(CT))
        print("training done")

    def accuracy(self, X, Y_true):
        output = self.fowardpropagation(X.T)["A2"]
        y_hat = np.argmax(output, axis=0)
        acc = np.mean(np.int32(y_hat == Y_true))
        return acc

    def test(self):
        print("start evaluation of trained model")
        y_pred = self.fowardpropagation(self.x_test.T)["A2"]
        label_pre=np.argmax(y_pred,axis=0)
        count=0
        for i in range(len(label_pre)):
            if label_pre[i] == self.y_test[i]:
                count+=1

        print("test set accuracy", float(count)/len(label_pre))


if __name__=="__main__":
    model=SelfNet(LR=0.01,batchN=1,HiddenN=120)
    model.initialization()
    model.load()
    model.train(iteration=35,batchN=1)
    model.test()
