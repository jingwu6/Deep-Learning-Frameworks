import numpy as np
import h5py
from time import *
from random import randint

class selfCNN():
    def __init__(self,LR):
        super(selfCNN,self).__init__()
        self.LR=LR
    def sigmoid(self,x):
        return (1./(1+np.exp(-x)))

    def sigmoidP(self,x):
        return x*(1-x)

    def softmax(self,x):
        return np.exp(x)/np.sum(np.exp(x))

    def load(self):
        Data=h5py.File("MNISTdata.hdf5","r")


        self.x_train = np.float32(Data['x_train'][:])
        self.x_train =self.x_train.reshape(self.x_train.shape[0],28,28)

        self.y_train = np.int32(np.array(Data['y_train'][:,0]))
        # print(self.x_train.shape)

        self.x_test = np.float32(Data['x_test'][:])
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28)
        self.y_test = np.int32(np.array(Data['y_test'][:,0]))
        # print(self.x_test.shape)
        Data.close
        print("data loading done")

    def initialization(self,width,height,filterN):
        self.width=width
        self.height=height
        self.filterN=filterN
        self.filter=np.random.normal(0,np.sqrt(2/(784+self.width*self.width)),
                                     size=(self.filterN,self.width,self.height))
        # print(self.filter.shape)
        self.W=np.random.normal(0,np.sqrt(2 / (self.filterN*self.width * self.width+10)),
                                       size=(10,self.filterN, 29-self.width, 29-self.height))
        # print(self.W.shape)
        self.b=np.random.normal(0,0.1,size=(10,1))
        print("initialization done")

    def fowardpropagation(self, X):
        Z=np.zeros((self.filterN,29-self.width,29-self.height))
        for f in range(self.filterN):
            for i in range(29-self.width):
                for j in range(29-self.height):
                    Z[f, j, i]=np.tensordot(self.filter[f], X[i:(i + self.width), j:(j + self.height)])

        A1=self.sigmoid(Z)
        z2=(np.tensordot(self.W,A1,axes=((1,2,3),(0,1,2)))).reshape(-1,1)+self.b
        # print(U.shape)
        A2=self.softmax(z2)
        cache={"A1":A1,"z2":z2,"A2":A2}
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

    def convolution(self, X, K):
        d = X.shape[0]
        d_y = K.shape[0]
        d_x = K.shape[1]
        Z = np.zeros((d-d_y+1, d-d_x+1))
        # Z = signal.convolve2d(X, K, mode='valid')
        for i in range(d-d_y+1):
            for j in range(d-d_x+1):
                Z[i,j] = np.tensordot(K, X[i:(i+d_y), j:(j+d_x)], axes=((0,1),(0,1)))
        return Z

    def backpropagation(self,cache,X,Y):
        dz2=cache["A2"]
        dz2[Y]-=1
        db2 = dz2
        dw2=np.zeros(self.W.shape)

        for i in range(dw2.shape[0]):
            dw2[i]=dz2[i]*cache["A1"]
        # print(self.W.shape)
        # print(dw2.flatten().shape)
        delta=np.tensordot(self.W, dz2.flatten(),axes=((0),(0)))
        df=np.zeros(self.filter.shape)
        for i in range(self.filterN):
            # print(X.shape)
            # print(cache["H"][i].shape)
            # print(delta[i].shape)
            df[i]=self.convolution(X, self.sigmoidP(cache["A1"][i]) * delta[i])

        self.W-=self.LR*dw2
        self.b-=self.LR*db2
        self.filter -= self.LR*df

    def train(self,iteration):
        print("start our training")
        start=time()
        for i in range(iteration):
            for j in range(60000):

                index = randint(0, 59999)
                input=self.x_train[index]
                Y=np.array(self.y_train[index])
                cache = self.fowardpropagation(input)
                # print(np.argmax(cache["A2"], axis=0))
                # print(cache["A2"])
                self.backpropagation(cache, input, Y)

                if ((j+1)%1000==0): print ("Trained %i samples"%(j+1), " in epoch %i"%(i+1), "Cost: %.2f seconds"%(time()-start))

            print("Epoch %i Finined" % (i + 1))
                # print ("Epoch %i: "%(i+1), ", Loss: ", self.loss(f.T, self.y_train), ", Train Accuracy: ", self.accuracy(f.T, self.y_train))

        print("training process done, Used %.2f seconds" % (time() - start))



    def test(self):
        print("start evaluation of trained model")
        print(self.x_test.shape)

        count=0
        for j in range(self.x_test.shape[0]):
            # print(j)
            y_pred = self.fowardpropagation(self.x_test[j])["A2"]
            label_pre = np.argmax(y_pred, axis=0)
            # print(self.y_test[j])
            # print(label_pre[0])
            if label_pre[0] == self.y_test[j]:
                count+=1
        print("count:",count)
        print("test set accuracy", float(count)/self.x_test.shape[0])





if __name__=="__main__":
    model=selfCNN(0.01)
    model.load()
    model.initialization(width=3,height=3,filterN=4)
    model.train(iteration=7 )
    model.test()

