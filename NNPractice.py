import numpy as np
#   NN USING NUMPY ONLY - ONE HIDDEN LAYER
class NeuralNetwork:
    def __init__(self, input_size, hidden_size,output_size):
        #initialize weights
        self.W1 = np.random.randn(input_size, hidden_size)*0.01
        self.b1=np.zeros((1,hidden_size))
        self.W2=np.random.randn(hidden_size, output_size)*0.01
        self.b2=np.zeros((1,output_size))

    def relu(self,Z):
        return np.maximum(0,Z)
    def relu_derivative(self, Z):
        return Z>0

#Forward pass
    def forward(self,X):
        self.Z1=np.dot(X, self.W1)+self.b1
        self.A1=self.relu(self.Z1)
        self.Z2=np.dot(self.A1,self.W2)+self.b2
        self.A2=self.relu(self.Z2)
        return self.A2
    #loss - binary cross entropy BCE
    def compute_loss(self, y, y_pred):
        epsilon=1e-8
        loss=-np.mean(y*np.log(y_pred+epsilon)+(1-y)*np.log(1-y_pred+epsilon))
        return loss
    #backprop
    def backward(self, X,y):
        m= X.shape[0]
        #Ouptu layer gradient
        dZ2= self.A2-y
        dW2=(1/m)*np.dot(self.A1.T,dZ2)
        db2=(1/m)*np.sum(dZ2,axis=0, keepdims=True)
        #Hidden layer gradients
        dA1= np.dot(dZ2, self.W2.T)
        dZ1 = dA1 *self.relu_derivative(self.Z1)
        dW1=(1/m)*np.dot(X.T,dZ1)
        db1=(1/m)*np.sum(dZ1, axis=0, keepdims=True)
        #store gradients
        self.dW1, self.db1 =dW1, db1
        self.dW2, self.db2= dW2, db2
    #update weights
    def update(self, lr):
        self.W1-=lr*self.dW1
        self.b1-=lr*self.db1
        self.W2-=lr*self.dW2
        self.b2-=lr*self.db2
    #training loop
    def train(self, X, y, epochs, lr):
        for i in range(epochs):
            y_pred=self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.backward(X,y)
            self.update(lr)
            if i %100==0:
                print(f"epoch {i}, loss {loss}")

#TEST USE DUMMY DATASET
X=np.array([[0,0],[0,1],[1,0],[1,1]])
y= np.array([[0],[1],[1],[0]])
nn=NeuralNetwork(input_size=2, hidden_size=4, output_size=4)
nn.train(X,y, epochs=1000,lr=0.1)
pred=nn.forward(X)
print(pred)
print(nn.compute_loss(y, pred))
# print(nn.W1)