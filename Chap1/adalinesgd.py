#Upgraded version of perceptron model
#Adaptive Linear Neuron Classifier Model
#instead of a step function, a linear activation function is introduced
#the values will now be continuous
#the true labels are mapped to continuous activation outputs for error calculation
#also called full batch grad desc
import numpy as np
class AdalineSGD:
    def __init__(self, eta=0.01,n_iter=50,random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.random_state=random_state
    def fit(self,X,y):
        rgen=np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=X.shape[1])
        self.b_=np.float_(0.)
        self.losses_=[]

        for i in range(self.n_iter):
            net_input=self.net_input(X)
            output=self.activation(net_input)
            errors=(y-output)
            self.w_+=self.eta*2.0*X.T.dot(errors)/X.shape[0]
            self.b_+=self.eta*2.0*errors.mean()
            loss=(errors**2).mean()
            self.losses_.append(loss)
        return self
    
    def net_input(self,X):
        return np.dot(X,self.w_)+self.b_
    
    def activation(self,X):
        return X
    
    def predict(self,X):
        return np.where(self.activation(self.net_input(X))>=0.5,1,0)
    