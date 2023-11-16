import numpy as np
class Perceptron:
    def __init__(self, eta=0.01,n_iter=50, random_state=1):
        self.eta=eta #learning rate
        self.n_iter=n_iter #number of passes over training dataset
        self.random_state=random_state #random nmber generator seed for random weight initializn

#loops over all individual examples in the training dataset and uodates the weights accordin to the perceptron learning rule
    def fit(self,X,y):
        rgen=np.random.RandomState(self.random_state) 
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=X.shape[1]) #w_: weights after fitting
        #self.w_=np.zeros(X.shape[1]) #initializing to all zeros
        self.b_=np.float_(0.) #b_: bias unit after fitting
        self.errors_=[]#number of misclassifications in each epoch
        #X: {input/ examples, features} TRAINING EXAMPLES
        #Y: [n_examples] TARGET VALUES
        #returns self object

        for _ in range(self.n_iter):
            errors=0
            for xi, target in zip(X,y):
                update=self.eta*(target-self.predict(xi))
                self.w_+=update*xi
                self.b_+=update
                errors+=int(update!=0.0) #collecting the number of misclassifications
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        #you can either use for loop for performing dot produc tusing multiplication
        #Benefit of using numoy--> vectorization-- arithmetic operations are vectorized - the elemental operation(like * ) is applied automatically to all elements in an array
        #makes better use of CPU with simd supportsingle instruction multiple data
        return np.dot(X,self.w_)+self.b_#calculates the vector dot product, w^Tx + b
    
    #class labels are predicted here
    def predict(self,X):
        #returns class label after one step
        #if the net input calculated is >0 then returns 1 else 0
        return np.where(self.net_input(X) >=0.0,1,0)
    



