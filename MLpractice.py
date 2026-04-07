import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr=lr
        self.epochs=epochs


    def fit(self, X,y):
        n,m=X.shape #rowsxcolumns 
        self.W=np.zeros(m) #number of columns 1xfeatures
        self.b=0 #it is only a constant

        for epoch in range(self.epochs):
            #prediction
            y_pred=np.dot(X,self.W)+self.b #shape=nx1
            #loss
            loss=np.mean((y_pred-y)**2)
            #gradients
            dW=(2/n)*np.dot(X.T,(y_pred-y)) #X.T because mxn and nx1 dot product# tells how much each feature contributes to the loss
            db=(2/n)*sum(y_pred-y)
            #update
            self.W-=self.lr*dW
            self.b-=self.lr*db

            if epoch%100==0:
                print(f"Epoch{epoch}, Loss{loss}")


    def predict(self,X):
        return np.dot(X,self.W)+self.b

def sigmoid(z):
    return 1/(1+np.exp(-z))

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr=lr
        self.epochs=epochs
    def fit(self, X,y):
        n,m=X.shape #shape is an attribute here of X
        self.W=np.zeros(m)
        self.b=0

        for epoch in range(self.epochs):
            #predict
            z=np.dot(X, self.W)+self.b
            y_pred=sigmoid(z)

            #loss BCE -logloss
            loss = -np.mean(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
            #gradients
            dW=(1/n)*np.dot(X.T,(y_pred-y))
            db=(1/n)*np.sum(y_pred-y)
            #update weights
            self.W-=self.lr*dW
            self.b-=self.lr*db
            print("weight ",self.W)
            print("bias ",self.b)

    def predict(self, X):
        z=np.dot(X,self.W)+self.b
        probs=sigmoid(z)
        val_bin=(probs>0.5).astype(int)
        return val_bin


#y=2x+3
#TEST CASE FOR LINEAR REGRESSION
# X= np.array([[1],[2],[3],[4],[5]])
# y=np.array([5,7,9,11,13])
# model = LinearRegression(lr=0.01,epochs=5000)
# model.fit(X,y)
# preds=model.predict(X)
# print(preds)
# mse=np.mean((preds-y)**2)
# print("mse is ",mse)
# X_new=np.array([[6],[7]])
# preds_new=model.predict(X_new)
# print(preds_new)
# print("Weight:", model.W)
# print("Bias: ",model.b)


#TESTCASE FOR LOGISTIC REGRESSION -- probabilities -- binary classification
X=np.array([[1],[2],[3],[4]])
y=np.array([0,0,1,1])
model=LogisticRegression()
model.fit(X,y)
y_pred=model.predict(X)
epsilon = 1e-15 #to handle the case where y_pred becomes 0 or 1, the log will give inf NaN. to avoid this we clip the y_pred value to a minimal value but never let it become fully 0 or 1
y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
error=-np.mean(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
print(error)
print("Weight ", model.W)
print("bias ", model.b)
#NOTE: Sigmoid causes vanishing gradients problem because if z value becomes too large or too small then sigmoid of z becomes 0 or 1 due to the exponential term
#then th loss value will also become 0. This will make the change in weights/ gradients to be very small whoich makes the model to not get updated 
#This problem may occur in logistic regression, however this can be soled by using techniques liek clipping that we donot let the predicted value become 0 or 1 fully we clip it to a closer constant value
#Vanishing gradiets is till not a problem in logistic regression becuase sigmoid is only used once at the iutput layer for probabilites. 
#However this is not used in DL models and is replced by ReLU=max(0,x) becuse in dl models, the hidden layers will haeve may multiplications layerwise for the activation function
#this makes the gradients vanish as the layers increase as a result the model doesnt get updated hence sigmoid is not used as an activation in dl and relu is often seen as hiodden layer
