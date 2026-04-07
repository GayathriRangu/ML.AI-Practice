import torch
import torch.nn as nn
import torch.optim as optim
#sample dataset XOR
X=torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y=torch.tensor([[0.],[1.],[1.],[0.]])

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet,self).__init__()
        self.fc1=nn.Linear(2,4)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(4,1)
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.sigmoid(x)
        return x
    
model= NeuralNet()
#loss and optimizer
criterion=nn.BCELoss()
optimizer=optim.SGD(model.parameters(),lr=0.1)

for epoch in range(100000):
    outputs=model(X)
    loss=criterion(outputs,y)
    optimizer.zero_grad() #clear old gradients before cpmputing the current ones
    loss.backward()#compute gradients 
    optimizer.step()#update the weights based on gradients 
    if epoch % 500 == 0:
        print(f"epoch {epoch}, Loss {loss.item()}")
#testing
with torch.no_grad():
    print(model(X))

