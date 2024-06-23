import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\pradh\Downloads\Iris.csv")

class Model(nn.Module):

    #Input layer(4 features of the flower) --> Hidden Layer1 (number of neurons) --> Hidden Layer H2 (n) --> output (one of 3 classes of iris flower)
    def __init__(self, input_features=4, h1 = 8, h2 = 9, out_features = 3):
        super().__init__() #instantiates our nn.module
        self.fc1 = nn.Linear(input_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x): # to move forward in our nn
      x = F.relu(self.fc1(x)) #relu = rectified linear unit function = do something, and return a numerical output
      x = F.relu(self.fc2(x))
      x = self.out(x)
                              # in this seuence, we're basically pushing our input through the nn
      return x

torch.manual_seed(5)
#creating an instance of a model
model = Model()

#Now, we creat a clear classifying column to help us with torch
#Let's replace the last col with a numerical value

df['Species'] = df['Species'].replace('Iris-setosa', 0.0)
df['Species'] = df['Species'].replace('Iris-versicolor', 1.0)
df['Species'] = df['Species'].replace('Iris-virginica', 2.0)


#Now, let's split our training data, and our testing data
#This will be used to train/test our nn model

X = df.drop(['Species', 'Id'], axis = 1)
y = df['Species']

X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 41)

X_train = torch.FloatTensor(X_train) #converting each tensor with respect to specific dtypes
X_test = torch.FloatTensor(X_test) #converting tensors to specific dtypes
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


#Set criterion and optimizer, how far off the predictions are from test data

criterion = nn.CrossEntropyLoss()
#Choose Adam Optimizer (popular ptimizer to use) lr = learning rate (if the error doesn't go down after a bunch of iterations) (epochs)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) #lr/learning rate represents how accurate you want your model to be
                                                            # obv, when you only have 150 rows of data, it's easy to achieve high accuracy
                                                            # but when u have a million rows, an lr of 0.01 vs 0.001 might make a whole 2-day difference!
                                                            #model.params() = passing in the params from our model class, so that our model can train
                                                            # based on layers we have in our nn
                                                            # it will run tons of epochs (iterations through the layers of the nn) until it reaches the desired lr

#PERFECT, NOW TIME TO TRAIN OUR MODEL!!

#Epoch = one run thru all the training data in our nn model
epochs = 1000
losses = []
for i in range(epochs):
   #Go forward and get a prediction
   y_pred = model.forward(X_train)

   #measure the loss/error (will be high at first)

   loss = criterion(y_pred, y_train) #comparing a difference between our model's predicted value and the actual values  
   losses.append(loss.detach().numpy()) #keeping track of our losses

#    if i % 10 == 0: print(f'Epoch {i}; loss: {loss}')

   #now let's take the error rate of forward proprgation and feed it back into the network
   # so we can fine tune the weights

   optimizer.zero_grad()
   loss.backward()
   optimizer.step()

plt.plot(range(epochs), losses)
plt.ylabel('losses')
plt.xlabel('epoch')
# plt.show()

#Now let's evaluate our model based on our test dataset
with torch.no_grad(): #will turn off back propogation, wil send the model only through forward passes
    y_eval = model.forward(X_test) #X_test will be the features from our test set
    loss = criterion(y_eval, y_test) #find the loss/error of y_eval vs y_test
    print(loss) #output is 0.1359, which i far off from our runtime training of 0.02


correct = 0
with torch.no_grad(): #don't account for gradient values
   for i, data in enumerate(X_test): #for every index + value in our test data
        y_val = model.forward(data) #

        print(f'{i+1}: {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}')

        #Correct or nah
        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f'We got {correct} correct')   



#THIS IS NOW OUR FULL FLEDGED COMPLETED MODEL (SIMPLE) 
#NOW, let's test out our model with our own data

new_iris = torch.tensor([4.7, 3.2, 1.3, 0.2])
with torch.no_grad():
    op = model(new_iris)
    if op.argmax() == 0: print('setosa')
    if op.argmax() == 1: print('versicolor')
    if op.argmax() == 2: print('virginica')

#Now we save our Nn model
torch.save(model.state_dict(), 'my_first_NN_model.pt')

#If we wanna load our saved modell...
new_model = Model()
new_model.load_state_dict(torch.load('my_first_NN_model.pt'))
print(new_model.eval())

    
