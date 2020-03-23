import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
# other utils
import pandas as pd
import numpy as np
from tqdm import tqdm

# read in the dataset (both train and test)
Eye_train = pd.read_csv("ann_train.csv",index_col=0)
Eye_test = pd.read_csv("ann_test.csv",index_col=0)
# shuffle the data set and prepare dataset used for ANN input
Eye_train = Eye_train.sample(frac=1).reset_index(drop=True)
Eye_train["Sex"] = [1 if item == 'M' else 0 for item in Eye_train["Sex"]]
Eye_test = Eye_test.sample(frac=1).reset_index(drop=True)
Eye_test["Sex"] = [1 if item == 'M' else 0 for item in Eye_test["Sex"]]

# The data set structure, allow loader to load batches
class EyeDataset(Dataset):
    def __init__(self, Eye_df):
        super(EyeDataset, self).__init__()
        int_ethnicity = np.array([0 if item=="Persian" else 1 for item in Eye_df.Ethnicity])
        self.y = torch.tensor(int_ethnicity)
        self.x = torch.tensor(Eye_df[["Age","Sex","MRD-1(R)","PTB (R)","TPS (R)","MRD-1 (L)","PTB (L)","TPS (L)"]].to_numpy(),dtype=torch.float32)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])
    
    def __len__(self):
        return len(self.x)

# The ANN model
class EyeData_ANN(nn.Module):
    # The model is made up with input, body, and output part
    # We use the following architecture overall:
    ## data input size: 8

    ## input part (activation: relu)
    ### 2nd input layer: 32 neurons

    ## body part (activation: relu)
    ### 1st hidden: 64 neurons, dropout rate 0.3
    ### 2nd hidden: 128 neurons, dropout rate 0.3
    ### 3rd hidden: 32 neurons
    ### 4th hidden: 16 neurons

    ## output (activation: softmax)
    ### 1st output: 2 neurons

    def __init__(self):
        super(EyeData_ANN, self).__init__()
        self.input = nn.Sequential(
            nn.Linear(8,32),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU()
        )
        self.body = nn.Sequential(
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.FC = nn.Sequential(
            nn.Linear(16, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.input(x)
        x = self.body(x)
        y = self.FC(x)

        return y

# function to calculate accuracy
def accuracy_function(prediction, y):
    _, classification = torch.max(prediction, 1)
    correct = (classification == y).float()
    accuracy = correct.sum() / len(correct)
    return accuracy

# function to run the training
def train(model, train_loader, optimizer, loss_function, device):
    model.train()
    total_batch = len(train_loader)
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        # print(x_batch)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(x_batch).squeeze(1)
        loss = loss_function(y_pred,y_batch)
        acc = accuracy_function(y_pred, y_batch)
        # back propgation
        loss.backward()
        optimizer.step()

def evaluate(model, evaluation_set, loss_function, device):
    total_loss, total_acc = 0,0
    model.eval() # dropout will be deactivated
    # no gradient will be calculated
    with torch.no_grad():
        # go over all batches in the set
        for x_batch, y_batch in evaluation_set:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch).squeeze(1)
            loss = loss_function(y_pred, y_batch)
            acc = accuracy_function(y_pred, y_batch)

            total_loss += loss.item()
            total_acc += acc.item()
    return total_loss/len(evaluation_set), total_acc/len(evaluation_set)

# for small model, cpu is good enough
# you could also use gpu if you won't, but that
# could cause you extra time by running extra PCl-e bus
device = 'cpu'

# construct the model, optimizer, and loss function
model = EyeData_ANN()
optimizer = optim.Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()

# send the model to device
# this is needed if one uses gpu to train, but 
# it can be omitted if cpu is used for train
model = model.to(device)
loss_function = loss_function.to(device)

# load in the dataset, which is esstential for builidng dataloader
training_dataset = EyeDataset(Eye_train)
testing_dataset = EyeDataset(Eye_test)

# setup dataloader, they passes the dataset batches by batches to the ANN model
train_load = DataLoader(dataset=training_dataset,batch_size=12,shuffle=True)
test_load = DataLoader(dataset=testing_dataset,batch_size=1,shuffle=True)

# perform training
# for each epoch, all observation will be gone through
# since the dataset is small, we uses a high number of 
# epoch. It is ok to decrease the number of epoch to 
# prevent overfitting
for epoch in range(25):
    ## During training, it is possible to see the loss goes up for some iteration
    ## this is perfectly normal. Since the optimizer uses a decay in the step size.
    ## The ANN model has a non-convex loss function, and the optimizer tends to jump
    ## between local min region when the step size is large.
    ## By convention, the model with best score in all epoch will used, not necessarily
    ## the one that has been trained by most epochs
    train(model,train_load,optimizer,loss_function, device)
    train_loss, train_acc = evaluate(model, train_load, loss_function, device)
    test_loss, test_acc = evaluate(model, test_load, loss_function, device)

    print('Epoch: {}'.format(epoch+1))
    print('Train Loss: {0:.3f} | Train Acc: {1:.2f}'.format(train_loss, train_acc))
    print('Val. Loss: {0:.3f} |  Val. Acc: {1:.2f}'.format(test_loss, test_acc))
