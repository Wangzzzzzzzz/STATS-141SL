import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
# other utils
import pandas as pd
import numpy as np
from tqdm import tqdm

Eye_train = pd.read_csv("ann_train.csv",index_col=0)
Eye_test = pd.read_csv("ann_test.csv",index_col=0)
Eye_train = Eye_train.sample(frac=1).reset_index(drop=True)
Eye_train["Sex"] = [1 if item == 'M' else 0 for item in Eye_train["Sex"]]
Eye_test = Eye_test.sample(frac=1).reset_index(drop=True)
Eye_test["Sex"] = [1 if item == 'M' else 0 for item in Eye_test["Sex"]]

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


class EyeData_ANN(nn.Module):
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

def accuracy_function(prediction, y):
    _, classification = torch.max(prediction, 1)
    correct = (classification == y).float()
    accuracy = correct.sum() / len(correct)
    return accuracy

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
    with torch.no_grad():
        for x_batch, y_batch in evaluation_set:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch).squeeze(1)
            loss = loss_function(y_pred, y_batch)
            acc = accuracy_function(y_pred, y_batch)

            total_loss += loss.item()
            total_acc += acc.item()
    return total_loss/len(evaluation_set), total_acc/len(evaluation_set)

device = 'cpu'

model = EyeData_ANN()
optimizer = optim.Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()

model = model.to(device)
loss_function = loss_function.to(device)

training_dataset = EyeDataset(Eye_train)
testing_dataset = EyeDataset(Eye_test)


train_load = DataLoader(dataset=training_dataset,batch_size=12,shuffle=True)
test_load = DataLoader(dataset=testing_dataset,batch_size=1,shuffle=True)

for epoch in range(25):

    train(model,train_load,optimizer,loss_function, device)
    train_loss, train_acc = evaluate(model, train_load, loss_function, device)
    test_loss, test_acc = evaluate(model, test_load, loss_function, device)

    print('Epoch: {}'.format(epoch+1))
    print('Train Loss: {0:.3f} | Train Acc: {1:.2f}'.format(train_loss, train_acc))
    print('Val. Loss: {0:.3f} |  Val. Acc: {1:.2f}'.format(test_loss, test_acc))
