is_Import = True
path = "Cat.jpg"
from cProfile import label
from time import sleep
from PIL import Image #PIL form (W,H,C)
import matplotlib.pyplot as plt
import os
if is_Import :
    import numpy as np
    import torch 
    import torch.nn as nn
    import torchvision.transforms as transforms
    
    from torch.utils.data import ConcatDataset, DataLoader, Subset
    from torchvision.datasets import DatasetFolder
    from tqdm.auto import tqdm
#--------------------------------------------------------
#Traning Parameter
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
n_epoch = 10
learning_rate = 0.001
#train_data_path ="./dataset/training_set" #cat 995 dot 1000 1995
train_data_path ="./dataset/train"
test_data_path =  "./dataset/test"
#Data Pre-Processing
train_tfm = transforms.Compose([
    transforms.Resize([128,128]),
    transforms.ToTensor(),
    ])
test_tfm = transforms.Compose([
    transforms.Resize([128,128]),
    transforms.ToTensor(),
    ])
#--------------------------------------------------------
#Data Loader
train_data = DatasetFolder(train_data_path,loader=lambda x: Image.open(x),
                           extensions="jpg", transform=train_tfm)
test_data = DatasetFolder(test_data_path,loader=lambda x: Image.open(x),
                           extensions="jpg", transform=test_tfm)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True
                          , pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=True)

#--------------------------------------------------------
#Module Layer
class Classifer(nn.Module): 
    def __init__(self):
        super(Classifer,self).__init__()
        self.cnn_layers = nn.Sequential(
            
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
            
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            
            nn.ReLU(),
            nn.Linear(256, 2),
            
        )
        
    def forward(self, x): 
        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x

if __name__ == '__main__':
    #model
    model = Classifer().to(device)
    model.device = device

    #loss
    criterion = nn.CrossEntropyLoss()

    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    #traning
    model.train()
    train_loss = []
    train_accs = []

    for epoch in range(n_epoch):
        for batch in tqdm(train_loader):

            imgs, labels = batch
            
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            print(logits)
            optimizer.zero_grad()
            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            train_loss.append(loss.item())
            train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    print(f"[ Train | {epoch + 1:03d}/{n_epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")


    
    valid_loss = []
    valid_accs = []
    model.eval()
    for batch in tqdm(test_loader):
            
            imgs, labels = batch
            #print(labels)
            logits = model(imgs.to(device))
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            loss = criterion(logits, labels.to(device))

            valid_loss.append(loss.item())
            valid_accs.append(acc)
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    #print("cou is ",cou)
    print(f"[ Valid | {epoch + 1:03d}/{n_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

