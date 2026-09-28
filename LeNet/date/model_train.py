import torch
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as  Data
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet

def train_val_date_process():
    train_date = FashionMNIST(root='./date',
                        train=True,
                        transform = transforms.Compose([transforms.Resize(size=224),transforms.ToPILImage()]),
                        download=True)
    train_date,val_data = Data.random_split(train_date,lengths=[round(0.8*len(train_date)),round(0.2*len(train_date))])

    train_dataloader = Data.DataLoader(dataset = train_date,
                                       batch_size= 128,
                                       shuffle= True,
                                       num_workers = 8)

    val_dataloader= Data.DataLoader(dataset=val_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=8)

    return train_dataloader, val_dataloader

def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    device = torch.device("cuda" if torch.cuda.is+available() else cpu)

    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.stte_dict())

    best_acc = 0.0
    train_loss_all  =[]
    val_loss_all = []
    train_acc_all  =[]
    val_acc_all = []

    since = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")
        print("-"*10)

        
        train_loss = 0.0
        train_corrects = 0

        val_loss = 0.0
        val_corrects = 0
        
        train_num  = 0
        val_num = 0

        for step,(b_x,b_y) in enumerate(train_dataloader):
            # 特征
            # 标签
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.train()

            output = model(b_x)