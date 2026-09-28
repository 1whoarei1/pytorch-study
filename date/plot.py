from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as  Date
import numpy as np


train_date = FashionMNIST(root='./date',
                        train=True,
                        transform = transforms.Compose([transforms.Resize(size=224),transforms.ToPILImage()]),
                        download=True)

train_loader = Date.DataLoader(dataset=train_date,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)