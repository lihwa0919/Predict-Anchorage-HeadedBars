#Optuna를 통해서 하이퍼파라미터 변수 적용한 클래스, Optuna과정에서 Kfold로 학습 시킴 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import os
import openpyxl
from torch.utils.data import TensorDataset, DataLoader,random_split, Dataset
import random 

class ANN_batch_1(nn.Module):
    #torch.cuda.manual_seed_all(42)
    def __init__(self, input_size, hidden_size1, output_size):
        super(ANN_batch_1, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1) #배치 정규화 과정_?
        # 각 층의 과정 정규화 
        # 내부 공변량변화 문제 완화_ 신경망의 각 층을 지날 때 마다 입력 데이터의 분포가 변화  
        
        self.layer2 = nn.Linear(hidden_size1, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.layer2(x)
        return x
ANN1_batch_size = 32


class ANN_batch_2(nn.Module):
    #torch.cuda.manual_seed_all(42)
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ANN_batch_2, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1) #배치 정규화 과정_?
        # 각 층의 과정 정규화 
        # 내부 공변량변화 문제 완화_ 신경망의 각 층을 지날 때 마다 입력 데이터의 분포가 변화  
        
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        
        self.layer3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.layer3(x)
        return x
ANN2_batch_size = 32    

class ANN_batch_3(nn.Module):
    #torch.cuda.manual_seed_all(42)
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(ANN_batch_3, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1) #배치 정규화 과정_?
        # 각 층의 과정 정규화 
        # 내부 공변량변화 문제 완화_ 신경망의 각 층을 지날 때 마다 입력 데이터의 분포가 변화  
        
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        
        self.layer3 = nn.Linear(hidden_size2, hidden_size3)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        
        self.layer4 = nn.Linear(hidden_size3, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.layer4(x)
        return x
ANN3_batch_size = 32    

class ANN_batch_4(nn.Module):
    #torch.cuda.manual_seed_all(42)
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size):
        super(ANN_batch_4, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1) #배치 정규화 과정_?
        # 각 층의 과정 정규화 
        # 내부 공변량변화 문제 완화_ 신경망의 각 층을 지날 때 마다 입력 데이터의 분포가 변화  
        
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        
        self.layer3 = nn.Linear(hidden_size2, hidden_size3)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        
        self.layer4 = nn.Linear(hidden_size3, hidden_size4)
        self.bn4 = nn.BatchNorm1d(hidden_size4)
        
        self.layer5 = nn.Linear(hidden_size4, output_size)
        self.relu = nn.ReLU()
    
        
    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.relu(self.bn4(self.layer4(x)))
        x = self.layer5(x)
        return x
ANN4_batch_size = 64
