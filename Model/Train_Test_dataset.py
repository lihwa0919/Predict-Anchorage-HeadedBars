#%%
#균일한 train, test 데이터 셋 사용하기 위해 
from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import TensorDataset, DataLoader,random_split, Dataset, Subset
import random 

def seed_everything(seed:int = 1004):
    random.seed(seed) #
    np.random.seed(seed) #
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # current gpu seed
    #torch.cuda.manual_seed_all(seed) # All gpu seed

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
seed_everything(40)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cuda" if torch.cuda.is_available() else "cpu"

## 1. 데이터 불러오기 
df1 = pd.read_excel("/home/kds/VSProjects/HeadedBars_231030/Final_file_share/used_data/Data for headed bars_for DataFrame_220725.xlsx", skiprows = 17, engine = 'openpyxl', sheet_name= 'headed (2)' )


df = pd.DataFrame(df1, columns = ["No.", "Author", "Year", "Test type", "Remark", "Specimen", "fy", "Ld", "fcm", "db", "b", "cos,avg",
                                 "cth", "ch", "Nh", "Bottom cover", "Ah/Ab", "Fsu at La, test", "dtr", "Ntr", "st"]) # st 제거시


X = df[["fy", "Ld", "fcm", "db", "b", "cos,avg", "cth", "ch", "Nh", "Bottom cover","st", "Ah/Ab",'Ntr','dtr', "Test type", "Fsu at La, test",]] 

X = X.dropna() # null 있는 행 삭제 
X = X[X["Fsu at La, test"] != 0] # fsu =0 값 삭제 

# test_type 종류별 갯수
count_by_category = X['Test type'].value_counts()
#print(count_by_category)

X_Joint_type = X[X["Test type"] == "Joint type"]
X_Joint = X_Joint_type.drop("Test type", axis= 1)
## 이상치 제거 
#XX = X_Joint[X_Joint['Fsu at La, test'] <=120]
#X_Joint = X_Joint.drop(XX.index)
# 사용한 데이터만 출력 하여 따로 저장하기 위한 df
used_data = pd.DataFrame(df1)
used_data = df1.iloc[X_Joint.index.tolist()]
used_data.to_excel('Used_data.xlsx', sheet_name='Sheet1', index=False) # 오리진 프로에 그림 그리기 위한 데이터 


##
X_Joint1 = X_Joint.copy()

X_Joint_for_split = X_Joint.drop('Fsu at La, test', axis = 1) # 270*14(columns만 존재 ) 

Y_Joint_for_split = X_Joint['Fsu at La, test'] #Series 
Y_Joint_for_split = Y_Joint_for_split.values.reshape(-1,1)

class MinMaxScaler_tn:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit(self, tensor_data):
        tensor_data = tensor_data.to(device)
        self.min_val = torch.min(tensor_data, dim=0).values
        self.max_val = torch.max(tensor_data, dim=0).values

    def transform(self, tensor_data):
        tensor_data = tensor_data.to(device)
        # Transform using min_val and max_val
        scaled_data = (tensor_data - self.min_val) / (self.max_val - self.min_val)
        return scaled_data

    def inverse_transform(self, scaled_tensor):
        scaled_tensor = scaled_tensor.to(device)
        # Inverse transform using min_val and max_val
        restored_data = scaled_tensor * (self.max_val - self.min_val) + self.min_val
        return restored_data
    def print_minmax(self):
        print(self.min_val)
        print(self.max_val)

def create_train_val_test_datasets(x_tensor_sc, y_tensor_sc, train_ratio=0.8,  seed=42):
    x_tensor = x_tensor_sc.to(device)
    y_tensor = y_tensor_sc.to(device)
    dataset = TensorDataset(x_tensor, y_tensor)
    
    generator = torch.Generator().manual_seed(seed)
    dataset_size = len(dataset) #267
    train_size = int(dataset_size * train_ratio) #213
    test_size = dataset_size - train_size #54
    # [train : test]
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator) # 비율이 아닌 샘플수를 입력으로 받음 
    test_dataset_size = len(test_dataset)
    val_size = int(test_dataset_size * train_ratio)
    val_size = test_size - val_size
    #  [ : val]
    only_test_dataset, val_dataset = random_split(test_dataset, [34, 20], generator=generator)

    return train_dataset, val_dataset, test_dataset

x_scaler = MinMaxScaler_tn()
y_scaler = MinMaxScaler_tn()
x_scaler.fit(torch.FloatTensor(X_Joint_for_split.values))
y_scaler.fit(torch.FloatTensor(Y_Joint_for_split))

# 전체 데이터 먼저 스케일 
X_Joint_for_split_sc_tn = x_scaler.transform(torch.tensor(X_Joint_for_split.values, dtype = torch.float64, requires_grad = True)) # torch.Size([270, 14])
Y_Joint_for_split_sc_tn = y_scaler.transform(torch.tensor(Y_Joint_for_split, dtype = torch.float64, requires_grad = False)) # torch.Size([270])
# 스케일된 전체 데이터 학습, 검증, 테스트로  분류하여 데이터셋으로 생성


train_dataset, val_dataset, test_dataset = create_train_val_test_datasets( X_Joint_for_split_sc_tn, Y_Joint_for_split_sc_tn)  # train_val_dataset 이 데이터는 ml 학습 용으로 사용

# 분류된 데이터셋을 다시 데이터 프레임으로 변형하기 
train_dataset_size = list(range(len(train_dataset)))
val_dataset_size = list(range(len(val_dataset)))
test_dataset_size = list(range(len(test_dataset)))

train_subset = Subset(train_dataset, train_dataset_size)
val_subset = Subset(val_dataset, val_dataset_size)
test_subset = Subset(test_dataset, test_dataset_size)

train_subset_list = [train_dataset[idx] for idx in train_subset.indices]
test_subset_list = [test_dataset[idx] for idx in test_subset.indices]
val_subset_list = [val_dataset[idx] for idx in val_subset.indices]

train_data_array = x_scaler.inverse_transform(torch.stack([item[0] for item in train_subset_list])).detach().cpu().numpy()
train_labels_array = y_scaler.inverse_transform(torch.stack([item[1] for item in train_subset_list])).detach().cpu().numpy()

test_data_array = x_scaler.inverse_transform(torch.stack([item[0] for item in test_subset_list])).detach().cpu().numpy()
test_labels_array = y_scaler.inverse_transform(torch.stack([item[1] for item in test_subset_list])).detach().cpu().numpy()

val_data_array = x_scaler.inverse_transform(torch.stack([item[0] for item in val_subset_list])).detach().cpu().numpy()
val_labels_array = y_scaler.inverse_transform(torch.stack([item[1] for item in val_subset_list])).detach().cpu().numpy()

df_columns = ["fy", "Ld", "fcm", "db", "b", "cos,avg", "cth", "ch", "Nh", "Bottom cover","st", "Ah/Ab",'Ntr','dtr']
train_df = pd.DataFrame(train_data_array, columns=[name for name in df_columns])
train_df["Fsu at La, test"] = train_labels_array

val_df = pd.DataFrame(val_data_array, columns=[name for name in df_columns])
val_df["Fsu at La, test"] = val_labels_array

test_df = pd.DataFrame(test_data_array, columns=[name for name in df_columns])
test_df["Fsu at La, test"] = test_labels_array


train_x_sc_tn = torch.stack([sample[0] for sample in train_dataset]) # torch.Size([216,14])
train_y_sc_tn = torch.stack([sample[1] for sample in train_dataset]) # torch.Size([216,1])
val_x_sc_tn = torch.stack([sample[0] for sample in val_dataset]) # torch.Size([27,14])
val_y_sc_tn = torch.stack([sample[1] for sample in val_dataset]) # torch.Size([27,1])
test_x_sc_tn = torch.stack([sample[0] for sample in test_dataset]) # torch.Size([27,14])
test_y_sc_tn = torch.stack([sample[1] for sample in test_dataset]) # torch.Size([27,1])
ML_train_x_sc_tn = torch.stack([sample[0] for sample in train_dataset])
ML_train_y_sc_tn = torch.stack([sample[1] for sample in train_dataset])


train_x_unsc_tn= x_scaler.inverse_transform(train_x_sc_tn)
train_y_unsc_tn = y_scaler.inverse_transform(train_y_sc_tn)
val_x_unsc_tn = x_scaler.inverse_transform(val_x_sc_tn)
val_y_unsc_tn = y_scaler.inverse_transform(val_y_sc_tn)

test_x_unsc_tn = x_scaler.inverse_transform(test_x_sc_tn)
test_y_unsc_tn = y_scaler.inverse_transform(test_y_sc_tn) 
ML_train_x_unsc_tn = x_scaler.inverse_transform(ML_train_x_sc_tn)
ML_train_y_unsc_tn = y_scaler.inverse_transform(ML_train_y_sc_tn)

#ML에 필요한 np.array 형태
ML_train_x_sc_np = ML_train_x_sc_tn.detach().cpu().numpy()
ML_train_y_sc_np = ML_train_y_sc_tn.detach().cpu().numpy()

test_x_sc_np = test_x_sc_tn.detach().cpu().numpy()
test_y_sc_np = test_y_sc_tn.detach().cpu().numpy()

ML_train_x_unsc_np = ML_train_x_unsc_tn.detach().cpu().numpy()
ML_train_y_unsc_np = ML_train_y_unsc_tn.detach().cpu().numpy()
    
# %%
