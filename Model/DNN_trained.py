# DNN 모델 학습 후 가중치 저장 
from Train_Test_dataset import train_dataset, val_dataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
import ANNs_class as ANN
from joblib import dump

def ann_training(device, model, criterion, optimizer, nb_epochs, train_dataloader, validation_dataloader):
    torch.manual_seed(123)
    
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_wts = None
    best_epoch = -1
    patience_counter = 0
    for epoch in range(nb_epochs + 1):
        model.train()  # 모델을 훈련 모드로 설정
        train_loss = 0

        for data in train_dataloader:
            optimizer.zero_grad()
            
            x, y = data
            x = x.to(device)
            y_train = y.to(device)
            p_train = model(x)
            #condition = p_train < 0
            #loss = criterion(p_train, y_train)
            #zero_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
            #train_cost = torch.where(condition, zero_loss, loss).mean()
            
            train_cost = criterion(p_train, y_train) # loss
            #print(train_cost.dim()) train cost 값이 스칼라 텐서인지 확인 
            train_cost.backward(retain_graph = True)
            optimizer.step()
            
            train_loss += train_cost.item()
        
        # 모델 검증
        model.eval()  # 모델을 평가 모드로 설정
        val_loss = 0
        with torch.no_grad():
            for data in validation_dataloader:
                x, y = data
                x = x.to(device)
                y_val = y.to(device)
                p_val = model(x)
                val_cost = criterion(p_val, y_val)
                
                val_loss += val_cost.item()
        
        # calculate mean for each batch
        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(validation_dataloader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
    
        if epoch % 100 == 0:
            print(f"Epoch: {epoch:4d}/{nb_epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")
        
        # Check for best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = model.state_dict().copy()
            best_epoch = epoch
            patience_counter = 0
            print(f"New best model found at epoch {epoch} with val loss {avg_val_loss:.6f}")
        else:
            patience_counter += 1
        #if patience_counter >= patience:
         #   print(f"Early stopping triggered at epoch {epoch}.")
          #  break
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    print(f"Best model was found at epoch {best_epoch} with validation loss {best_val_loss:.6f}")
    print(f' Best model wts : \n {best_model_wts}')
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0.0, 0.01)
    plt.title(f'Train and Validation Loss over Epochs')
    plt.legend()
    plt.show()
    
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nb_epochs = 1000
generator=torch.Generator().manual_seed(44) #42

criterion1 = nn.MSELoss().to(device) 
model1 = ANN.ANN_batch_1(14,100,1).to(device).double()
optimizer1 = torch.optim.Adam(model1.parameters(), lr =0.0006607488696945926, weight_decay=0) #
ann1_train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True ,generator=generator) # 데이터 고정
ann1_validation_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True ,generator=generator) #데이터 고정

# optuna before :  {'hidden_size1': 84, 'hidden_size2': 42, 'lr': 0.021209527983720448, 'batch_size': 32}
#ann2 Best parameters ann4 model found:  {'hidden_size1': 84, 'hidden_size2': 9, 'lr': 0.0014176780199220882, 'batch_size': 16}
criterion2 = nn.MSELoss().to(device) 
model2 = ANN.ANN_batch_2(14,84,9,1).to(device).double()
optimizer2 = torch.optim.Adam(model2.parameters(), lr = 0.00014176780199220882, weight_decay=0) #
ann2_train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True , generator=generator) # 데이터 고정
ann2_validation_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, generator=generator) #데이터 고정

#optuna before : {'hidden_size1': 42, 'hidden_size2': 79, 'hidden_size3': 25, 'lr': 0.08139841930700309, 'batch_size': 32}
#ann3 :  {'hidden_size1': 70, 'hidden_size2': 8, 'hidden_size3': 4, 'lr': 0.0029387120402525296, 'batch_size': 32}
criterion3 = nn.MSELoss().to(device) 
model3 = ANN.ANN_batch_3(14,70,8,4,1).to(device).double()
optimizer3 = torch.optim.Adam(model3.parameters(), lr = 0.00029387120402525296, weight_decay=0) #
ann3_train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, generator=generator) # 데이터 고정
ann3_validation_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, generator=generator) #데이터 고정

criterion4 = nn.MSELoss().to(device) 
model4 = ANN.ANN_batch_4(14,83,39,49,83,1).to(device).double()
optimizer4 = torch.optim.Adam(model4.parameters(), lr =  0.03444508252211158) #
ann4_train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, generator=generator) # 데이터 고정
ann4_validation_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, generator=generator) #데이터 고정n4_validation_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, generator=generator) #데이터 고정

# 모델 학습 
trained_ann1= ann_training(device, model1, criterion1, optimizer1, nb_epochs, ann1_train_dataloader, ann1_validation_dataloader)
trained_ann2 = ann_training(device, model2, criterion2, optimizer2, nb_epochs, ann2_train_dataloader, ann2_validation_dataloader)
trained_ann3 = ann_training(device, model3, criterion3, optimizer3, nb_epochs, ann3_train_dataloader, ann3_validation_dataloader)
trained_ann4 = ann_training(device, model4, criterion4, optimizer4, nb_epochs, ann4_train_dataloader, ann4_validation_dataloader)

# 학습된 모델 가중치 저장
torch.save(trained_ann4.state_dict(), "trained_model/DNN4_90_new.pt") 
#torch.save(trained_ann3.state_dict(), "trained_model/DNN3.pt")
#torch.save(trained_ann2.state_dict(), "trained_model/DNN2.pt")
#torch.save(trained_ann1.state_dict(), "trained_model/DNN1.pt")

