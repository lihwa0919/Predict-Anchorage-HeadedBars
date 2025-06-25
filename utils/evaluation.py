#%%
from utils.Train_Test_dataset import y_scaler, x_scaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score, mean_squared_error
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ACI(x_data): #x_data = sc_tn
    x_data_unsc_tn = x_scaler.inverse_transform(x_data)
    fcm = x_data_unsc_tn[:,2]
    Ld = x_data_unsc_tn[:,1]
    db = x_data_unsc_tn[:,3]
    aci_y_unsc = (Ld*torch.sqrt(fcm))/(0.19*db)
    #aci_y_sc  = y_scaler.transform(aci_y_unsc)
    return aci_y_unsc # output type : unscaled tensor

def plot_ld_db(x_sc_tn_data, y_sc_tn_data, model, model_type):
    if model_type == 'ANN4':
        test_y_unsc_tn = y_scaler.inverse_transform(y_sc_tn_data)
        test_y_unsc_np = test_y_unsc_tn.detach().cpu().numpy()
        x_sc_tn_data = x_sc_tn_data.to(device)
        pred_sc_tn = model(x_sc_tn_data)  # Size: [54, 1]
        pred_unsc_tn = y_scaler.inverse_transform(pred_sc_tn)  # Inverse scaling
        predict_np = pred_unsc_tn.detach().cpu().numpy()
        x_data_unsc_tn = x_scaler.inverse_transform(x_sc_tn_data)
        # Extracting Ld and db values
        Ld = x_data_unsc_tn[:, 1].detach().cpu().numpy()
        #print(Ld)
        db = x_data_unsc_tn[:, 3].detach().cpu().numpy()
        #print(db)
        # Calculate x and y values for the scatter plot
        x = Ld / db
        y = test_y_unsc_np / predict_np
        
        show_df = pd.DataFrame({
            'Ld/db' :x.squeeze(),
            'Label / Predict': y.squeeze(),
            'Predict': predict_np.squeeze(),
            'Label': test_y_unsc_np.squeeze(),
            
        })

        plt.figure(figsize=(6, 7))
        plt.scatter(x, y, color='white',edgecolors='DarkBlue', alpha=0.9, label='Data Points')
        plt.axhline(y=1, color='Black', linestyle='--', alpha = 0.5,label='y = 1 ')
        plt.xlabel('Ld / db')
        plt.ylabel('Test / Predicted')
        plt.title(f'Test values / {model_type} predicted values')
        #plt.legend()
        plt.grid(True)
        plt.ylim(0,3.0)
        plt.show()
        return show_df.round(1)
        
    elif model_type == 'ACI':
        test_y_unsc_tn = y_scaler.inverse_transform(y_sc_tn_data)
        aci_y_unsc_tn = ACI(x_sc_tn_data)
        y_aci_unsc_np = aci_y_unsc_tn.detach().cpu().numpy()
        y_aci_unsc = y_aci_unsc_np.reshape(-1,1)
        y_real_unsc = test_y_unsc_tn.detach().cpu().numpy()
        y_real_unsc = y_real_unsc.reshape(-1,1)
        x_data_unsc_tn = x_scaler.inverse_transform(x_sc_tn_data)
        # Extracting Ld and db values
        Ld = x_data_unsc_tn[:, 1].detach().cpu().numpy()
        #print(Ld)
        db = x_data_unsc_tn[:, 3].detach().cpu().numpy()
        #print(db)
        # Calculate x and y values for the scatter plot
        x = Ld / db
        y =y_real_unsc / y_aci_unsc
        show_df = pd.DataFrame({
            'Ld/db' :x.squeeze(),
            'Label / Predict': y.squeeze(),
            'Predict': y_aci_unsc.squeeze(),
            'Label': y_real_unsc.squeeze(),
        })
        
        
        plt.figure(figsize=(6, 7))
        plt.scatter(x, y, color='white',edgecolors='DarkBlue', alpha=0.9, label='Data Points')
        plt.axhline(y=1, color='Black', linestyle='--', alpha = 0.5,label='y = 1 ')
        plt.xlabel('Ld / db')
        plt.ylabel('Test / Predicted')
        plt.title(f'Test values / {model_type} predicted values')
        #plt.legend()
        plt.grid(True)
        plt.ylim(0,3.0)
        plt.show()
        return show_df.round(1)
    else:
        test_y_unsc_tn = y_scaler.inverse_transform(y_sc_tn_data)
        test_x_unsc_tn = x_scaler.inverse_transform(x_sc_tn_data)
        
        test_y_unsc_np = test_y_unsc_tn.detach().cpu().numpy()
        test_x_sc_np = x_sc_tn_data.detach().cpu().numpy()
        test_x_unsc_np = test_x_unsc_tn.detach().cpu().numpy()
        
        pred_sc_np = model.predict(test_x_sc_np)
        pred_sc_tn = torch.FloatTensor(pred_sc_np)
        pred_unsc_tn = y_scaler.inverse_transform(pred_sc_tn)
        pred_unsc_np = pred_unsc_tn.detach().cpu().numpy()
        pred_unsc_np = pred_unsc_np.reshape(-1,1)
        
        Ld = test_x_unsc_tn[:, 1].detach().cpu().numpy()
        #print(Ld)
        db = test_x_unsc_tn[:, 3].detach().cpu().numpy()
        #print(db)
        # Calculate x and y values for the scatter plot
        x = Ld / db
        y =test_y_unsc_np / pred_unsc_np
        
        show_df = pd.DataFrame({
            'Ld/db' :x.squeeze(),
            'Label / Predict': y.squeeze(),
            'Predict': pred_unsc_np.squeeze(),
            'Label': test_y_unsc_np.squeeze(),
        })
        
        plt.figure(figsize=(6, 7))
        plt.scatter(x, y, color='white',edgecolors='DarkBlue', alpha=0.9, label='Data Points')
        plt.axhline(y=1, color='Black', linestyle='--', alpha = 0.5,label='y = 1 ')
        plt.xlabel('Ld / db')
        plt.ylabel('Test / Predicted')
        plt.title(f'Test values / {model_type} predicted values')
        #plt.legend()
        plt.grid(True)
        plt.ylim(0,3.0)
        plt.show()
        return show_df.round(1)
    
def print_model_performance(model_name, model, x_sc_tn_data, y_sc_tn_data, device,y_scaler , title):
    if model_name == 'ANN4':
        test_y_unsc_tn = y_scaler.inverse_transform(y_sc_tn_data)
        test_y_unsc_np = test_y_unsc_tn.detach().cpu().numpy()
        
        x_sc_tn_data = x_sc_tn_data.to(device)
        pred_sc_tn = model(x_sc_tn_data) #torch.Size([54, 1])
        pred_unsc_tn = y_scaler.inverse_transform(pred_sc_tn) #torch.Size([54, 1])
        div = test_y_unsc_tn/pred_unsc_tn #torch.Size([54, 1])
        cov = torch.std(div)/torch.mean(div) #torch.Size([])
        cov = cov.item()
        predict_np = pred_unsc_tn.detach().cpu().numpy()
        r2 = r2_score(test_y_unsc_np, predict_np)
        
        mse = torch.mean((test_y_unsc_tn - pred_unsc_tn) ** 2)
        rmse = torch.sqrt(mse)
        rmse = rmse.cpu().detach().numpy()
        mape = torch.mean(torch.abs((test_y_unsc_tn - pred_unsc_tn) / test_y_unsc_tn))
        mape = mape.cpu().detach().numpy()
        mae = torch.mean(torch.abs(test_y_unsc_tn - pred_unsc_tn))
        mae = mae.cpu().detach().numpy()

        #y_aci_unsc = ACI(test_x_sc_tn)
        #y_aci_unsc = y_aci_unsc.detach().cpu().numpy()
        return {'Title':title,
            'cov':cov,
            'rmse':rmse,
            'mape':mape,
            'mae':mae,
            'r2' : r2
            }
        
    else:
        test_y_unsc_tn = y_scaler.inverse_transform(y_sc_tn_data)
        test_y_unsc_np = test_y_unsc_tn.detach().cpu().numpy()
        test_x_sc_np = x_sc_tn_data.detach().cpu().numpy()
        pred_sc_np = model.predict(test_x_sc_np)
        pred_sc_tn = torch.FloatTensor(pred_sc_np)
        pred_unsc_tn = y_scaler.inverse_transform(pred_sc_tn)
        pred_unsc_np = pred_unsc_tn.detach().cpu().numpy()
        pred_unsc_np = pred_unsc_np.reshape(-1,1)
        
        div = test_y_unsc_np/pred_unsc_np 
        cov = np.std(div)/np.mean(div)
        r2 = r2_score(test_y_unsc_np, pred_unsc_np)
        
        mse = mean_squared_error(test_y_unsc_np, pred_unsc_np)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((test_y_unsc_np - pred_unsc_np)/test_y_unsc_np))
        mae = np.mean(np.abs(test_y_unsc_np - pred_unsc_np))
        
        #y_aci_unsc_tn = ACI(x_sc_tn_data)
        #y_aci_unsc_np = y_aci_unsc_tn.detach().cpu().numpy()
        return {'Title': title,
            'cov':cov,
            'rmse':rmse,
            'mape':mape,
            'mae':mae,
            'r2' : r2
            }
        
def output_result_df(x_sc_tn_data, y_sc_tn_data, model, model_type):
    if model_type == 'ANN4':
        test_y_unsc_tn = y_scaler.inverse_transform(y_sc_tn_data)
        test_y_unsc_np = test_y_unsc_tn.detach().cpu().numpy()
        x_sc_tn_data = x_sc_tn_data.to(device)
        pred_sc_tn = model(x_sc_tn_data)  # Size: [54, 1]
        pred_unsc_tn = y_scaler.inverse_transform(pred_sc_tn)  # Inverse scaling
        predict_np = pred_unsc_tn.detach().cpu().numpy()
        x_data_unsc_tn = x_scaler.inverse_transform(x_sc_tn_data)
        # Extracting Ld and db values
        Ld = x_data_unsc_tn[:, 1].detach().cpu().numpy()
        #print(Ld)
        db = x_data_unsc_tn[:, 3].detach().cpu().numpy()
        #print(db)
        # Calculate x and y values for the scatter plot
        x = Ld / db
        y = test_y_unsc_np / predict_np
        
        show_df = pd.DataFrame({
            'Ld/db' :x.squeeze(),
            'Label / Predict': y.squeeze(),
            'Predict': predict_np.squeeze(),
            'Label': test_y_unsc_np.squeeze(),
            
        })

        return show_df.round(1)
        
    elif model_type == 'ACI':
        test_y_unsc_tn = y_scaler.inverse_transform(y_sc_tn_data)
        aci_y_unsc_tn = ACI(x_sc_tn_data)
        y_aci_unsc_np = aci_y_unsc_tn.detach().cpu().numpy()
        y_aci_unsc = y_aci_unsc_np.reshape(-1,1)
        y_real_unsc = test_y_unsc_tn.detach().cpu().numpy()
        y_real_unsc = y_real_unsc.reshape(-1,1)
        x_data_unsc_tn = x_scaler.inverse_transform(x_sc_tn_data)
        # Extracting Ld and db values
        Ld = x_data_unsc_tn[:, 1].detach().cpu().numpy()
        #print(Ld)
        db = x_data_unsc_tn[:, 3].detach().cpu().numpy()
        #print(db)
        # Calculate x and y values for the scatter plot
        x = Ld / db
        y =y_real_unsc / y_aci_unsc
        show_df = pd.DataFrame({
            'Ld/db' :x.squeeze(),
            'Label / Predict': y.squeeze(),
            'Predict': y_aci_unsc.squeeze(),
            'Label': y_real_unsc.squeeze(),
        })
        
        return show_df.round(1)
    else:
        test_y_unsc_tn = y_scaler.inverse_transform(y_sc_tn_data)
        test_x_unsc_tn = x_scaler.inverse_transform(x_sc_tn_data)
        
        test_y_unsc_np = test_y_unsc_tn.detach().cpu().numpy()
        test_x_sc_np = x_sc_tn_data.detach().cpu().numpy()
        test_x_unsc_np = test_x_unsc_tn.detach().cpu().numpy()
        
        pred_sc_np = model.predict(test_x_sc_np)
        pred_sc_tn = torch.FloatTensor(pred_sc_np)
        pred_unsc_tn = y_scaler.inverse_transform(pred_sc_tn)
        pred_unsc_np = pred_unsc_tn.detach().cpu().numpy()
        pred_unsc_np = pred_unsc_np.reshape(-1,1)
        
        Ld = test_x_unsc_tn[:, 1].detach().cpu().numpy()
        #print(Ld)
        db = test_x_unsc_tn[:, 3].detach().cpu().numpy()
        #print(db)
        # Calculate x and y values for the scatter plot
        x = Ld / db
        y =test_y_unsc_np / pred_unsc_np
        
        show_df = pd.DataFrame({
            'Ld/db' :x.squeeze(),
            'Label / Predict': y.squeeze(),
            'Predict': pred_unsc_np.squeeze(),
            'Label': test_y_unsc_np.squeeze(),
        })
        
        return show_df.round(1)
    
    
# %%
