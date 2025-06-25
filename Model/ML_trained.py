# sklearn, xgb 라이브러리 사용하여 ML 학습 시킨 뒤 가중치 저장 
from Train_Test_dataset import  ML_train_x_sc_np, ML_train_y_sc_np
from joblib import dump
import joblib

#DT 
from sklearn.tree import DecisionTreeRegressor
tree_regressor = DecisionTreeRegressor(random_state=42, max_depth=20, max_features='sqrt', min_samples_leaf= 1, min_samples_split= 10)  
#모델 학습 후 저장 
tree_regressor.fit(ML_train_x_sc_np, ML_train_y_sc_np)
joblib.dump(tree_regressor, 'trained_model/tree_regressor_new.pkl')
# 학습된 가중치 불러오기 
#tree_regressor = joblib.load('trained_model/tree_regressor.pkl')

#KNN
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor( algorithm= 'auto', n_neighbors= 3, p =1, weights= 'uniform')  # distance 로 하면 과적합 나옴 
knr.fit(ML_train_x_sc_np, ML_train_y_sc_np)
joblib.dump(knr, 'trained_model/knr_new.pkl')
#knr = joblib.load('trained_model/knr.pkl')

#RF
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(max_depth= 10, max_features= 'auto', n_estimators=900, random_state=42)
rf_regressor.fit(ML_train_x_sc_np, ML_train_y_sc_np)
joblib.dump(rf_regressor, 'trained_model/rf_regressor_new.pkl')
#rf_regressor = joblib.load('trained_model/rf_regressor.pkl')

#XGB
import xgboost as xgb
xgb_model = xgb.XGBRegressor(random_state = 42, learning_rate = 0.4, max_depth = 2, n_estimators = 100, objective='reg:logistic')
xgb_model.fit(ML_train_x_sc_np, ML_train_y_sc_np)
joblib.dump(xgb_model, 'trained_model/xgb_model_new.pkl')
#xgb_model = joblib.load('trained_model/xgb_model.pkl')

#SVR
from sklearn.svm import SVR 
svr_model = SVR( C = 200, degree = 1, gamma = 0.1, kernel= 'rbf', epsilon=0.05)
svr_model.fit(ML_train_x_sc_np, ML_train_y_sc_np)
joblib.dump(svr_model, 'trained_model/svr_model_new.pkl')
#svr_model = joblib.load('trained_model/svr_model.pkl')

