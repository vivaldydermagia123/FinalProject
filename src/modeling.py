import os 
import sys
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from dataclasses import dataclass 

# Tambahkan root folder Spam-Message-Detection ke sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import save_model, write_metrics

# Column Definitions
ohe_columns = ['FUNCTION']
transform_columns = ['CONDUCT LIMIT']
label_columns = ['BACTERIA_STATUS']
num_columns = ['DAY_RUN', 'RR', 'AVG_T_RETURN', 'AVG_FRC', 'AVG_CHLORIDE']
material_columns = ['MATERIAL']
target_columns = ['NAOCL']

@dataclass
class ModelConfig():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(BASE_DIR, "artifacts")
    model_path = os.path.join(artifacts_dir, "model.pkl")
    evaluation_path = os.path.join(artifacts_dir, "metrics.txt")
    prediction_path = os.path.join(artifacts_dir, "Train_prediction.csv")

class ModelTrain():
    def __init__(self):
        self.model_config = ModelConfig()

    def split(self, df):
        X = df.drop(columns=target_columns)
        y = df[target_columns]

        return X, y 
    
    def modeling(self, df):
        # read data
        data = pd.read_csv(df)
        X, y = self.split(data)

        # create model
        xg_params = {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 3, 'n_estimators': 100, 'subsample': 0.8}
        lgbm_params = {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 5, 'min_child_samples': 10, 'n_estimators': 100, 'num_leaves': 31, 'reg_alpha': 0.1, 'reg_lambda': 0.5, 'subsample': 0.8}
        rf_params = {'bootstrap': False, 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 50}

        base_model = [
                ('rf', RandomForestRegressor(**rf_params, random_state=72)),
                ('xgb', XGBRegressor(**xg_params)),
                ('lgbm', LGBMRegressor(**lgbm_params))
        ]

        meta_model = SVR(kernel='linear')

        model = StackingRegressor(estimators=base_model, final_estimator=meta_model, cv=7)

        # Latih model
        model.fit(X, y)
        train_pred = model.predict(X)
        r2 = r2_score(y, train_pred)
        mae = mean_absolute_error(y, train_pred)
        rmse = np.sqrt(mean_squared_error(y, train_pred))
        mape = round(mean_absolute_percentage_error(y, train_pred) *100, 2)

        df_pred = pd.DataFrame(train_pred, columns=['model_pred'])
        df_pred['actual'] = y
        df_pred.to_csv(self.model_config.prediction_path, index=False, header=True)

        # save model
        save_model(self.model_config.model_path, model)
        write_metrics(self.model_config.evaluation_path, r2, mae, rmse, mape)

        return r2, mae, rmse, mape
    





    