import os 
import sys 
import pandas as pd 
import numpy as np 

# Tambahkan root folder Spam-Message-Detection ke sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_file

class PredictionPipeline():
    def __init__(self):
        pass

    def prediction(self, features):
        # load model and preprocessor
        model_path = os.path.join("artifacts", "model.pkl")
        preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

        model = load_file(model_path)
        preprocessor = load_file(preprocessor_path)

        # ML pipeline
        data_preprocessing = preprocessor.transform(features)
        pred = model.predict(data_preprocessing)

        return pred
    
# Class for get custom data
class CustomData():
    def __init__(self,  DAY_RUN : float,
                RR : float, AVG_T_RETURN : float,
                AVG_FRC : float,  AVG_CHLORIDE: float, 
                CONDUCT_LIMIT : float, BACTERIA_STATUS : str,
                FUNCTION : str, MATERIAL : str):
        
        self.DAY_RUN = DAY_RUN
        self.RR = RR
        self.AVG_T_RETURN = AVG_T_RETURN
        self.AVG_FRC = AVG_FRC
        self.AVG_CHLORIDE = AVG_CHLORIDE
        self.CONDUCT_LIMIT = CONDUCT_LIMIT
        self.BACTERIA_STATUS = BACTERIA_STATUS
        self.FUNCTION  = FUNCTION 
        self.MATERIAL = MATERIAL 

    def get_data_input(self):
        data_input_dict = {
            "DAY_RUN": [self.DAY_RUN],
            "RR": [self.RR],
            "AVG_T_RETURN": [self.AVG_T_RETURN],
            "AVG_FRC": [self.AVG_FRC],
            "AVG_CHLORIDE": [self.AVG_CHLORIDE],
            "CONDUCT LIMIT": [self.CONDUCT_LIMIT],
            "BACTERIA_STATUS": [self.BACTERIA_STATUS],
            "FUNCTION": [self.FUNCTION],
            "MATERIAL": [', '.join(self.MATERIAL)] 
        }
        return pd.DataFrame(data_input_dict)


     
 
        
    
