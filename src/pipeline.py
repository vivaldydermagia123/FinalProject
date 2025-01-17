import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Tambahkan root folder Spam-Message-Detection ke sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import CategoriConduct, LabelEncode, MaterialEncoder
from src.utils import save_model

# Column Definitions
ohe_columns = ['FUNCTION']
transform_columns = ['CONDUCT LIMIT']
label_columns = ['BACTERIA_STATUS']
num_columns = ['DAY_RUN', 'RR', 'AVG_T_RETURN', 'AVG_FRC', 'AVG_CHLORIDE']
material_columns = ['MATERIAL']
target_columns = ['NAOCL']

# Output columns
output_columns = (
    num_columns +
    transform_columns +
    label_columns +
    ['FUNCTION'] +
    ['Cuprum', 'Galvanized', 'Stainless steel']
)

@dataclass
class PipelineConfig():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(BASE_DIR, "artifacts")
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
    df_preprocessing = os.path.join(artifacts_dir, "data_preprocessed.csv")

class PipelinePrepro():
    def __init__(self):
        self.pipeline_config = PipelineConfig()

    def split(self, df):

        X = df.drop(columns=target_columns)
        y = df[target_columns]

        return X, y
    
    def pipeline(self, X):
        # Pipeline
        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categori_conduct_pipeline = Pipeline(steps=[
            ('categorize', CategoriConduct())
        ])

        label_pipeline = Pipeline(steps=[
            ('label_encode', LabelEncode())
        ])

        material_pipeline = Pipeline(steps=[
            ('binarizer', MaterialEncoder(column_name='MATERIAL'))  
        ])

        ohe_pipeline = Pipeline(steps=[
            ('ohe', OneHotEncoder())
        ])

        # Combine Pipelines into ColumnTransformer
        preprocessor = ColumnTransformer(transformers=[
            ('numeric_pipeline', numeric_pipeline, num_columns),
            ('conduct_category', categori_conduct_pipeline, transform_columns),
            ('label_encode', label_pipeline, label_columns),
            ('ohe', ohe_pipeline, ohe_columns),
            ('material_binarizer', material_pipeline, material_columns)
        ], remainder='drop')

        X_process = preprocessor.fit_transform(X)
        
        # save preprocessor 
        save_model(self.pipeline_config.preprocessor_path, preprocessor)

        return X_process

    def process(self, df):

        data = pd.read_csv(df)

        # split input and target
        X, y = self.split(data)

        # pipeline
        X_processed = self.pipeline(X)

        df_preprocessing = pd.DataFrame(X_processed)
        df_preprocessing['NAOCL'] = y

        df_preprocessing.to_csv(self.pipeline_config.df_preprocessing, index=False, header=True)

        return self.pipeline_config.df_preprocessing





