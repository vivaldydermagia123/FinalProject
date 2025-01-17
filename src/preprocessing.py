
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
import ast

# Column Definitions
ohe_columns = ['FUNCTION']
transform_columns = ['CONDUCT LIMIT']
label_columns = ['BACTERIA_STATUS']
num_columns = ['DAY_RUN', 'RR', 'AVG_T_RETURN', 'AVG_FRC', 'AVG_CHLORIDE']
material_columns = ['MATERIAL']
target_columns = ['NAOCL']

class CategoriConduct(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.apply(lambda col: col.map(lambda x: 1 if x <= 700 else 2 if x <= 1200 else 3))
    
class LabelEncode(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mapping = {'Acceptable': 2, 'Not Acceptable': 1, 'low': 1, 'medium': 2, 'high': 3}
        return self
    
    def transform(self, X):
        # Replace and then explicitly infer object types
        X_transformed = X.replace(self.mapping)
        X_transformed = X_transformed.infer_objects(copy=False)
        return X_transformed

class MaterialEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name
        self.mlb = MultiLabelBinarizer()
        self.classes_ = None

    def fit(self, X, y=None):
        X = self._preprocess_column(X)  # Memastikan nilai dalam bentuk list
        # Fit MultiLabelBinarizer pada kolom yang sudah berbentuk list
        self.mlb.fit(X[self.column_name])
        self.classes_ = self.mlb.classes_
        return self

    def transform(self, X):
        X = self._preprocess_column(X)  # Memastikan nilai dalam bentuk list
        # Transformasi data menjadi binary (0 atau 1)
        encoded_data = self.mlb.transform(X[self.column_name])
        encoded_df = pd.DataFrame(encoded_data, columns=self.classes_, index=X.index)
        return encoded_df

    def _preprocess_column(self, X):
        # Salin dataframe agar tidak memodifikasi data asli
        X = X.copy()
        
        # Ubah nilai string yang berupa daftar bahan menjadi list
        X[self.column_name] = X[self.column_name].apply(
            lambda x: [item.strip() for item in x.split(',')] if isinstance(x, str) else []  # Pisahkan berdasarkan koma
        )
        
        return X
    

