import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


# Tambahkan root folder Spam-Message-Detection ke sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import PipelinePrepro
from src.modeling import ModelTrain

@dataclass
class DataingestionConfig():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(BASE_DIR, "artifacts")
    train_file_path = os.path.join(artifacts_dir, "train.csv")
    test_file_path = os.path.join(artifacts_dir, "test.csv")
    raw_file_path = os.path.join(artifacts_dir, "data.csv")

class Dataingestion():
    def __init__(self):
        self.dataconfig = DataingestionConfig()

    def load_data(self):
        df = pd.read_csv('notebooks\Chem Trend.csv')

        # craate folder artifacts
        os.makedirs(os.path.dirname(self.dataconfig.train_file_path), exist_ok=True)

        #split data
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=72)

        # save dataset
        df.to_csv(self.dataconfig.raw_file_path, index=False, header=True)
        train_data.to_csv(self.dataconfig.train_file_path, index=False, header=True)
        test_data.to_csv(self.dataconfig.test_file_path, index=False, header=True)

        return (
            self.dataconfig.train_file_path,
            self.dataconfig.test_file_path
        )
    
# Check code 
if __name__=="__main__":
    # Load data
    load_data = Dataingestion()
    train_data, test_data = load_data.load_data()

    # preprocessor pipeline
    prepro = PipelinePrepro()
    df_prepro = prepro.process(train_data)

    # train model
    modeling = ModelTrain()
    r2_train, mae_train, rmse_train, mape_train = modeling.modeling(df_prepro)
    print(f'r2_train : {r2_train}')
    print(f'mae_train : {mae_train}')
    print(f'rmse_train : {rmse_train}')
    print(f'mape_train : {mape_train}')