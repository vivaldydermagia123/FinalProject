import pickle
import os 

def save_model(file_name, model):
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)

def write_metrics(path, r2, mae, rmse, mape):
    with open(path, "w") as file:
        file.write(f'Evaluation Model : \n')
        file.write(f'R2_score : {r2}\n')
        file.write(f'MAE : {mae}\n')
        file.write(f'RMSE : {rmse}\n')  
        file.write(f'MAPE : {mape}\n')

def load_file(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)
