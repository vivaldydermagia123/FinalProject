from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sys
import os

# Tambahkan root folder Spam-Message-Detection ke sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict_pipeline import PredictionPipeline, CustomData

application = Flask(__name__)
app = application

# Route for first page
@app.route("/")
def index():
    return render_template('index.html')

# Route Predict page
@app.route("/predict", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('predict_page.html')
    else:
        data = CustomData(
                DAY_RUN=float(request.form.get('DAY_RUN')),
                RR=float(request.form.get('RR')),
                AVG_T_RETURN=float(request.form.get('AVG_T_RETURN')),
                AVG_FRC=float(request.form.get('AVG_FRC')),
                AVG_CHLORIDE=float(request.form.get('AVG_CHLORIDE')),
                CONDUCT_LIMIT=int(request.form.get('CONDUCT_LIMIT')),
                BACTERIA_STATUS=request.form.get('BACTERIA_STATUS'),
                FUNCTION=request.form.get('FUNCTION'),
                MATERIAL=request.form.getlist('MATERIAL')  # Ambil sebagai list
            )


        df_pred = data.get_data_input()
        print(df_pred)
        print(f'Before prediction')

        predict_pipeline = PredictionPipeline()
        results = predict_pipeline.prediction(df_pred)
        results = np.round(results, 2)
        print(f'After prediction')

        return render_template('predict_page.html', results=results[0])
    

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))

