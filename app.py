from flask import Flask,request,render_template
import numpy as mp
import pandas as pd
import os
from src.pipeline.predict_pipeline import Customdata,PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline

from sklearn.preprocessing import  StandardScaler

application = Flask(__name__)

app=application

## Route for Home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data= Customdata(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    

@app.route('/traindata',methods=['GET','POST'])
def train_datapoint():
    if request.method == 'POST':
        train_pipeline = TrainPipeline()
        
        model_report={}
        best_r2_score_over_test_data,best_model_name,best_model_score,model_report = train_pipeline.train()

        return render_template(
            'train.html',
            best_model=best_model_name,
            best_model_score=best_r2_score_over_test_data,
            model_report=model_report
            )
    else:
        file_path = os.path.join('artifacts','model_report.txt')
        if os.path.exists(file_path):
            with open(file_path,'r') as file_obj:
                contents = file_obj.readlines()
                title = contents[0]
                best_model_name = contents[-2]
                best_model_score = contents[-1]
        else:
            contents = "The exist's no model report for previously trained model. Please re-train the model"
        return render_template('prevtrain.html',title=title,contents=contents[1:-2],best_model=best_model_name,best_model_score=best_model_score)

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)       