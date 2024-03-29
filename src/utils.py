import os
import sys
import dill

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path =  os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        logging.info("Model Evaluation Inititated")
        logging.info("Performing Hyper Parameter tuning")
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            logging.info('Evaluating {0} with parameters: {1}'.format(list(models.keys())[i],para))
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
    
        logging.info("Model evaluation Completed")
        return report
    
    except Exception as e:
        raise CustomException(e,sys)    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
    
def save_model_report(file_path,best_model,best_model_score,model_report):
    try:
        logging.info('Preparing Model Report') 
        with open(file_path,'w') as file_obj:
            file_obj.write('Training Report of various models\n')
            for key, value in model_report.items():
                file_obj.write('{0}: {1}\n'.format(key,value)) 
            file_obj.write('Best Model: {0}\n'.format(best_model))
            file_obj.write('Best Model Score: {0}\n'.format(best_model_score))
        logging.info('Model Report Generated')
        return None
    
    except Exception as e:
        raise CustomException(e,sys)
    
