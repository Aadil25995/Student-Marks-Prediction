import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.components.data_ingestion import DataIngestion,DataTransformation,ModelTrainer

import numpy as np
import pandas as pd
from dataclasses import dataclass


class TrainPipeline:
    def __init__(self):
        pass
    def train(self):
        try:
            obj = DataIngestion()
            train_data,test_data = obj.initiate_data_ingestion()

            data_transformation = DataTransformation()
            train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_data,test_data)

            modeltrainer = ModelTrainer()
            return (modeltrainer.initiate_model_trainer(train_arr,test_arr))
            
        except Exception as e:
            raise CustomException(e,sys)