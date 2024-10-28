import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np


from src.utils import save_object
#data models
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

#models metrics

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

class ModelTrainerConfig():
    model_path = os.path.join("artifacts" , "model.pkl")

class ModelTrainer():
    def __init__(self):

        self.dataconfig = ModelTrainerConfig()
        
        self.models = {
                        "LogisticRegression": LogisticRegression(), 
                        "AdaBoostClassifier": AdaBoostClassifier() , 
                        "XGBClassifier":XGBClassifier() , 
                    }
     
        self.models_params = {
                        "LogisticRegression": {
                                                'C': [ 0.1, 1,],
                                                    'solver': ['liblinear', 'saga'],  # Agrega solver para compatibilidad
                                                'penalty': ['l1', 'l2', 'none'],
                                                'max_iter': [ 200, 500],
                                                
                        },
                        "AdaBoostClassifier": {
                                                'n_estimators': [50, 100,],
                                                'learning_rate': [0.1, 0.5,],
                        },
                        "XGBClassifier": {
                                                'n_estimators': [50, 100],
                                                'learning_rate': [ 0.1, 0.2],
                                                'max_depth': [ 5, 7],
                        }
                        
                    }
        
        self.threshold = 0.9
        self.model_report = None
        
    def evaluate_model(self , X_train , y_train , X_test , y_test):
        try:
            report = []

            for model_name , model in self.models.items():
                params = self.models_params[model_name]
                gs = GridSearchCV(model , params , cv = 3 )
                gs.fit(X_train , y_train)

                model.set_params(** gs.best_params_)

                model.fit(X_train,  y_train)

                y_test_pred = model.predict(X_test)

                report.append(
                    {
                        "model_name": model_name ,
                        "accuracy_score":accuracy_score(y_test , y_test_pred),
                        "recall_score": recall_score(y_test , y_test_pred, average="macro"),
                        "precision_score":precision_score(y_test , y_test_pred, average="macro") , 
                        "f1_score":f1_score(y_test , y_test_pred, average="macro")
                    }
                )
            
            report = pd.DataFrame(report)
            
            report["average_score"] = ( report["accuracy_score"] + 
                                                report["recall_score"] + 
                                                report["precision_score"] + 
                                                report["f1_score"] ) / 4
            
            self.model_report = report
            return report

        except Exception as e:
            raise CustomException(e , sys)

    def show_report(self):
        print(self.model_report.sort_values(by = "average_score" , ascending=False))

    def initiate_model_training(self ,train_array , test_array ):
        try:
            X_train , y_train , X_test , y_test = ( train_array[: , :-1] ,  train_array[: , -1] , 
                                                     test_array[:  , :-1 ] ,test_array[: , -1] )
            
            model_report_df:pd.DataFrame = self.evaluate_model( X_train , y_train , X_test , y_test )
    
            best_model_row = model_report_df.loc[model_report_df["average_score"].idxmax()]
            best_model_name = best_model_row["model_name"]
            best_model_score = best_model_row["average_score"]
            best_model = self.models[best_model_name] 

            if best_model_score < self.threshold:
                raise CustomException("the models perfomance was not achieve")
            
            logging.info("best model found")
            
            save_object(self.dataconfig.model_path , obj = best_model)

            return (best_model_name , best_model_score , best_model)

        except Exception as e:
            raise CustomException( e  , sys ) 