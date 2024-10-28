import sys
import os
from src.logger import logging 
from src.exception import CustomException
from src.utils import * 

from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer 

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator , TransformerMixin

import pandas as pd
import numpy as np

@dataclass
class DataTransformationConfig():
    preprocessor_path = os.path.join( "artifacts", "preprocessor.pkl")

#tiene que ser llamado en un pipeline
class RestoreNamesTransformer(BaseEstimator , TransformerMixin):
    def __init__(self , features_names):
        self.features_names = features_names
    def fit(self , X, y = None):
        return self
    def transform(self , X ):
        return pd.DataFrame( X , columns=self.features_names)

##class for feature engineering
class FeatureEngineeringTransformer(BaseEstimator , TransformerMixin):
    def __init__(self):
        pass
    def fit(self , X , y = None ):
        pass
    def transform(self , X):
        '''se crean las feature nuevas'''
        X["px_area"] = X["px_height"] * X["px_width"] 
        X["sc_area"] =  X["sc_h"] * X["sc_w"] 
        return X
    def fit_transform(self, X, y = None):
        '''se crean las feature nuevas'''
        X["px_area"] = X["px_height"] * X["px_width"] 
        X["sc_area"] =  X["sc_h"] * X["sc_w"] 
        return X


class DataTransformation():
    def __init__(self):
        self.dataconfig = DataTransformationConfig()
        self.numerical_columns = ['battery_power', 'int_memory', 'mobile_wt', 'px_height', 'px_width', 'ram']
        self.categorical_columns = ['blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'm_dep', 'n_cores', 'pc', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi']
        self.features_columns = self.numerical_columns + self.categorical_columns
        self.label = "price_range"

    def get_preprocessor(self):
        
        ##imputer
       
        cleaning_num_steps = Pipeline(
            [
                ("num_imputer" , SimpleImputer(strategy="mean"))
            ]
                                    )


        cleaning_cat_steps = Pipeline(
            [
                ("cat_imputer" , SimpleImputer(strategy="most_frequent"))
            ]
                     )
        
   
        cleaning_transformer = ColumnTransformer(
            [
                 ("imputer_num_steps" , cleaning_num_steps , self.numerical_columns),
                 ("imputer_cat_steps" ,  cleaning_cat_steps, self.categorical_columns) 
            ]
        )

        ##restaring feature name
        restore_names_transformer = RestoreNamesTransformer(self.features_columns)

        ##feature engineering
        feature_engineering_transformer = FeatureEngineeringTransformer()
        
        ##preprocessing

        preprocessing_num_steps = Pipeline(
            steps = [
                 ("num_scaler" , StandardScaler()) 
            ]         )
        
        preprocessing_cat_steps = Pipeline(
            steps = [
             ( "scaler", StandardScaler() )
            ]
        )
        preprocesing_transformer = ColumnTransformer(
            [
                ("preprocessing_num_steps",preprocessing_num_steps  , self.numerical_columns) ,
                ("preprocessing_cat_steps" ,preprocessing_cat_steps ,self.categorical_columns)
            ] 
        )

        preprocessor = Pipeline(
            steps = [
                ("cleaning" , cleaning_transformer) ,
                ("restore_names" , restore_names_transformer ), 
                ("feature_engineering" , feature_engineering_transformer) , 
                ("preprocessing" , preprocesing_transformer )
                    ] 
        )
        return preprocessor

    def initiate_data_transformation(self ,train_df:pd.DataFrame , test_df:pd.DataFrame ):
        logging.info("enter to initate_da_transformation")
        try:
            preprocessor = self.get_preprocessor()
            logging.info("separating the labels from features")
            
            train_input_raw = train_df.drop(columns = [self.label])
            train_label =     train_df[self.label]
            
            test_input_raw = test_df.drop(columns = [self.label] )
            test_label =     test_df[self.label]

            logging.info("applyiend preprocessing to the dataset")

            train_input_array = preprocessor.fit_transform(train_input_raw)
            test_input_array = preprocessor.transform(test_input_raw)

            train_array =  np.hstack(( train_input_array ,train_label.values.reshape(-1,1) ))
            test_array = np.hstack((test_input_array , test_label.values.reshape(-1,1)))

            logging.info("saving preprocessor")

            save_object(file_path= self.dataconfig.preprocessor_path , obj = preprocessor)
            
            return ( train_array , test_array )

        except Exception as e:
            raise CustomException(e , sys)