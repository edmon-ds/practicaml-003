import sys
import os
from src.logger import logging 
from src.exception import CustomException

from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer 

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

@dataclass
class DataTransformationConfig():
    pass

###

###

from sklearn.base import BaseEstimator , TransformerMixin

class FeatureEngineeringTransformer( BaseEstimator , TransformerMixin):
    def __init__(self):
        pass
    def fit(self , X , y = None ):
        pass
    def transform(self , X):
        pass
    def fit_transform(self, X, y = None):
        pass