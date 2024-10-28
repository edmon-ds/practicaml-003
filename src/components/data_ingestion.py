import os
import sys
from src.logger import logging 
from src.exception import CustomException
from dataclasses import dataclass

from sqlalchemy import create_engine

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig():
    raw_data_path = os.path.join("artifacts" ,"dataset.csv" )

    ##--------------------Database credentials
    driver:str = "ODBC+Driver+17+for+SQL+Server"
    server_name:str = "localhost"
    database:str = "BDdatasets"
    UID:str = "sa"
    PWD:str = "0440"

    connection_string:str = f"mssql+pyodbc://{UID}:{PWD}@{server_name}/{database}?driver={driver}"

class DataIngestion():
    def __init__(self):
        self.dataconfig = DataIngestionConfig()
   
    def inittiate_data_ingestion(self):
        logging.info("enter to the data ingestion  method")
        try:
            engine = create_engine( self.dataconfig.connection_string )
            query = "SELECT * FROM MobilePrice"
           
            logging.info("reading database as dataframe")

            df = pd.read_sql_query( query, engine )

            logging.info("saving top 5 record of dataset")
            #save for look the features after
            df.head().to_csv(self.dataconfig.raw_data_path , header = True , index= False) 

            train_df , test_df = train_test_split(df , test_size = 0.2 ,random_state  = 42 )
            
            return ( train_df , test_df )

        except Exception as e:
            raise CustomException(e , sys) 
            