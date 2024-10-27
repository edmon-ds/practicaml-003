from src.components.data_ingestion import DataIngestion

train_df , test_df  =  DataIngestion().inittiate_data_ingestion()
print(train_df.head())