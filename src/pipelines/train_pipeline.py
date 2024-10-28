from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import warnings
warnings.filterwarnings("ignore")

data_ingestion = DataIngestion()
data_transformation = DataTransformation()
model_trainer = ModelTrainer()

print("starting data ingestion")
train_df , test_df  =  data_ingestion.inittiate_data_ingestion()


print("preprocessing data")
train_array , test_array = data_transformation.initiate_data_transformation(train_df , test_df)


print("training models ...")
best_model_name , best_model_score , best_model = model_trainer.initiate_model_training(train_array , test_array)
print()
model_trainer.show_report()
print()
print()
print(f"el mejor modelos fue {best_model_name} con score de {best_model_score:.5f}")
