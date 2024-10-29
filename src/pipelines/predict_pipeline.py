import pandas as pd

from src.utils import * 
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.exception import CustomException

class CustomData:
    def __init__(self  ,battery_power,blue,clock_speed,
                 dual_sim,fc,four_g,int_memory,m_dep,mobile_wt,
                 n_cores,pc,px_height,px_width,ram,sc_h,sc_w,
                 talk_time,three_g,touch_screen,wifi
                 ):
        self.user_data = pd.DataFrame({
            "battery_power":[float(battery_power)],
            "blue":[float(blue)],
            "clock_speed":[float(clock_speed)],
            "dual_sim":[float(dual_sim)],
            "fc":[float(fc)],
            "four_g":[float(four_g)] , 
            "int_memory":[float(int_memory)],
            "m_dep":[float(m_dep)],
            "mobile_wt":[float(mobile_wt)],
            "n_cores":[float(n_cores)],
            "pc":[float(pc)],
            "px_height":[float(px_height)],
            "px_width":[float(px_width)],
            "ram":[float(ram)],
            "sc_h":[float(sc_h)],
            "sc_w":[float(sc_w)],
            "talk_time":[float(talk_time)], 
            "three_g":[float(three_g)],
            "touch_screen":[float(touch_screen)],
            "wifi":[float(wifi)]
        })

    def get_data_as_dataframe(self):
        return self.user_data

class PredictPipeline():
    def __init__(self):
        self.preprocessor = load_object(DataTransformationConfig.preprocessor_path)
        self.model = load_object(ModelTrainerConfig.model_path)
        
    def predict(self , user_data):
        try:
            data_transformed = self.preprocessor.transform(user_data)
            preds = self.model.predict(data_transformed)
            return preds
        except Exception as e:
            raise CustomException(e , sys)