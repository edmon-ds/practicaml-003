from flask import Flask , request , render_template
from src.exception import CustomException
from src.pipelines.predict_pipeline import * 

app  = Flask(__name__)

@app.route("/")
def index():
    print("se entro al index")
    return render_template("index.html")

@app.route("/predict" , methods = [ "GET", "POST"])
def predict_datapoint():
    if request.method =="GET":
        #print("in get")
        return render_template("predict.html")
    else:
        try:
           # print("in post")
            user_data = CustomData(
                battery_power = request.form.get("battery_power"),
                blue = request.form.get("blue"),
                clock_speed = request.form.get("clock_speed"),
                 dual_sim = request.form.get("dual_sim"),
                 fc = request.form.get("fc"),
                 four_g = request.form.get("four_g"),
                 int_memory = request.form.get("int_memory"),
                 m_dep = request.form.get("m_dep"),
                 mobile_wt = request.form.get("mobile_wt"),
                 n_cores = request.form.get("n_cores"),
                 pc = request.form.get("pc"),
                 px_height = request.form.get("px_height"),
                 px_width = request.form.get("px_width"),
                 ram = request.form.get("ram"),
                 sc_h = request.form.get("sc_h"),
                 sc_w = request.form.get("sc_w"),
                 talk_time = request.form.get("talk_time"),
                 three_g = request.form.get("three_g"),
                 touch_screen = request.form.get("touch_screen"),
                 wifi = request.form.get("wifi")
            )
            user_data_df = user_data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            preds = predict_pipeline.predict(user_data_df)
            
            return render_template("predict.html" , results = preds[0])

        except Exception as e:
            raise CustomException(e , sys)

if __name__ =="__main__":
    app.run(host = "0.0.0.0" , port = 8080, debug = True)