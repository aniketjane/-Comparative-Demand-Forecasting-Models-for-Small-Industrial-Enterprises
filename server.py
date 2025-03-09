from flask import Flask,request,render_template,url_for
from utils.ForecastModel import ForecastModel,ForecastModelSKU
import os
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'input')
OUTPUT_FOLDER = os.path.join(app.root_path, 'static', 'output')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # Ensure the folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
forecast_model = ForecastModel()
forecast_model_sku  = ForecastModelSKU()
prediction_data = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict",methods = ["POST"])
def predict():
    global prediction_data
    file  = request.files.get("file")
    print(request.form) 
    weeks = request.form.get("noofweeks")
    print(file)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "data.csv")
    file.save(file_path)
    forecast_model.set_DataFile(datafile=file_path)
    if weeks.isdigit():
        forecast_model.set_prediction_week(int(weeks))
    forecast_model.train_model()
    output_file_path = os.path.join(app.config['OUTPUT_FOLDER'], "final.png")
    f1,f2,f3 = forecast_model.Predict()
    prediction_data = forecast_model.getPredictionDatawithDates(f1,f2,f3)
    forecast_model.GeneratePredictionGraph(f1,f2,f3,output_file_path)
    return url_for("ShowResult")

@app.route("/predict_sku",methods = ["POST"])
def predict_for_sku():
    global prediction_data
    file  = request.files.get("file")
    print(request.form) 
    weeks = request.form.get("noofweeks")
    sku_id = request.form.get("sku_id")
    store_id = request.form.get("store_id")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "data.csv")
    file.save(file_path)
    forecast_model_sku.set_DataFile(datafile=file_path)
    
    if weeks.isdigit():
        forecast_model_sku.set_prediction_week(int(weeks))
    if sku_id.isdigit():
        forecast_model_sku.set_sku_id(int(sku_id))
    if store_id.isdigit():
        forecast_model_sku.set_store_id(int(store_id))

    forecast_model_sku.prepar_dataset()
    forecast_model_sku.train_model()
    output_file_path = os.path.join(app.config['OUTPUT_FOLDER'], "final_sku.png")
    f1,f2,f3 = forecast_model_sku.Predict()
    prediction_data = forecast_model_sku.getPredictionDatawithDates(f1,f2,f3)
    forecast_model_sku.GeneratePredictionGraph(f1,f2,f3,output_file_path)
    return url_for("ShowResult")



@app.route("/result")
def ShowResult():
    accuaracy_data = forecast_model_sku.getAccuracy()
    print(accuaracy_data)
    print(prediction_data)
    return render_template("result.html",accuracy_data = accuaracy_data,prediction_data=prediction_data,sku_id = forecast_model_sku.getSkuId(),store_id= forecast_model_sku.getStoreId())

app.run(debug=True)