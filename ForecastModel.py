import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from gluonts.dataset.common import ListDataset
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from sklearn.metrics import mean_absolute_error
import numpy as np


class ForecastModel:
    def __init__(self,datafile=None,no_of_weeks = 12):
        self.__datafile = datafile
        self.__no_of_weeks = no_of_weeks
        self.__accuracy = {}
        if(self.__datafile and self.__no_of_weeks):
            self.__prepare_datasets()
        

    def set_DataFile(self,datafile):
        self.__datafile = datafile
        if(self.__datafile and self.__no_of_weeks):
            self.__prepare_datasets()

    def set_prediction_week(self,weeks):
        self.__no_of_weeks = weeks
        if(self.__datafile and self.__no_of_weeks):
            self.__prepare_datasets()

        
    def train_model(self):
        self.__train_SARIMA_model()
        self.__train_DeepAR_model()
        
    def __prepare_datasets(self):
        __data = pd.read_csv(self.__datafile, parse_dates=['week'])
        __data.set_index('week', inplace=True)
        self.__weekly_data = __data.groupby('week')['units_sold'].sum()
        # Split data into train and test sets
        self.__train_data = self.__weekly_data[:-self.__no_of_weeks]
        self.__test_data = self.__weekly_data[-self.__no_of_weeks:]
        # Prepare and fit DeepAR Model
        self.__train_ds = ListDataset(
            [{"start": self.__weekly_data.index[0], "target": self.__train_data.values}],
            freq="W"
        )
        self.__model_sarima_fit = None
        self.__deepar_predictor = None


    def __train_SARIMA_model(self):
        # Fit SARIMA Model
        model_sarima = SARIMAX(self.__train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
        self.__model_sarima_fit = model_sarima.fit(disp=False)
    
    def __train_DeepAR_model(self):
        

        deepar_estimator = DeepAREstimator(
            freq="W",
            prediction_length=self.__no_of_weeks,
            trainer=Trainer(ctx="cpu", epochs=25, learning_rate=1e-3)
        )

        self.__deepar_predictor = deepar_estimator.train(self.__train_ds)

    def __predict_SARIMA_forecast(self):
        sarima_forecast = self.__model_sarima_fit.forecast(steps=self.__no_of_weeks)
        return sarima_forecast
    
    def __predict_DeepAR_forecast(self):
        deep_ar_forecast_it = self.__deepar_predictor.predict(self.__train_ds)
        deep_ar_forecast = next(deep_ar_forecast_it)
        return deep_ar_forecast
    def Predict(self):
        f1 = self.__predict_SARIMA_forecast()
        f2 = self.__predict_DeepAR_forecast()
        combined_forecast = (f1 + f2.mean) / 2
        self.__computeAccuacy(f1,f2,combined_forecast)
        return f1,f2,combined_forecast
        
    def GeneratePredictionGraph(self,f1,f2,f3,filepath):
        output_file = filepath
        # Plot results
        plt.figure(figsize=(12, 6))
        #plt.plot(self.__weekly_data.index, self.__weekly_data, label="Actual Units Sold", color="blue")
        #plt.plot(self.__weekly_data.index[-self.__no_of_weeks:], f1, label="SARIMA Forecast", color="orange", linestyle="--", marker="o")
        #plt.plot(self.__weekly_data.index[-self.__no_of_weeks:], f2.mean, label="DeepAR Forecast", color="green", linestyle="--", marker="o")
        plt.plot(self.__weekly_data.index[-self.__no_of_weeks:], f3, label="Final Forecast", color="red", linestyle="--", marker="o")
        # Add data point labels for forecasts
        # for x, y in zip(self.__weekly_data.index[-self.__no_of_weeks:], f1):
        #     plt.text(x, y, f'{y:.0f}\n{x.strftime("%Y-%m-%d")}', color="red", ha="right", va="bottom", fontsize=8)
        # for x, y in zip(self.__weekly_data.index[-self.__no_of_weeks:], f2.mean):
        #     plt.text(x, y, f'{y:.0f}\n{x.strftime("%Y-%m-%d")}', color="red", ha="right", va="bottom", fontsize=8)
        for x, y in zip(self.__weekly_data.index[-self.__no_of_weeks:], f3):
            plt.text(x, y, f'{y:.0f}\n{x.strftime("%Y-%m-%d")}', color="red", ha="right", va="bottom", fontsize=8)

        # Add labels and title
        plt.title("Combined Forecast of Units Sold")
        plt.xlabel("Time")
        plt.ylabel("Units Sold")
        plt.legend()
        plt.savefig(output_file, dpi=300)  # Adjust dpi as needed
        return output_file
        #plt.show()
    def __computeAccuacy(self,f1,f2,f3):
        # Calculate accuracy metrics for each model
        # SARIMA
        
        
        test_data_values = self.__test_data.values
        sarima_forecast_values = f1.values
        deep_ar_forecast_values = f2.mean
        combined_forecast = (sarima_forecast_values + deep_ar_forecast_values) / 2
        # Calculate accuracy metrics for each model
        # SARIMA
        mae_sarima = mean_absolute_error(test_data_values, sarima_forecast_values)
        # mse_sarima = mean_squared_error(test_data_values, sarima_forecast_values)
        mape_sarima = np.mean(np.abs((test_data_values - sarima_forecast_values) / test_data_values)) * 100

        # DeepAR
        mae_deepar = mean_absolute_error(test_data_values, deep_ar_forecast_values)
        # mse_deepar = mean_squared_error(test_data_values, deep_ar_forecast_values)
        mape_deepar = np.mean(np.abs((test_data_values - deep_ar_forecast_values) / test_data_values)) * 100

        # Calculate MAE, MSE, and MAPE for Combined model
        mae_combined = mean_absolute_error(test_data_values, combined_forecast)
        # mse_combined = mean_squared_error(test_data_values, combined_forecast)
        mape_combined = np.mean(np.abs((test_data_values - combined_forecast) / test_data_values)) * 100
        data = {
            "sarima":{
                "accuracy":100-mape_sarima
            },
            "deepar":{
                "accuracy":100-mape_deepar
            },
            "combine":{
                "accuracy":100-mape_combined
            }
        }
        print(data)
        self.__accuracy  = data

    def getAccuracy(self):
        return self.__accuracy
    
    def getPredictionDatawithDates(self,f1,f2,f3):
        dates = [str(x.date()) for x in self.__weekly_data.index[-self.__no_of_weeks:]]
        return {
            "sarima":dict(zip(dates,f1.to_list())),
            "deepar":dict(zip(dates,f2.mean.tolist())),
            "combine":dict(zip(dates,f3.tolist()))
        }
    






class ForecastModelSKU:
    def __init__(self,datafile=None,no_of_weeks = 12,sku_id = None,store_id = None):
        self.__datafile = datafile
        self.__no_of_weeks = no_of_weeks
        self.__sku_id = sku_id
        self.__store_id = store_id
        self.__accuracy = {}
        

    def set_DataFile(self,datafile):
        self.__datafile = datafile
        

    def set_sku_id(self,id):
        self.__sku_id = id

    def set_store_id(self,id):
        self.__store_id = id

    
    def set_prediction_week(self,weeks):
        self.__no_of_weeks = weeks

    def prepar_dataset(self):
        if(self.__no_of_weeks and self.__datafile and self.__sku_id and self.__store_id):
            self.__prepare_datasets()
        
    def train_model(self):
        self.__train_SARIMA_model()
        self.__train_DeepAR_model()
        
    def __prepare_datasets(self):
        __data = pd.read_csv(self.__datafile, parse_dates=['week'])
        __data = __data[(__data['sku_id'] == self.__sku_id) & (__data['store_id'] == self.__store_id)]
        __data.set_index('week', inplace=True)
        self.__weekly_data = __data.groupby('week')['units_sold'].sum()
        # Split data into train and test sets
        self.__train_data = self.__weekly_data[:-self.__no_of_weeks]
        self.__test_data = self.__weekly_data[-self.__no_of_weeks:]
        # Prepare and fit DeepAR Model
        self.__train_ds = ListDataset(
            [{"start": self.__weekly_data.index[0], "target": self.__train_data.values}],
            freq="W"
        )
        self.__model_sarima_fit = None
        self.__deepar_predictor = None


    def __train_SARIMA_model(self):
        # Fit SARIMA Model
        model_sarima = SARIMAX(self.__train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
        self.__model_sarima_fit = model_sarima.fit(disp=False)
    
    def __train_DeepAR_model(self):
        

        deepar_estimator = DeepAREstimator(
            freq="W",
            prediction_length=self.__no_of_weeks,
            trainer=Trainer(ctx="cpu", epochs=25, learning_rate=1e-3)
        )

        self.__deepar_predictor = deepar_estimator.train(self.__train_ds)

    def __predict_SARIMA_forecast(self):
        sarima_forecast = self.__model_sarima_fit.forecast(steps=self.__no_of_weeks)
        return sarima_forecast
    
    def __predict_DeepAR_forecast(self):
        deep_ar_forecast_it = self.__deepar_predictor.predict(self.__train_ds)
        deep_ar_forecast = next(deep_ar_forecast_it)
        return deep_ar_forecast
    def Predict(self):
        f1 = self.__predict_SARIMA_forecast()
        f2 = self.__predict_DeepAR_forecast()
        combined_forecast = (f1 + f2.mean) / 2
        self.__computeAccuacy(f1,f2,combined_forecast)
        return f1,f2,combined_forecast
        
    def GeneratePredictionGraph(self,f1,f2,f3,filepath):
        output_file = filepath
        # Plot results
        plt.figure(figsize=(12, 6))
        #plt.plot(self.__weekly_data.index, self.__weekly_data, label="Actual Units Sold", color="blue")
        #plt.plot(self.__weekly_data.index[-self.__no_of_weeks:], f1, label="SARIMA Forecast", color="orange", linestyle="--", marker="o")
        #plt.plot(self.__weekly_data.index[-self.__no_of_weeks:], f2.mean, label="DeepAR Forecast", color="green", linestyle="--", marker="o")
        plt.plot(self.__weekly_data.index[-self.__no_of_weeks:], f3, label="Final Forecast", color="red", linestyle="--", marker="o")
        # Add data point labels for forecasts
        # for x, y in zip(self.__weekly_data.index[-self.__no_of_weeks:], f1):
        #     plt.text(x, y, f'{y:.0f}\n{x.strftime("%Y-%m-%d")}', color="red", ha="right", va="bottom", fontsize=8)
        # for x, y in zip(self.__weekly_data.index[-self.__no_of_weeks:], f2.mean):
        #     plt.text(x, y, f'{y:.0f}\n{x.strftime("%Y-%m-%d")}', color="red", ha="right", va="bottom", fontsize=8)
        for x, y in zip(self.__weekly_data.index[-self.__no_of_weeks:], f3):
            plt.text(x, y, f'{y:.0f}\n{x.strftime("%Y-%m-%d")}', color="red", ha="right", va="bottom", fontsize=8)

        # Add labels and title
        plt.title(f"Combined Forecast of Units Sold fo SKU id ${self.__sku_id} and Store id ${self.__store_id}")
        plt.xlabel("Time")
        plt.ylabel("Units Sold")
        plt.legend()
        plt.savefig(output_file, dpi=300)  # Adjust dpi as needed
        return output_file
        #plt.show()
    def __computeAccuacy(self,f1,f2,f3):
        # Calculate accuracy metrics for each model
        # SARIMA
        
        
        test_data_values = self.__test_data.values
        sarima_forecast_values = f1.values
        deep_ar_forecast_values = f2.mean
        combined_forecast = (sarima_forecast_values + deep_ar_forecast_values) / 2
        # Calculate accuracy metrics for each model
        # SARIMA
        mae_sarima = mean_absolute_error(test_data_values, sarima_forecast_values)
        # mse_sarima = mean_squared_error(test_data_values, sarima_forecast_values)
        mape_sarima = np.mean(np.abs((test_data_values - sarima_forecast_values) / test_data_values)) * 100

        # DeepAR
        mae_deepar = mean_absolute_error(test_data_values, deep_ar_forecast_values)
        # mse_deepar = mean_squared_error(test_data_values, deep_ar_forecast_values)
        mape_deepar = np.mean(np.abs((test_data_values - deep_ar_forecast_values) / test_data_values)) * 100

        # Calculate MAE, MSE, and MAPE for Combined model
        mae_combined = mean_absolute_error(test_data_values, combined_forecast)
        # mse_combined = mean_squared_error(test_data_values, combined_forecast)
        mape_combined = np.mean(np.abs((test_data_values - combined_forecast) / test_data_values)) * 100
        data = {
            "sarima":{
                "accuracy":100-mape_sarima
            },
            "deepar":{
                "accuracy":100-mape_deepar
            },
            "combine":{
                "accuracy":100-mape_combined
            }
        }
        print(data)
        self.__accuracy  = data

    def getAccuracy(self):
        return self.__accuracy
    
    def getPredictionDatawithDates(self,f1,f2,f3):
        dates = [str(x.date()) for x in self.__weekly_data.index[-self.__no_of_weeks:]]
        return {
            "sarima":dict(zip(dates,f1.to_list())),
            "deepar":dict(zip(dates,f2.mean.tolist())),
            "combine":dict(zip(dates,f3.tolist()))
        }
    def getSkuId(self):
        return self.__sku_id
    
    def getStoreId(self):
        return self.__store_id