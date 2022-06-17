import os
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import pandas as pd


class Model():
    def __init__(self, data):
        self.data = os.path.abspath(data)
        if not os.path.exists(self.data):
            raise FileNotFoundError("File is missing...")
        self.model = self.get_model()
        self.data = pd.read_csv(self.data)
        self.predictors = None
        self.features = None
        self.prepro_cols = ["MNase_TSSm150", "H3K27me3_TSSm150", 
                            "H3K4me3_TSSm150", "H3K9ac_TSSm150", 
                            "Pol2_TSSm150", "MNase_TSSp300", 
                            "H3K27me3_TSSp300", "H3K4me3_TSSp300", 
                            "H3K9ac_TSSp300", "Pol2_TSSp300", 
                            "MNase_TTSm200", "H3K27me3_TTSm200", 
                            "H3K4me3_TTSm200", "H3K9ac_TTSm200", 
                            "Pol2_TTSm200", "MNase_GB", 
                            "H3K27me3_GB", "H3K4me3_GB", 
                            "H3K9ac_GB", "Pol2_GB"]
        self.response = ["mRNA"]


    def get_model(self):
        return MLPRegressor(verbose=True, hidden_layer_sizes=(128,100,64,48), 
                            learning_rate='invscaling', activation='relu', 
                            early_stopping=True)

    def preprocess_data(self, data, columns):
        data_transformed = self.log_transform(data, columns)
        scaler = preprocessing.RobustScaler().fit(data_transformed)
        return scaler.transform(data_transformed), scaler

    def log_transform(self, data, columns):
        log2_transform = lambda x: np.log2(x+1)
        transformer = preprocessing.FunctionTransformer(log2_transform)
        return transformer.transform(data[columns])

    def backtransform_data(self, values):
        return np.power(2, values)

    def select_predictors(self, columns):
        header = [x for x in columns if x in self.data.columns]
        features = self.data.loc[:, header]
        features = features.to_numpy()
        self.predictors = features

    def select_response(self, columns):
        response = self.data.loc[:, columns]
        response = response.to_numpy()
        self.response = response

    def model_predict(self, X):
        return self.model.predict(X)

    def model_train(self, X, y):
        self.model = self.model.fit(X, y)