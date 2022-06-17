import os
import sys
import numpy as np
from argparse import ArgumentParser
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy import stats
import seaborn as sns
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

    def get_model(self):
        return MLPRegressor(verbose=True, hidden_layer_sizes=(128,100,64,32), 
                            learning_rate='invscaling', activation='relu', 
                            early_stopping=True)

