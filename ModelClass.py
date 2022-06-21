import os
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import HuberRegressor
from sklearn.feature_selection import SequentialFeatureSelector
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
        # columns selected by feature selection
        self.prepro_cols = ['MNase_GB', 'H3K27me3_GB', 'sRNA', 'GC']
        self.response = ["mRNA"]


    def get_model(self):
        return RandomForestRegressor()
        #return MLPRegressor(verbose=True, hidden_layer_sizes=(128,64,32), 
        #                    learning_rate='invscaling', activation='relu', 
        #                    early_stopping=True)
        #return MLPRegressor(verbose=True, hidden_layer_sizes=(128,100,64,48), 
        #                    learning_rate='invscaling', activation='relu', 
        #                    early_stopping=True)

    def preprocess_data(self, data, columns):
        if "GC" in columns:
            cols_to_transf = [i for i in columns if i != "GC"]
            #columns.remove("GC")
            gc = data.GC.to_numpy()/100
        data_transformed = self.log_transform(data, cols_to_transf)
        scaler = preprocessing.RobustScaler().fit(data_transformed)
        data_preprocessed = scaler.transform(data_transformed)
        data_preprocessed = pd.DataFrame(data_preprocessed, columns=cols_to_transf)
        if "GC" in columns:
            data_preprocessed["GC"] = gc
        return data_preprocessed.to_numpy(), scaler

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

    def select_best_predictors(self, X, y, n_features=5, direction="forward", cpu=4):
        sfs_selector = SequentialFeatureSelector(
            estimator=self.model, n_features_to_select=n_features, direction=direction, n_jobs=cpu)
        sfs_selector.fit(X, y)
        feature_columns = sfs_selector.get_support()
        return X.columns[feature_columns]

    def select_response(self, columns):
        response = self.data.loc[:, columns]
        response = response.to_numpy()
        self.response = response

    def model_predict(self, X):
        return self.model.predict(X)

    def model_train(self, X, y):
        self.model = self.model.fit(X, y)
