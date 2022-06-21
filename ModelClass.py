import os
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import HuberRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from utils import one_hot_dna
import pandas as pd


class Model():
    def __init__(self, data):
        self.data = os.path.abspath(data)
        if not os.path.exists(self.data):
            raise FileNotFoundError("File is missing...")
        self.model = self.get_model()
        self.data = pd.read_csv(self.data)
        self.predictors = ["PromoterSeq"]
        self.features = None
        # columns selected by feature selection
        self.prepro_cols = ['MNase_GB', 'H3K27me3_GB', 'sRNA', 'GC', 'H3K9ac_TSSm150', 'PromoterSeq']
        self.response = ["mRNA"]
        self.alphabet = "ACGTN"
        self.used_columns = []


    def get_model(self):
        return RandomForestRegressor()
        #return MLPRegressor(verbose=True, hidden_layer_sizes=(100,64,32), 
        #                    learning_rate='invscaling', activation='relu', 
        #                    early_stopping=True)
        #return MLPRegressor(verbose=True, hidden_layer_sizes=(128,100,64,48), 
        #                    learning_rate='invscaling', activation='relu', 
        #                    early_stopping=True)

    def preprocess_data(self, data, columns):
        cols_to_transf = [i for i in columns if i not in ["GC", "PromoterSeq", "mRNA", "GeneID"]]
        data_preprocessed = pd.DataFrame()
        if len(cols_to_transf) != 0:
            data_transformed = self.log_transform(data, cols_to_transf)
            scaler = preprocessing.RobustScaler().fit(data_transformed)
            data_preprocessed = scaler.transform(data_transformed)
            data_preprocessed = pd.DataFrame(data_preprocessed, columns=cols_to_transf)
        if "GC" in columns:
            gc = data.GC.to_numpy()/100
            data_preprocessed["GC"] = gc
        if "PromoterSeq" in columns:
            one_hot = lambda x: one_hot_dna(x, self.alphabet).reshape(1, len(x)*len(self.alphabet))[0]
            data_preprocessed["PromoterSeq"] = data["PromoterSeq"].apply(one_hot)
        self.used_columns = data_preprocessed.columns
        return data_preprocessed.to_numpy()

    def log_transform(self, data, columns):
        log2_transform = lambda x: np.log2(x+1)
        transformer = preprocessing.FunctionTransformer(log2_transform)
        return transformer.transform(data[columns])

    def backtransform_data(self, values):
        return np.power(2, values)

    def select_predictors(self, columns):
        header = [x for x in columns if x in self.data.columns]
        return self.data.loc[:, header]

    def select_best_predictors(self, X, y, n_features=5, direction="forward", cpu=4):
        X_df = pd.DataFrame(X, columns=self.used_columns)
        sfs_selector = SequentialFeatureSelector(
            estimator=self.model, n_features_to_select=n_features, direction=direction, n_jobs=cpu)
        sfs_selector.fit(X_df, y)
        feature_columns = sfs_selector.get_support()
        print(X_df.columns[feature_columns])
        return X_df.columns[feature_columns]

    def select_response(self, columns):
        response = self.data.loc[:, columns]
        response = response.to_numpy()
        self.response = response

    def model_predict(self, X):
        return self.model.predict(X)

    def model_train(self, X, y):
        self.model = self.model.fit(X, y)
