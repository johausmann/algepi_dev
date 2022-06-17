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
    def __init__(self, data, model):
        self.data = os.path.abspath(data)
        if not os.path.exists(self.data):
            raise FileNotFoundError("File is missing...")
        self.model = model
        self.data = pd.read_csv(self.data)
        self.predictors = None
        self.features = None

    #def k_split(self, number_of_splits=5, shuffle=True):
    #    assert type(number_of_splits) == int
    #    kf = KFold(n_splits = number_of_splits, shuffle=shuffle)
    #    kf_split = kf.split(self.data)
    #    return kf_split

    def preprocess_data(self, data, columns):
        prepro_data = self.data[columns]
        log2_transform = lambda x: np.log2(x+1)
        transformer = preprocessing.FunctionTransformer(log2_transform)
        data_transformed = transformer.transform(prepro_data)
        scaler = preprocessing.RobustScaler().fit(data_transformed)
        return scaler.transform(data_transformed), scaler

    def backtransform_data(self, values, scaler):
        values = values.reshape(len(values), 1)
        values_re = scaler.inverse_transform(values)
        values_re = values_re.reshape(1, len(values_re))[0]
        return np.square(values_re)

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

    @staticmethod
    def get_model():
        return MLPRegressor(verbose=True, hidden_layer_sizes=(128,100,64,32), 
                            learning_rate='invscaling', activation='relu', 
                            early_stopping=True)

    #def random_forest_regressor(self, split=False, kf=None):
    #    if self.predictors is None or self.response is None:
    #        return
    #    if split and kf is None:
    #        print("Split option requires an kf.split instance...")
    #        return
    #    if not split:
    #        regr = RandomForestRegressor(n_estimators=50,
    #                                     random_state=0,
    #                                     n_jobs=10).fit(self.predictors, self.response)
    #        y_pred = regr.predict(self.predictors[test_index])
    #    else:
    #        for train_index, test_index in kf:
    #            regr = RandomForestRegressor(n_estimators=50,
    #                                         random_state=0,
    #                                         n_jobs=10).fit(self.predictors[train_index], self.response[train_index])
    #            y_pred = regr.predict(self.predictors[test_index])
    #    return y_pred


data = pd.read_csv("data.csv")

kf = KFold(n_splits=5, shuffle=True)

X = data.loc[:, data.columns.str.endswith("150")].to_numpy()
print(X)
y = data["mRNA"].to_numpy()
print(y)
i = 1

for train_index, test_index in kf.split(X):
    regr = RandomForestRegressor(n_estimators=50, random_state=0, n_jobs=10).fit(X[train_index], y[train_index])
    #regr = MLPRegressor(random_state=0).fit(X[train_index], y[train_index])
    y_pred = regr.predict(X[test_index])
    score = mean_squared_error(y[test_index], y_pred, squared=False)
    corr = stats.pearsonr(y[test_index], y_pred)
    df = pd.DataFrame()
    df["y"] = y[test_index]
    df["y_pred"] = y_pred
    print(df)
    plot = sns.scatterplot(df["y"], df["y_pred"])
    fig = plot.get_figure()
    fig.savefig(f"out{i}.png")
    fig.clear()
    i+=1
    print(score, ", ", corr)

#cv = ShuffleSplit(n_splits=5, random_state=0)#

#regr = RandomForestRegressor(n_estimators=50, random_state=0, n_jobs=10)

#regr = MLPRegressor(random_state=0, verbose=True)

#scores = cross_val_score(regr, X, y, cv=cv, scoring='neg_root_mean_squared_error')

#print(scores)
