import os
import sys
from argparse import ArgumentParser
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy import stats
import seaborn as sns
import pandas as pd


class testModel(Model):
    def __init__(self, data, model):
        super.__init__(self, data, model)
        self.prepro_cols = ["MNase_TSSm150", "H3K27me3_TSSm150", 
                            "H3K4me3_TSSm150", "H3K9ac_TSSm150", 
                            "Pol2_TSSm150", "MNase_TSSp300", 
                            "H3K27me3_TSSp300", "H3K4me3_TSSp300", 
                            "H3K9ac_TSSp300", "Pol2_TSSp300", 
                            "MNase_TTSm200", "H3K27me3_TTSm200", 
                            "H3K4me3_TTSm200", "H3K9ac_TTSm200", 
                            "Pol2_TTSm200"]

    def k_split(self, number_of_splits=5, shuffle=True):
        assert type(number_of_splits) == int
        kf = KFold(n_splits = number_of_splits, shuffle=shuffle)
        kf_split = kf.split(self.data)
        return kf_split

    def select_predictors(self, columns):
        header = [x for x in columns if x in self.data.columns]
        features = self.data.loc[:, header]
        features = features.to_numpy()
        self.predictors = features

    def select_response(self, columns):
        response = self.data.loc[:, columns]
        response = response.to_numpy()
        self.response = response

    def run_tests(self, kf_split, data):
        for i, (train_index, test_index) in enumerate(kf_split):
            
            X_preprocessed, X_scaler = self.preprocess_data(data.iloc[train_index, :], self.prepro_cols)
            y_preprocessed, y_scaler = self.preprocess_data(data.iloc[train_index, :], ["mRNA"])
            self.model.fit(X_preprocessed[train_index], y_preprocessed[train_index])
            #regr = MLPRegressor(verbose=True).fit(X_preprocessed[train_index], y_preprocessed[train_index])
            #regr = LogisticRegression(random_state=0, n_jobs=10).fit(X_preprocessed[train_index], y_preprocessed[train_index])
            #regr = SGDRegressor().fit(X_preprocessed[train_index], y_preprocessed[train_index])

            
            y_pred = self.model.predict(X_preprocessed[test_index])
            
            # back transformation
            y_pred_re = self.backtransform_data(y_pred, scaler)

            #y_pred = y_pred.reshape(len(y_pred), 1)
            #y_pred_re = pt.inverse_transform(y_pred) - 1
            #y_pred_re = y_pred_re.reshape(1, len(y_pred_re))[0]
            #print(y_pred[:None][:None])
            #y_pred = y_pred.reshape(len(y_pred), 1)
            #y_pred_re = y_scaler.inverse_transform(y_pred)
            #y_pred_re = y_pred_re.reshape(1, len(y_pred_re))[0]
            #y_pred_re = np.square(y_pred_re)
            score = mean_squared_error(y[test_index], y_pred_re, squared=False)
            corr = stats.pearsonr(y[test_index], y_pred_re)
            df = pd.DataFrame()
            df["y"] = y[test_index]
            df["y_pred"] = y_pred_re
        
            plot = sns.jointplot(data=df, x="y", y="y_pred", kind="reg")
            plot.fig.suptitle(f"Fold {i}. RMSE={score}, corr={corr}")
            plot.savefig(f"out{i}.png")
            i+=1
            print(score, ", ", corr)

    def random_forest_regressor(self, split=False, kf=None):
        if self.predictors is None or self.response is None:
            return
        if split and kf is None:
            print("Split option requires an kf.split instance...")
            return
        if not split:
            regr = RandomForestRegressor(n_estimators=50,
                                         random_state=0,
                                         n_jobs=10).fit(self.predictors, self.response)
            y_pred = regr.predict(self.predictors[test_index])
        else:
            for train_index, test_index in kf:
                regr = RandomForestRegressor(n_estimators=50,
                                             random_state=0,
                                             n_jobs=10).fit(self.predictors[train_index], self.response[train_index])
                y_pred = regr.predict(self.predictors[test_index])
        return y_pred


def main():
    parser = ArgumentParser(description='Test a specified model.')
    parser.add_argument('-i', dest='input', help='The dataset where the model should be tested on.')



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
