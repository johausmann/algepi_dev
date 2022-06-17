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
from ModelClass import Model

class testModel(Model):
    def __init__(self, data):
        Model.__init__(self, data)
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
            data_train = data.iloc[train_index]
            data_test = data.iloc[test_index]
            
            X_train, _ = self.preprocess_data(data_train, self.prepro_cols)
            y_train = self.log_transform(data_train, ["mRNA"])

            X_test, _ = self.preprocess_data(data_test, self.prepro_cols)
            y_test = data_test.mRNA.to_numpy()

            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)
            
            # back transformation
            y_pred_re = self.backtransform_data(y_pred)

            score = mean_squared_error(y_test, y_pred_re, squared=False)
            corr = stats.pearsonr(y_test, y_pred_re)
            df = pd.DataFrame()
            df["y"] = y_test
            df["y_pred"] = y_pred_re
        
            plot = sns.jointplot(data=df, x="y", y="y_pred", kind="reg")
            plot.fig.suptitle(f"Fold {i}. RMSE={score}, corr={corr}")
            plot.savefig(f"out{i}.png")
            i+=1
            print(f"RSME: {score}, correlation: {corr}")


def main():
    parser = ArgumentParser(description='Test a specified model.')
    parser.add_argument('-i', dest='input', help='The dataset where the model should be tested on.')
    args = parser.parse_args()

    test_model = testModel(args.input)
    kf_split = test_model.k_split()
    test_model.run_tests(kf_split, test_model.data)

if __name__ == "__main__":
    main()
