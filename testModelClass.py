import os
from argparse import ArgumentParser
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from scipy import stats
import seaborn as sns
import pandas as pd
from ModelClass import Model
from utils import create_dir

class testModel(Model):
    def __init__(self, data, outdir, plot):
        """Class initialization"""
        Model.__init__(self, data)
        self.outdir = outdir if os.path.exists(outdir)
        self.plot = plot
        if self.plot and not self.outdir:
            raise FileNotFoundError("If plots should be generated a directory must be specified...")


    def k_split(self, number_of_splits=5, shuffle=True):
        """Create KFold instance to perform k split cross validation"""
        assert type(number_of_splits) == int
        kf = KFold(n_splits = number_of_splits, shuffle=shuffle)
        kf_split = kf.split(self.data)
        return kf_split

    def search_grid(self):
        """Method to search for optimal parameters of the selected estimator"""
        data_train = self.data
        X_train = self.preprocess_data(data_train, self.predictors)
        y_train = self.log_transform(data_train, self.response)
        estimator = RandomForestRegressor()
        param_grid = { 
            "n_estimators" : [75,80,90,95,100,105,110,115,120,125],
            "max_depth" : [7,10,14,20,25,30,50,75],
            "max_features" : ["sqrt", "log2"],
            "min_samples_split" : [2,3,4],
            "min_samples_leaf" : [1,2],
            }
        grid = GridSearchCV(estimator, param_grid, n_jobs=2, cv=5)
        grid.fit(X_train, y_train.values.ravel())
        return grid.best_params_

    def run_tests(self, kf_split):
        """Test model performance/quality with cross validation"""
        for i, (train_index, test_index) in enumerate(kf_split):
            data_train = self.data.iloc[train_index]
            data_test = self.data.iloc[test_index]

            X_train = self.preprocess_data(data_train, self.predictors)
            y_train = self.log_transform(data_train, self.response)

            X_test = self.preprocess_data(data_test, self.predictors)
            y_test = data_test.mRNA.to_numpy()

            # Fit estimator to split train data
            self.model.fit(X_train, y_train.values.ravel())

            y_pred = self.model.predict(X_test)

            # back transformation
            y_pred_re = self.backtransform_data(y_pred)

            print("Calculating quality of model...")
            score = mean_squared_error(y_test, y_pred_re, squared=False)
            corr = stats.pearsonr(y_test, y_pred_re)
            print(f"Fold {i}:\nRSME: {score}, correlation: {corr}")

            if self.plot:
                print("Started ploting...")
                create_dir(self.outdir)
                df = pd.DataFrame()
                df["y"] = y_test
                df["y_pred"] = y_pred_re

                plot = sns.jointplot(data=df, x="y", y="y_pred", kind="reg")
                plot.fig.suptitle(f"Fold {i}. RMSE={score}, corr={corr}")
                plot.savefig(f"{self.outdir}/fold_{i}.png")


def main():
    """Main method for cmd interface"""
    parser = ArgumentParser(description='Test a specified model.')
    parser.add_argument('-i', dest='input', help='The dataset where the model should be tested on.')
    parser.add_argument('-p', action='store_true', help='If plots should be created.')
    parser.add_argument('-o', dest='output', help='Directory name where plots should be stored.')
    parser.add_argument('-j', dest='jobs', help='Number of jobs', default=1, required=False)
    args = parser.parse_args()
    test_model = testModel(args.input, args.output, args.p)

    select_col = list(test_model.data.columns)
    select_col.remove('PromoterSeq')
    select_dat = test_model.preprocess_data(test_model.data, select_col)
    test_model.prepro_cols = test_model.select_best_predictors(
           select_dat,
           test_model.log_transform(test_model.data, test_model.response),
           n_features=5,
           direction="forward",
           cpu=args.jobs)
    kf_split = test_model.k_split(number_of_splits=5)
    test_model.run_tests(kf_split)

if __name__ == "__main__":
    main()
