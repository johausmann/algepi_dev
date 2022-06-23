import os
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

from utils import one_hot_dna

class Model():
    """
    Base class that provides general functions for model selection
    training and prediction.
    """
    def __init__(self, data):
        self.data = os.path.abspath(data)
        if not os.path.exists(self.data):
            raise FileNotFoundError("File is missing...")
        self.model = self.get_model()
        self.data = pd.read_csv(self.data)
        # the column for promoter sequence if promoter should be used for
        # prediction only on the promoter sequence prediction will be done.
        self.promoter = ["PromoterSeq"]
        # the columns on which the model should be trained and predict
        self.predictors = ["MNase_GB", "H3K4me3_GB", "H3K9ac_GB", "GeneLen", 
                           "sRNA", "IntronFreq", "GC"]
        # old predictor columns
        #['MNase_GB', 'H3K27me3_GB', 'sRNA', 'GC', 'H3K9ac_TSSm150']
        
        self.response = ["mRNA"]

        # the possible characters in promoter seq 
        # needed for one hot encoding if used
        self.alphabet = "ACGTN"
        self.used_columns = []

    def get_model(self):
        return AdaBoostRegressor(RandomForestRegressor(max_depth=10, max_features="sqrt", 
                                                       min_samples_leaf=2,min_samples_split=3,
                                                       n_estimators=110,n_jobs=5))

    def preprocess_data(self, data, columns):
        """
        Perform log2(x+1) transformation on features that are not normally
        distributed. Scaling to get the values in range ~ [-1,1] afterwards.
        GC and PromoterSeq columns are handled differently because of normal
        distribution and one hot encoding.
        """
        cols_to_transf = [i for i in columns if i not in ["GC", "PromoterSeq", 
                                                          "mRNA", "GeneID"]]
        data_preprocessed = pd.DataFrame()
        if len(cols_to_transf) != 0:
            data_transformed = self.log_transform(data, cols_to_transf)
            scaler = preprocessing.RobustScaler().fit(data_transformed)
            data_preprocessed = scaler.transform(data_transformed)
            data_preprocessed = pd.DataFrame(data_preprocessed, columns=cols_to_transf)
            if "GC" in columns:
                gc = data.GC.to_numpy()/100
                data_preprocessed["GC"] = gc
            self.used_columns = data_preprocessed.columns
            data_preprocessed = data_preprocessed.to_numpy()
        if "PromoterSeq" in columns:
            print("Converting sequence data (promoter) to ")
            data_preprocessed = one_hot_dna(data["PromoterSeq"].values)
            self.used_columns = ["PromoterSeq"]
        return data_preprocessed

    def log_transform(self, data, columns):
        """
        Transforms specified columns in dataframe using log2(x+1).
        """
        log2_transform = lambda x: np.log2(x+1)
        transformer = preprocessing.FunctionTransformer(log2_transform)
        return transformer.transform(data[columns])

    def backtransform_data(self, values):
        """
        Backtransforms the log2(x+1).
        Needed for getting back to TPM values.
        """
        return np.power(2, values) - 1

    def model_predict(self, X):
        """
        Wrapper method to predict using a trained model.
        """
        return self.model.predict(X)

    def model_train(self, X, y):
        """
        Wrapper method to train a model.
        """
        self.model = self.model.fit(X, y.values.ravel())
