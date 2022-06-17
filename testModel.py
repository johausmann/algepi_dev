from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")

#data = data.loc[data.mRNA < 2000]

numeric_cols = ["MNase_TSSm150", "H3K27me3_TSSm150", "H3K4me3_TSSm150", 
                "H3K9ac_TSSm150", "Pol2_TSSm150", "MNase_TSSp300", 
                "H3K27me3_TSSp300", "H3K4me3_TSSp300", "H3K9ac_TSSp300", 
                "Pol2_TSSp300", "MNase_TTSm200", "H3K27me3_TTSm200", 
                "H3K4me3_TTSm200", "H3K9ac_TTSm200", "Pol2_TTSm200", 
                "MNase_GB", "H3K27me3_GB", "H3K4me3_GB", "H3K9ac_GB", 
                "Pol2_GB", "IntergenicLen", "GeneLen", "sRNA", "GC", 
                "IntronFreq"]

prepro_cols = ["MNase_TSSm150", "H3K27me3_TSSm150", "H3K4me3_TSSm150", 
               "H3K9ac_TSSm150", "Pol2_TSSm150", "MNase_TSSp300", 
               "H3K27me3_TSSp300", "H3K4me3_TSSp300", "H3K9ac_TSSp300", 
               "Pol2_TSSp300", "MNase_TTSm200", "H3K27me3_TTSm200", 
               "H3K4me3_TTSm200", "H3K9ac_TTSm200", "Pol2_TTSm200"]

# get the features that the model should use
X = data[prepro_cols]

# gc value is already normal distributed and has only to be mapped into
# values between 0 and 1
gc = data.GC.to_numpy()/100

# get the target variable
y = data["mRNA"].to_numpy()

# log2 transformation due to distribution of values
# we have many small values and only some large values
log2_transform = lambda x: np.log2(x+1)
transformer = preprocessing.FunctionTransformer(log2_transform)
X_transformed = transformer.transform(X)

# scaling to get the values in a range between -1 to 1 or 
# 0 and 1
#scaler = preprocessing.StandardScaler().fit(X_transformed)
#scaler = preprocessing.MinMaxScaler().fit(X_transformed)
scaler = preprocessing.RobustScaler().fit(X_transformed)
X_preprocessed = scaler.transform(X_transformed)

# add gc content after preprocessing of the other values
X_preprocessed = np.concatenate((X_preprocessed, gc[:,None]), axis=1)

# also transform the target
#y_transformed = transformer.transform(y)
y_1 = y + 1
y_1 = y_1.reshape(len(y_1), 1)
pt = preprocessing.PowerTransformer(method="box-cox")
y_preprocessed = pt.fit_transform(y_1)
#y_scaler = preprocessing.RobustScaler().fit(y_transformed[:,None])

#y_preprocessed = y_scaler.transform(y_transformed[:,None])

# split the training dataset into 5 folds
kf = KFold(n_splits=6, shuffle=True)
# and train on these 5 folds of test and train set
#regr = RandomForestRegressor(n_jobs=10,max_depth = 10, n_estimators = 70, random_state = 18,
#                                 #min_samples_split = 3, min_samples_leaf = 2,
#                                 max_features = 4)
regr = MLPRegressor(verbose=True, hidden_layer_sizes=(128,100,64, 32), learning_rate='invscaling', activation='relu', early_stopping=True)

for i, (train_index, test_index) in enumerate(kf.split(X_preprocessed)):
    regr.fit(X_preprocessed[train_index], y_preprocessed[train_index])
    #regr = MLPRegressor(verbose=True).fit(X_preprocessed[train_index], y_preprocessed[train_index])
    #regr = LogisticRegression(random_state=0, n_jobs=10).fit(X_preprocessed[train_index], y_preprocessed[train_index])
    #regr = SGDRegressor().fit(X_preprocessed[train_index], y_preprocessed[train_index])
    y_pred = regr.predict(X_preprocessed[test_index])
    
    # back transformation
    y_pred = y_pred.reshape(len(y_pred), 1)
    y_pred_re = pt.inverse_transform(y_pred) - 1
    y_pred_re = y_pred_re.reshape(1, len(y_pred_re))[0]
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