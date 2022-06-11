from random import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import matthews_corrcoef
from scipy import stats
import seaborn as sns
import pandas as pd

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