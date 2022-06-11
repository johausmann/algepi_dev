import seaborn as sns
import pandas as pd

data = pd.read_csv("data.csv")

for col in data.iloc[:, 1:-2].columns.values:
    plot = sns.scatterplot(data=data, x=col, y=list(data.columns.values)[-1])
    fig = plot.get_figure()
    fig.savefig(f"{col}_y.png")
    fig.clear()