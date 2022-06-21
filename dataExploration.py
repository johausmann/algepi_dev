from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

numeric_cols = ["MNase_TSSm150", "H3K27me3_TSSm150", "H3K4me3_TSSm150", 
                "H3K9ac_TSSm150", "Pol2_TSSm150", "MNase_TSSp300", 
                "H3K27me3_TSSp300", "H3K4me3_TSSp300", "H3K9ac_TSSp300", 
                "Pol2_TSSp300", "MNase_TTSm200", "H3K27me3_TTSm200", 
                "H3K4me3_TTSm200", "H3K9ac_TTSm200", "Pol2_TTSm200", 
                "MNase_GB", "H3K27me3_GB", "H3K4me3_GB", "H3K9ac_GB", 
                "Pol2_GB", "IntergenicLen", "GeneLen", "sRNA", "GC", 
                "IntronFreq", "mRNA"]

cols_to_transform = ["MNase_TSSm150", "H3K27me3_TSSm150", "H3K4me3_TSSm150", 
                "H3K9ac_TSSm150", "Pol2_TSSm150", "MNase_TSSp300", 
                "H3K27me3_TSSp300", "H3K4me3_TSSp300", "H3K9ac_TSSp300", 
                "Pol2_TSSp300", "MNase_TTSm200", "H3K27me3_TTSm200", 
                "H3K4me3_TTSm200", "H3K9ac_TTSm200", "Pol2_TTSm200", 
                "MNase_GB", "H3K27me3_GB", "H3K4me3_GB", "H3K9ac_GB", 
                "Pol2_GB", "IntergenicLen", "GeneLen", "sRNA", 
                "IntronFreq", "mRNA"]

plot_dirs = ["plots", "plots/dist", "plots/dist_transformed", 
             "plots/dist_scaled"]

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_distribution(data, cols):
    print(data.columns.values)
    for col in cols:
        plot = sns.displot(data, x=col, kind="kde", fill=True)
        plot.set_titles("{col} distribution")
        plot.savefig(f"plots/dist/{col}_distribution.png")

def plot_transformed_distribution(data, cols):
    for col in cols:
        plot = sns.displot(data, x=col, kind="kde", fill=True)
        plot.set_titles("{col} transformed distribution")
        plot.savefig(f"plots/dist_transformed/{col}_transformed_distribution.png")

def plot_scaled_distribution(data, cols):
    for col in cols:
        plot = sns.displot(data, x=col, kind="kde", fill=True)
        plot.set_titles("{col} scaled distribution")
        plot.savefig(f"plots/dist_scaled/{col}_scaled_distribution.png")

def plot_correlation_heatmap(data):
    cor = data.corr()
    fig, ax = plt.subplots(figsize=(20,20))  
    plot = sns.heatmap(cor, annot=True, cmap=plt.cm.Reds, ax=ax)
    fig = plot.get_figure()
    fig.savefig(f"plots/correlation_heatmap.png")



data = pd.read_csv("data.csv")

for dir in plot_dirs:
    create_dir(dir)

gc = data.GC.to_numpy()/100
# check outliers
print(data.loc[data["mRNA"] > 2000][["mRNA", "GeneLen"]])

# get only columns containing numeric values
nv = data[cols_to_transform].to_numpy()

# log2 transformation on the numeric columns
log2_transform = lambda x: np.log2(x+1)
transformer = preprocessing.FunctionTransformer(log2_transform)
nv_transformed = transformer.transform(nv)

# scaling of the previously transformed values
scaler = preprocessing.RobustScaler().fit(nv_transformed)
nv_preprocessed = scaler.transform(nv_transformed)

trans_data = pd.DataFrame(nv_transformed, columns=cols_to_transform)
trans_data["GC_scaled"] = gc

scaled_data = pd.DataFrame(nv_preprocessed, columns=cols_to_transform)
scaled_data["GC_scaled"] = gc

plot_distribution(data[numeric_cols], numeric_cols)
plot_transformed_distribution(trans_data, trans_data.columns.values)
plot_scaled_distribution(scaled_data, scaled_data.columns.values)
plot_correlation_heatmap(trans_data)



