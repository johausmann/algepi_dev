from sklearn import preprocessing
import seaborn as sns
import pandas as pd
import numpy as np

numeric_cols = ["MNase_TSSm150", "H3K27me3_TSSm150", "H3K4me3_TSSm150", 
                "H3K9ac_TSSm150", "Pol2_TSSm150", "MNase_TSSp300", 
                "H3K27me3_TSSp300", "H3K4me3_TSSp300", "H3K9ac_TSSp300", 
                "Pol2_TSSp300", "MNase_TTSm200", "H3K27me3_TTSm200", 
                "H3K4me3_TTSm200", "H3K9ac_TTSm200", "Pol2_TTSm200", 
                "MNase_GB", "H3K27me3_GB", "H3K4me3_GB", "H3K9ac_GB", 
                "Pol2_GB", "IntergenicLen", "GeneLen", "sRNA", "GC", 
                "IntronFreq", "mRNA"]

def plot_distribution(data, cols):
    for col in cols:
        plot = sns.displot(data, x=col, kind="kde", fill=True)
        plot.set_titles("{col} distribution")
        plot.savefig(f"dist/{col}_distribution.png")

def plot_transformed_distribution(data, cols):
    for col in cols:
        plot = sns.displot(data, x=col, kind="kde", fill=True)
        plot.set_titles("{col} transformed distribution")
        plot.savefig(f"dist_transformed/{col}_transformed_distribution.png")

def plot_scaled_distribution(data, cols):
    for col in cols:
        plot = sns.displot(data, x=col, kind="kde", fill=True)
        plot.set_titles("{col} scaled distribution")
        plot.savefig(f"dist_scaled/{col}_scaled_distribution.png")



data = pd.read_csv("data.csv")
# get only columns containing numeric values
print(data.loc[data["mRNA"] > 2000][["mRNA", "GeneLen"]])
nv = data[numeric_cols].to_numpy()

# log2 transformation on the numeric columns
log2_transform = lambda x: np.log2(x+1)
transformer = preprocessing.FunctionTransformer(log2_transform)
nv_transformed = transformer.transform(nv)

# scaling of the previously transformed values
#scaler = preprocessing.RobustScaler().fit(nv_transformed)
print(data.shape[0])
nv = data["mRNA"].to_numpy()
print(nv)
nv = nv.reshape(len(nv), 1)
print(nv)
nv = nv + 1
pt = preprocessing.PowerTransformer(method="yeo-johnson")
pt_transformer = pt.fit(nv)
nv_preprocessed = pt_transformer.transform(nv)
#nv_preprocessed = preprocessing.power_transform(nv, method='box-cox')
#nv_preprocessed = boxcox.transform(nv)
#nv_preprocessed = scaler.transform(nv_transformed)

#trans_data = pd.DataFrame(nv_transformed, columns=numeric_cols)

scaled_data = pd.DataFrame(nv_preprocessed, columns=["mRNA"])

#plot_distribution(data, numeric_cols)
#plot_transformed_distribution(trans_data, numeric_cols)
plot_scaled_distribution(scaled_data, ["mRNA"])

print(pt.inverse_transform(nv_preprocessed))


for b, a in zip(nv, nv_preprocessed):
    if b != a:
        print(b, " ", a)