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
        plot.savefig(f"dist/{col}_transformed_distribution.png")




data = pd.read_csv("data.csv")
# get only columns containing numeric values
nv = data[numeric_cols].to_numpy()

# log2 transformation on the numeric columns
log2_transform = lambda x: np.log2(x+1)
transformer = preprocessing.FunctionTransformer(log2_transform)
nv_transformed = transformer.transform(nv)

# scaling of the previously transformed values
scaler = preprocessing.StandardScaler().fit(nv_transformed)
nv_preprocessed = scaler.transform(nv_transformed)

trans_data = pd.DataFrame(nv_transformed, columns=numeric_cols)

scaled_data = pd.DataFrame(nv_preprocessed, columns=numeric_cols)

plot_distribution(data, numeric_cols)
plot_transformed_distribution(data, numeric_cols)