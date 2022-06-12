import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def one_hot_dna(dna_string):
    # basic sanit check
    assert type(dna_string) == str
    # Create numpy array from string
    dna_array = np.array(list(dna_string))
    label_encoder = LabelEncoder()
    integer_encoded_seq = label_encoder.fit_transform(dna_array)
    onehot_encoder = OneHotEncoder(sparse=False)
    # Reshape numpy array to be a column vector
    integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
    # Create one hot encoding from integer encoding
    onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
    return onehot_encoded_seq
