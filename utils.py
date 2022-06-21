import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#def one_hot_dna(dna_string):
    # basic sanit check
#    assert type(dna_string) == str
    # Create numpy array from string
#    dna_array = np.array(list(dna_string))
#    label_encoder = LabelEncoder()
#    integer_encoded_seq = label_encoder.fit_transform(dna_array)
#    onehot_encoder = OneHotEncoder(sparse=False)
    # Reshape numpy array to be a column vector
#    integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
    # Create one hot encoding from integer encoding
#    onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
#    return onehot_encoded_seq


def one_hot_dna(dna_string, alphabet="ACTGN"):
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # integer encode input data
    integer_encoded = [char_to_int[char] for char in dna_string]
# one hot encode
    onehot_encoded = []
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return np.array(onehot_encoded)
