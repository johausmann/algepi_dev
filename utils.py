import numpy as np
import os

def one_hot_dna(promoters, alphabet="ACTGN"):
    """
    Converts list of promotor sequences into one hot encoding.
    """
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))

    onehot_matrix = []
    for promoter in promoters:
        # integer encode input data
        integer_encoded = [char_to_int[char] for char in promoter]
        # one hot encode
        onehot_encoded = []
        for value in integer_encoded:
            letter = [0 for _ in range(len(alphabet))]
            letter[value] = 1
            for e in letter: 
                onehot_encoded.append(e)
        onehot_matrix.append(onehot_encoded)
    return np.array(onehot_matrix)


def create_dir(directory):
    """
    Safely create directory if does not already exists.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)