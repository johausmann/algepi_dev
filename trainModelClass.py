from argparse import ArgumentParser
from sklearn.utils import shuffle
import pickle

from ModelClass import Model


class trainModel(Model):
    """
    Class to train a model and serialize it.
    """
    def __init__(self, data, outfile):
        Model.__init__(self, data)
        self.outfile = outfile
        
    def train_model(self):
        """
        Trains the model defined in ModelClass.py on predictor columns (also
        defined in ModelClass.py). Shuffles the data before training to avoid
        any biases like sorting of the input data.
        """
        X = self.preprocess_data(self.data, self.predictors)
        y = self.log_transform(self.data, self.response)
        X, y = shuffle(X, y)
        self.model_train(X, y)

    def save_model(self):
        """
        Serializes the trained model as .pickle file.
        """
        pickle.dump(self.model, open(self.outfile, 'wb'))

def main():
    parser = ArgumentParser(description='Train the final model with the entire dataset and write the model to disk.')
    parser.add_argument('-i', dest='input', help='The dataset where the model should be trained on.')
    parser.add_argument('-o', dest='output', help='Output file where the model should be stored.')
    args = parser.parse_args()

    train_model = trainModel(args.input, args.output)
    train_model.train_model()
    train_model.save_model()


if __name__ == "__main__":
    main()