import pickle
from argparse import ArgumentParser
from ModelClass import Model


class predictModel(Model):
    def __init__(self, data, modelfile, outfile):
        """Class initialization"""
        Model.__init__(self, data)
        self.model = None
        self.outfile = outfile
        self.modelfile = modelfile
        
    def add_prediction(self):
        """Run prediction and add response as a new column to the dataframe"""
        X = self.preprocess_data(self.data, self.predictors)
        y_pred = self.model_predict(X)
        y_pred_re = self.backtransform_data(y_pred)
        self.data['y_pred'] = y_pred_re
        self.data.to_csv(self.outfile, index=False)

    def load_model(self):
        """Load serialized model from pickle"""
        self.model = pickle.load(open(self.modelfile, 'rb'))

def main():
    """Main method for command line interface"""
    parser = ArgumentParser(description='Predicts expression scores using a pre-trained model '\
                                        'and adds the predictions to the input file.')
    parser.add_argument('-i', dest='input', help='The dataset where the model should predict on.')
    parser.add_argument('-m', dest='modelfile', help='Location of the previously trained model.')
    parser.add_argument('-o', dest='output', help='Output file where the result should be stored.')
    args = parser.parse_args()

    predict_model = predictModel(args.input, args.modelfile, args.output)
    predict_model.load_model()
    predict_model.add_prediction()


if __name__ == "__main__":
    main()
