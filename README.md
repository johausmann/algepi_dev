# algepi_dev

Our development branch for the Computational Epigenomics assignment 4, where expression should be predicted.

## Dependencies

Python3 and pip should be available in your path.
To install the required dependencies run:

```
pip install -r requirements.txt
```


## Test Class

The testModelClass.py module can be used to test different model settings and their performance using kfold splits.

The model is set in the method get_model() in the ModelClass.py module.

To run the tests execute:

```
python testModelClass.py -i data.csv
```

If plots should be generated please use the `-p` flag and set the directory where the plots should be stored using the `-o` flag.

## Train Class

If a good model was found using the test class the model can be trained using all given data points.

Therefore run, which trains the model and serializes it to the specified location (`-o`):

```
python trainModelClass.py -i data.csv -o test_model.pickle
```

## Predict Class

This class loads the model and predicts for a given dataset the expression values.

Example usage (after running this command the predicted values are added as a new column to the file specified after `-o`):

```
python predictModelClass.py -i data_sample.csv -m test_model.pickle -o result_test.csv
```
