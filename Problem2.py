# Logistic Regression and Regularization Algorithm

# Parts a, b, and c - implementation, training, and testing on chronic_kidney_disease dataset.

# Referencing this resource to gain a better understanding of logistic regression: https://datatab.net/tutorial/logistic-regression

# Logistic regression is a machine learning algorithm used for binary classification, and it models a binary dependent variable.
# Logistic regression predicts the probability of an observation belonging to a certain class or label.
# Classification is a part of supervised machine learning that predicts which category some observation (the dataset linked to the given entity) belongs to based on its features.
# Regularization attempts to limit the complexity of the data model (prevent overfitting of data and improve generalization of newly introduced data).

# In the chronic_kidney_disease_full.arff file, here is how I have decided that non-numeric values and question marks are handled and manipulated so that the logistic regression works properly:
#   1. Question marks:
#       A. All question marks are replaced with the current average value (mean) for that feature, which operates under the assumption that each column contains similar data.
#   2. Non-numeric values (listed by feature/attribute):
#       A. 'rbc', index 5: normal -> 0, abnormal -> 1
#       B. 'pc', index 6: normal -> 0, abnormal -> 1
#       C. 'pcc', index 7: present -> 1, notpresent -> 0
#       D. 'ba', index 8: present -> 1, notpresent -> 0
#       E. 'htn', index 18: yes -> 1, no -> 0
#       F. 'dm', index 19: yes -> 1, no -> 0
#       G. 'cad', index 20: yes -> 1, no -> 0
#       H. 'appet', index 21: good -> 0, poor -> 1
#       I. 'pe', index 22: yes -> 1, no -> 0
#       J. 'ane', index 23: yes -> 1, no -> 0
#       K. 'class', index 24: ckd -> 1, notckd -> 0

from scipy.io import arff
import pandas as pd
import numpy as np
from numpy import log, dot, e, shape
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class LogisticRegressionAlgorithm:
    def __init__(self, learning_rate, number_of_iterations_for_gradient_descent, regularization_parameter,
                 use_standardization):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations_for_gradient_descent
        self.regularization_parameter = regularization_parameter
        self.use_standardization = use_standardization
        self.training_weights = []
        self.bias = 0

    def train_using_gradient_descent(self, number_of_data_samples_by_number_of_features_2d_array):
        number_of_data_samples, number_of_features = number_of_data_samples_by_number_of_features_2d_array.shape
        weighted_sum_of_input_features = 0

        for i1 in range(len(number_of_features)):
            self.training_weights.append(0)

        for i2 in range(len(self.number_of_iterations)):
            pass

    def apply_regularization(self):
        pass

    def apply_standardization(self):
        pass

    def make_predictions_using_test_data(self):
        pass

    def compute_confusion_matrix_using_testing_data(self):
        pass

    def compute_f_measure_using_confusion_matrix(self):
        pass

    def run_algorithm(self, number_of_data_samples_by_number_of_features_per_data_sample_2d_array):
        if not self.use_standardization:
            self.train_using_gradient_descent(number_of_data_samples_by_number_of_features_per_data_sample_2d_array)
            self.apply_regularization()
            self.make_predictions_using_test_data()
            self.compute_confusion_matrix_using_testing_data()
            self.compute_f_measure_using_confusion_matrix()
        else:
            self.train_using_gradient_descent(number_of_data_samples_by_number_of_features_per_data_sample_2d_array)
            self.apply_regularization()
            self.apply_standardization()
            self.make_predictions_using_test_data()
            self.compute_confusion_matrix_using_testing_data()
            self.compute_f_measure_using_confusion_matrix()


# Process the data:

all_data = arff.loadarff('chronic_kidney_disease_full.arff')
dataframe_for_all_data = pd.DataFrame(all_data[0])
chronic_kidney_disease_data_columns_using_byte_strings = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe',
                                                          'ane', 'class']

# Pandas dataframe uses byte strings to represent strings by default. These byte strings need to be decoded into regular strings for the applicable columns:

for i1 in range(len(chronic_kidney_disease_data_columns_using_byte_strings)):
    dataframe_for_all_data[chronic_kidney_disease_data_columns_using_byte_strings[i1]] = dataframe_for_all_data[
        chronic_kidney_disease_data_columns_using_byte_strings[i1]].str.decode("utf-8")

# 1. Substitute non-numeric string values for numeric values in dataframe:
# 2. Substitute non-numeric nominal integers and floats for numeric values in dataframe:

for (feature_name, feature_data) in dataframe_for_all_data.items():
    if feature_name == 'rbc':
        dataframe_for_all_data.replace({feature_name: {'normal': '0', 'abnormal': '1'}}, inplace=True)
        dataframe_for_all_data[feature_name] = pd.to_numeric(dataframe_for_all_data[feature_name], errors='coerce')
    elif feature_name == 'pc':
        dataframe_for_all_data.replace({feature_name: {'normal': '0', 'abnormal': '1'}}, inplace=True)
        dataframe_for_all_data[feature_name] = pd.to_numeric(dataframe_for_all_data[feature_name], errors='coerce')
    elif feature_name == 'pcc':
        dataframe_for_all_data.replace({feature_name: {'present': '1', 'notpresent': '0'}}, inplace=True)
        dataframe_for_all_data[feature_name] = pd.to_numeric(dataframe_for_all_data[feature_name], errors='coerce')
    elif feature_name == 'ba':
        dataframe_for_all_data.replace({feature_name: {'present': '1', 'notpresent': '0'}}, inplace=True)
        dataframe_for_all_data[feature_name] = pd.to_numeric(dataframe_for_all_data[feature_name], errors='coerce')
    elif feature_name == 'htn':
        dataframe_for_all_data.replace({feature_name: {'yes': '1', 'no': '0'}}, inplace=True)
        dataframe_for_all_data[feature_name] = pd.to_numeric(dataframe_for_all_data[feature_name], errors='coerce')
    elif feature_name == 'dm':
        dataframe_for_all_data.replace({feature_name: {'yes': '1', 'no': '0'}}, inplace=True)
        dataframe_for_all_data[feature_name] = pd.to_numeric(dataframe_for_all_data[feature_name], errors='coerce')
    elif feature_name == 'cad':
        dataframe_for_all_data.replace({feature_name: {'yes': '1', 'no': '0'}}, inplace=True)
        dataframe_for_all_data[feature_name] = pd.to_numeric(dataframe_for_all_data[feature_name], errors='coerce')
    elif feature_name == 'appet':
        dataframe_for_all_data.replace({feature_name: {'good': '0', 'poor': '1'}}, inplace=True)
        dataframe_for_all_data[feature_name] = pd.to_numeric(dataframe_for_all_data[feature_name], errors='coerce')
    elif feature_name == 'pe':
        dataframe_for_all_data.replace({feature_name: {'yes': '1', 'no': '0'}}, inplace=True)
        dataframe_for_all_data[feature_name] = pd.to_numeric(dataframe_for_all_data[feature_name], errors='coerce')
    elif feature_name == 'ane':
        dataframe_for_all_data.replace({feature_name: {'yes': '1', 'no': '0'}}, inplace=True)
        dataframe_for_all_data[feature_name] = pd.to_numeric(dataframe_for_all_data[feature_name], errors='coerce')
    elif feature_name == 'class':
        dataframe_for_all_data.replace({feature_name: {'ckd': '1', 'notckd': '0'}}, inplace=True)
        dataframe_for_all_data[feature_name] = pd.to_numeric(dataframe_for_all_data[feature_name], errors='coerce')
    elif feature_name == 'sg':
        dataframe_for_all_data[feature_name] = pd.to_numeric(dataframe_for_all_data[feature_name], errors='coerce')
    elif feature_name == 'al':
        dataframe_for_all_data[feature_name] = pd.to_numeric(dataframe_for_all_data[feature_name], errors='coerce')
    elif feature_name == 'su':
        dataframe_for_all_data[feature_name] = pd.to_numeric(dataframe_for_all_data[feature_name], errors='coerce')

# Substitute NAN (forced from question marks via errors=coerce argument) for numeric averages in dataframe:

for (feature_name, feature_data) in dataframe_for_all_data.items():
    dataframe_for_all_data[feature_name].fillna(value=dataframe_for_all_data[feature_name].mean(), inplace=True)

# Split the data into 80% training data and 20% testing data:

dataframe_for_all_data_copy = dataframe_for_all_data.copy()
training_data, testing_data = train_test_split(dataframe_for_all_data_copy, test_size=0.2, shuffle=True)
# Need to remove class column from training data so logistic regression works as intended:
training_data.drop('class', inplace=True, axis=1)

# Convert training and testing data to Numpy arrays:

training_data_nd_array = training_data.to_numpy()
testing_data_nd_array = testing_data.to_numpy()



