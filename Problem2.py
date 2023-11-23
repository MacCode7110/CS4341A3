# Logistic Regression and Regularization Algorithm

# Parts a, b, and c - implementation, training, and testing on chronic_kidney_disease dataset.

# Referencing this resource to gain a better understanding of logistic regression: https://datatab.net/tutorial/logistic-regression

# Logistic regression is a machine learning algorithm used for binary classification, and it models a binary dependent variable.
# Logistic regression predicts the probability of an observation belonging to a certain class or label.
# Classification is a part of supervised machine learning that predicts which category some observation (the dataset linked to the given entity) belongs to based on its features.
# Regularization attempts to limit the complexity of the data model (prevent overfitting of data and improve generalization of newly introduced data).

# In the chronic_kidney_disease_full.arff file, here is how I have decided that non-numeric values and question marks are handled and manipulated so that the logistic regression works properly:
#   1. Question marks:
#       A. All question marks are replaced with the current average value for that feature.
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
data_frame_for_all_data = pd.DataFrame(all_data[0])


# Substitute non-numeric values for numeric values in dataframe:

def replace_non_numeric_values_in_rbc_column():
    pass

def replace_non_numeric_values_in_pc_column():
    pass


for (feature_name, feature_data) in data_frame_for_all_data.iteritems():
    if feature_name == 'rbc':
        data_frame_for_all_data['rbc'] = data_frame_for_all_data['rbc'].map(replace_non_numeric_values_in_rbc_column)
    elif feature_name == 'pc':
        data_frame_for_all_data['pc'] = data_frame_for_all_data['pc'].map(replace_non_numeric_values_in_pc_column)


# Substitute question marks for numeric values in dataframe:

all_data_nd_array = data_frame_for_all_data.to_numpy()
nd_array_for_training_data_processing = all_data_nd_array.copy()
modified_nd_array_for_training_data_processing = np.delete(nd_array_for_training_data_processing, 24, 1)
