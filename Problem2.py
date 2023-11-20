# Logistic Regression and Regularization Algorithm

# Parts a and b - implementation, training, and testing on chronic_kidney_disease dataset.

# Referencing this resource to gain a better understanding of logistic regression: https://datatab.net/tutorial/logistic-regression

# Logistic regression is a machine learning algorithm used for binary classification, and it models a binary dependent variable.
# Logistic regression predicts the probability of an observation belonging to a certain class or label.
# Classification is a part of supervised machine learning that predicts which category some observation (the dataset linked to the given entity) belongs to based on its features.
# Regularization attempts to limit the complexity of the data model (prevent overfitting of data and improve generalization of newly introduced data).

# Questions for OH 11/20/23:
#   1. Clarify: Is this a multiple logistic regression model problem?
#   2. How should we go about plotting the f-measure and what exactly does this plot look like?
#   3. What do the equations used to calculate the f-measure mean?
#   4. Should regularization be used with the standardization protocol?
#   5. How should we handle non-numeric values in the sum of weighted sum of inputs calculation?
#   6. What exactly is the weighted sum of input features?

import numpy as np
from numpy import log, dot, e, shape
import matplotlib.pyplot as plt


class LogisticRegressionAlgorithm:
    def __init__(self, learning_rate, number_of_iterations_for_gradient_descent, regularization_parameter, use_standardization):
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

    def make_predictions_using_test_data(self):
        pass

    def run_algorithm(self, number_of_data_samples_by_number_of_features_per_data_sample_2d_array):
        if not self.use_standardization:
            self.train_using_gradient_descent(number_of_data_samples_by_number_of_features_per_data_sample_2d_array)
            self.make_predictions_using_test_data()


