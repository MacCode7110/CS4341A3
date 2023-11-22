# Logistic Regression and Regularization Algorithm
import pandas as pd
# Parts a, b, and c - implementation, training, and testing on chronic_kidney_disease dataset.

# Referencing this resource to gain a better understanding of logistic regression: https://datatab.net/tutorial/logistic-regression

# Logistic regression is a machine learning algorithm used for binary classification, and it models a binary dependent variable.
# Logistic regression predicts the probability of an observation belonging to a certain class or label.
# Classification is a part of supervised machine learning that predicts which category some observation (the dataset linked to the given entity) belongs to based on its features.
# Regularization attempts to limit the complexity of the data model (prevent overfitting of data and improve generalization of newly introduced data).

# Questions for OH 11/20/23:
#   1. What exactly is the weighted sum of input features? What values are exactly in this equation?
#       In the 2D array where samples are rows and feature variables are columns, each box (value for the current sample and feature) is going to be multiplied by a DIFFERENT WEIGHT calculated for that box.
#       After each box of the 2D array has been multiplied by a SPECIFICALLY CALCULATED weight, then the bias is added at the end to the weighted sum of input feature values to finish the calculation.
#       We need to determine how important each feature value is to determining the probability a sample belongs to a class. In order to this, different weights need to be applied to different samples.
#       The calculation is then plugged into the sigmoid function, 1 / (1 + e^-x).
#   2. How should we handle the question marks in the sample data? Do these question marks factor into the weighted sum of input features?
#       Could you put zeroes in for the question marks? Maybe. This would skew the data.
#       Better approach with the assumption that all the data is similar: Take the average of all the values in the current column (feature), and assign that average to replace the current question mark.
#   2. How should we handle non-numeric values in the sum of weighted sum of inputs calculation? (If we are handling actual data values).
#       For non-numeric values, denote either a 0 or a 1 (given that the non-numeric value is part of a binary outcome) to represent that non-numeric values.
#   3. Clarify: Is this a multiple logistic regression model problem?
#       Yes
#   4. How should we go about plotting the f-measure and what exactly does this plot look like?
#       Plotting the f-measure is ONLY based on the testing data. Regularization parameters go on the x-axis, and the f-measure (how accurate the class predictions are) go on the y-axis.
#       Need to compute a confusion matrix to find the true positives, true negatives, false positive, false negatives in the testing data.
#       Use equation for f-measure to compute y-axis of graph. f-measure equation is dependent on the true positives, true negatives, false positives, and false negatives.
#       How to run this algorithm for multiple regularization parameters:
#           Loop through all the regularization parameters (create a list):
#               For each regularization parameter, go through ALL the steps in the logistic regression AND call the scatter function to scatter the data for the f-measure.
#           After f-measures have been scattered for each regularization parameter, finally show the resulting scatter plot.
#   6. Should regularization be used with the standardization protocol?
#       Yes, we need regularization to plot the f-measure, and this must be done with the standardization protocol.
#       We can plot the f-measures as a result of standardization with regularization on the same plot as the one scattered through just using regularization. No need to create separate runs for this problem.

#   Definitely use a Numpy array to store the data. Samples make up the rows and features make up the columns.
#   Need to remove last column of @data chunk (ckd or notckd) FROM JUST THE TRAINING DATA as this gives the class of each data sample (kidney disease or not), which defeats the purpose of making predictions.
#   Need to KEEP last column of @data chunk (ckd or notckd) in the TESTING DATA in order to compute the confusion matrix for calculating the f-measure.

from scipy.io import arff
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


all_data = arff.loadarff('chronic_kidney_disease_full.arff')
data_frame_for_all_data = pd.DataFrame(all_data[0])
all_data_nd_array = data_frame_for_all_data.to_numpy()
#  Need to perform replacement on non-numeric values in array and perform averages for question marks
training_data_nd_array = all_data_nd_array.copy()
np.delete(training_data_nd_array, 25, 0)




