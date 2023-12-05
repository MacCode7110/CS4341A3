# Logistic Regression and Regularization Algorithm

# Parts a, b, and c - implementation, training, and testing on chronic_kidney_disease dataset.

# Referencing the following resources to gain a better understanding of certain parts of logistic regression:
# https://datatab.net/tutorial/logistic-regression

# Logistic regression is a machine learning algorithm used for binary classification, and it models a binary dependent variable.
# Logistic regression predicts the probability of an observation belonging to a certain class or label.
# Classification is a part of supervised machine learning that predicts which category some observation (the dataset linked to the given entity) belongs to based on its features.
# Regularization attempts to limit the complexity of the data model (prevent overfitting of data and improve generalization of newly introduced data).
# This program uses regularization with log loss as shown on the class slides.

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

# When calculating the confusion matrix, 1 represents positive for chronic kidney disease, and 0 represents negative for chronic kidney disease.
# The confusion matrix takes the following form:
#                                   Actual Class
#                                Positive  Negative
#   Predicted Class | Positive   |  TP   |  FP   |
#                               -------------------
#                   | Negative   |  FN   |  TN   |
#
#  (0, 0) = TP, (0, 1) = FP, (1, 0) = FN, (1, 1) = TN

# In the f-measure scatterplot, red dots are those produced using standardized data, and blue dots are those produced using non-standardized data.

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class LogisticRegressionAlgorithm:

    def __init__(self, algorithm_learning_rate, number_of_iterations_for_gradient_descent, regularization_parameter,
                 using_standardization):
        self.learning_rate = algorithm_learning_rate
        self.number_of_iterations = number_of_iterations_for_gradient_descent
        self.regularization_parameter = regularization_parameter
        self.training_weights = []
        self.bias = 0
        self.using_standardization = using_standardization

    def train_using_gradient_descent_and_regularization(self, feature_data_nd_array, target_data_nd_array):
        number_of_data_samples, number_of_features = np.shape(feature_data_nd_array)
        self.training_weights = np.zeros((number_of_data_samples, number_of_features))
        self.training_weights = self.training_weights.T

        for iteration_index in range(self.number_of_iterations):
            # Take the dot product of the data_nd_array with a series of weights to perform gradient descent. Each entry in the feature data is multiplied by a particular weight:
            weighted_sum_of_input_features = np.dot(feature_data_nd_array, self.training_weights) + self.bias
            # Get the regularization expression:
            second_term_in_cost_calculation = (self.regularization_parameter / (2 * number_of_data_samples)) * np.sum(
                np.square(self.training_weights))
            # Compute predictions of chronic kidney disease using sigmoid function:
            chronic_kidney_disease_probability_prediction = 1 / (1 + np.exp(-weighted_sum_of_input_features))
            # Calculate the loss/cost equation:
            cost_calculation = ((-1 / number_of_data_samples) * np.sum((target_data_nd_array * np.log(chronic_kidney_disease_probability_prediction)) + ((1 - target_data_nd_array) * np.log(1 - chronic_kidney_disease_probability_prediction)))) + second_term_in_cost_calculation
            # Recalculate (cost function * derivatives) for training weights and bias. For the following dot product to work correctly, we need to transpose the feature_data_nd_array so that features are the rows and samples are the columns:
            gradient = (1 / number_of_data_samples) * (
                np.dot(feature_data_nd_array.T, (chronic_kidney_disease_probability_prediction - target_data_nd_array)) + np.sum(self.regularization_parameter * self.training_weights))
            adjustment_for_bias = (1 / number_of_data_samples) * np.sum(
                chronic_kidney_disease_probability_prediction - target_data_nd_array)
            print("Cost calculation for iteration " + str(iteration_index) + " using regularization parameter " + str(
                self.regularization_parameter) + " is " + str(cost_calculation))

            # Update the training weights and bias:
            self.training_weights = self.training_weights - (
                    self.learning_rate * gradient)
            self.bias = self.bias - (self.learning_rate * adjustment_for_bias)

    def make_predictions_using_test_data(self, testing_data_nd_array):
        weighted_sum_of_input_features = np.dot(testing_data_nd_array, self.training_weights) + self.bias
        chronic_kidney_disease_probability_predictions = 1 / (1 + np.exp(-weighted_sum_of_input_features))
        number_of_rows, number_of_columns = np.shape(chronic_kidney_disease_probability_predictions)
        chronic_kidney_disease_class_predictions = np.zeros((number_of_rows, number_of_columns))

        for prediction_index_x in range(len(chronic_kidney_disease_probability_predictions)):
            for prediction_index_y in range(len(chronic_kidney_disease_probability_predictions[prediction_index_x])):
                if chronic_kidney_disease_probability_predictions[prediction_index_x][prediction_index_y] > 0.5:
                    chronic_kidney_disease_class_predictions[prediction_index_x][prediction_index_y] = 1
                else:
                    chronic_kidney_disease_class_predictions[prediction_index_x][prediction_index_y] = 0

        return chronic_kidney_disease_class_predictions

    @staticmethod
    def compute_confusion_matrix_using_predictions(chronic_kidney_disease_class_predictions, target_data_nd_array):
        confusion_matrix = np.zeros((2, 2))
        for prediction_index_x in range(len(chronic_kidney_disease_class_predictions)):
            for prediction_index_y in range(len(chronic_kidney_disease_class_predictions[prediction_index_x])):
                # 1 is positive for chronic kidney disease, 0 is negative for chronic kidney disease
                # (0, 0) = TP, (0, 1) = FP, (1, 0) = FN, (1, 1) = TN
                if chronic_kidney_disease_class_predictions[prediction_index_x][prediction_index_y] == 1 and int(
                        target_data_nd_array[prediction_index_x]) == 1:
                    # True positive
                    confusion_matrix[0, 0] += 1
                elif chronic_kidney_disease_class_predictions[prediction_index_x][prediction_index_y] == 1 and int(
                        target_data_nd_array[prediction_index_x]) == 0:
                    # False positive
                    confusion_matrix[0, 1] += 1
                elif chronic_kidney_disease_class_predictions[prediction_index_x][prediction_index_y] == 0 and int(
                        target_data_nd_array[prediction_index_x]) == 0:
                    # True negative
                    confusion_matrix[1, 1] += 1
                elif chronic_kidney_disease_class_predictions[prediction_index_x][prediction_index_y] == 0 and int(
                        target_data_nd_array[prediction_index_x]) == 1:
                    # False negative
                    confusion_matrix[1, 0] += 1

        return confusion_matrix

    @staticmethod
    def compute_f_measure_using_confusion_matrix(confusion_matrix):
        precision = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
        recall = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])
        f_measure = (2 * precision * recall) / (precision + recall)
        return f_measure

    def draw_scatter_plot(self, f_measure, show_plot):
        if not self.using_standardization:
            plt.scatter(self.regularization_parameter, f_measure, c='b')
        else:
            plt.scatter(self.regularization_parameter, f_measure, c='r')

        if show_plot:
            plt.xlabel('Regularization Parameter')
            plt.ylabel('f-measure')
            plt.legend(["Data Without Standardization", "Standardized Data"], loc="lower right")
            plt.show()

    def run_algorithm(self, feature_data_nd_array, target_data_nd_array, run_on_training_data, show_plot):
        if not run_on_training_data:
            chronic_kidney_disease_class_predictions = self.make_predictions_using_test_data(feature_data_nd_array)
            confusion_matrix = self.compute_confusion_matrix_using_predictions(chronic_kidney_disease_class_predictions,
                                                                               target_data_nd_array)
            f_measure = self.compute_f_measure_using_confusion_matrix(confusion_matrix)
            self.draw_scatter_plot(f_measure, show_plot)
        else:
            self.train_using_gradient_descent_and_regularization(feature_data_nd_array, target_data_nd_array)


# Using standardization formula from slides: xi = (xi - mean of training_data) / standard deviation of training data. This standardization formula is applied for each column using the [:, feature_array_index] notation, which selects a particular column (subarray):
def apply_standardization_to_data(feature_data_nd_array):
    updated_feature_nd_array = np.copy(feature_data_nd_array)
    number_of_features = np.shape(updated_feature_nd_array)[1]
    for feature_array_index in range(number_of_features):
        updated_feature_nd_array[:, feature_array_index] = (updated_feature_nd_array[:, feature_array_index] - np.mean(
            updated_feature_nd_array[:, feature_array_index])) / np.std(
            updated_feature_nd_array[:, feature_array_index])
    return updated_feature_nd_array


# Process the data:

all_data = arff.loadarff('chronic_kidney_disease_full.arff')
dataframe_for_all_data = pd.DataFrame(all_data[0])
chronic_kidney_disease_data_columns_using_byte_strings = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe',
                                                          'ane', 'class']
feature_name_list = []

# Pandas dataframe uses byte strings to represent strings by default. These byte strings need to be decoded into regular strings for the applicable columns:

for data_index in range(len(chronic_kidney_disease_data_columns_using_byte_strings)):
    dataframe_for_all_data[chronic_kidney_disease_data_columns_using_byte_strings[data_index]] = dataframe_for_all_data[
        chronic_kidney_disease_data_columns_using_byte_strings[data_index]].str.decode("utf-8")

# 1. Substitute non-numeric string values for numeric values in dataframe:
# 2. Substitute non-numeric nominal integers and floats for numeric values in dataframe:
# 3. Get a list of feature names from the data:

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

    feature_name_list.append(feature_name)

# Remove 'class' feature from feature_name_list:
feature_name_list.remove(feature_name_list[len(feature_name_list) - 1])

# Substitute NAN (forced from question marks via errors=coerce argument) for numeric averages in dataframe:

for (feature_name, feature_data) in dataframe_for_all_data.items():
    dataframe_for_all_data[feature_name].fillna(value=dataframe_for_all_data[feature_name].mean(), inplace=True)

# Split the data into 80% training data and 20% testing data:

dataframe_for_all_data_copy = dataframe_for_all_data.copy()
feature_dataframe = dataframe_for_all_data_copy.loc[:, feature_name_list]
target_dataframe = dataframe_for_all_data_copy.loc[:, ['class']]

training_data_x, testing_data_x, training_data_y, testing_data_y = train_test_split(feature_dataframe, target_dataframe,
                                                                                    train_size=0.8, shuffle=True, stratify=target_dataframe)

# Convert training and testing data to Numpy arrays:

training_data_x_nd_array = training_data_x.to_numpy()
testing_data_y_nd_array = testing_data_y.to_numpy()
training_data_y_nd_array = training_data_y.to_numpy()
testing_data_x_nd_array = testing_data_x.to_numpy()

standardized_training_data_x_nd_array = apply_standardization_to_data(training_data_x_nd_array)
standardized_testing_data_x_nd_array = apply_standardization_to_data(testing_data_x_nd_array)

# Run the logistic regression algorithm for a range of -2 to 4 with a step of 0.2:
learning_rate = 0.001
show_scatter_plot = False
regularization_parameter_list = [-2, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
                                 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
for regularization_index in range(len(regularization_parameter_list)):

    lra1 = LogisticRegressionAlgorithm(learning_rate, 1000, regularization_parameter_list[regularization_index],
                                       False)
    # Run on training without standardization
    lra1.run_algorithm(training_data_x_nd_array, training_data_y_nd_array, True, show_scatter_plot)
    # Run on testing without standardization
    lra1.run_algorithm(testing_data_x_nd_array, testing_data_y_nd_array, False, show_scatter_plot)

    lra2 = LogisticRegressionAlgorithm(learning_rate, 1000, regularization_parameter_list[regularization_index],
                                       True)
    # Run on training with standardization
    lra2.run_algorithm(standardized_training_data_x_nd_array, training_data_y_nd_array, True, show_scatter_plot)

    if regularization_parameter_list[regularization_index] == 4:
        show_scatter_plot = True

    # Run on testing with standardization
    lra2.run_algorithm(standardized_testing_data_x_nd_array, testing_data_y_nd_array, False, show_scatter_plot)

    # Keep the learning rate constant at the end of each regularization parameter usage. In practice, it is best to decrease the learning rate such that the learning algorithm does not overstep the bottom of the gradient. In this case, keeping the learning rate constant is fine. Keeping the learning rate constant ensures that the cost calculation never gets too large (infinity or NAN).
