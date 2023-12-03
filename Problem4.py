# Load, process the data, and create training and testing samples:
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

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

# Question for OH 12/4/23:
# I am seeing this in Support Vector Machine documentation for predict function:
#   X{array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples_test, n_samples_train)
#   For kernel=”precomputed”, the expected shape of X is (n_samples_test, n_samples_train).
#   When testing the testing and training datasets, do we need to combine both datasets for x sets into a single 2D array? (This single 2D array would be the input to the predict function)

# Part a - Support Vector Machine with the linear kernel and default parameters:

# i. Training the model below using the training datasets and test the data using the predict method to make predictions:

# Create an SVC:
support_vector_machine_linear_kernel = svm.SVC(kernel='linear')
support_vector_machine_linear_kernel.fit(training_data_x_nd_array, training_data_y_nd_array.flatten())
svm_linear_kernel_predicted_class_labels_from_training = support_vector_machine_linear_kernel.predict(training_data_x_nd_array)
svm_linear_kernel_predicted_class_labels_from_testing = support_vector_machine_linear_kernel.predict(testing_data_x_nd_array)

# ii. Calculate the f1 scores:

f1_score_for_training_data_svm_linear_kernel = f1_score(training_data_y_nd_array, svm_linear_kernel_predicted_class_labels_from_training)
f1_score_for_testing_data_svm_linear_kernel = f1_score(testing_data_y_nd_array, svm_linear_kernel_predicted_class_labels_from_testing)


print("F-Measure for Support Vector Machine linear kernel tested with training data: " + str(f1_score_for_training_data_svm_linear_kernel))
print("F-Measure for Support Vector Machine linear kernel tested with testing data: " + str(f1_score_for_testing_data_svm_linear_kernel) + "\n")


# Part b - Support Vector Machine with the RBF kernel and default parameters:

# i. Training the model below using the training datasets and test the data using the predict method to make predictions:
# Create an SVC:

support_vector_machine_RBF_kernel = svm.SVC(kernel='rbf')
support_vector_machine_RBF_kernel.fit(training_data_x_nd_array, training_data_y_nd_array.flatten())
svm_RBF_kernel_predicted_class_labels_from_training = support_vector_machine_RBF_kernel.predict(training_data_x_nd_array)
svm_RBF_kernel_predicted_class_labels_from_testing = support_vector_machine_RBF_kernel.predict(testing_data_x_nd_array)

# ii. Calculate the f1 scores:

f1_score_for_training_data_svm_RBF_kernel = f1_score(training_data_y_nd_array, svm_RBF_kernel_predicted_class_labels_from_training)
f1_score_for_testing_data_svm_RBF_kernel = f1_score(testing_data_y_nd_array, svm_RBF_kernel_predicted_class_labels_from_testing)

print("F-Measure for Support Vector Machine RBF kernel tested with training data: " + str(f1_score_for_training_data_svm_RBF_kernel))
print("F-Measure for Support Vector Machine RBF kernel tested with testing data: " + str(f1_score_for_testing_data_svm_RBF_kernel) + "\n")


# Part c - Random Forest with Default Parameters:

# i. Training the model below using the training datasets and test the data using the predict method to make predictions:

random_forest = RandomForestClassifier()
random_forest.fit(training_data_x_nd_array, training_data_y_nd_array.flatten())
random_forest_predicted_class_labels_from_training = random_forest.predict(training_data_x_nd_array)
random_forest_predicted_class_labels_from_testing = random_forest.predict(testing_data_x_nd_array)

# ii. Calculate the f1 scores:

f1_score_for_training_data_random_forest = f1_score(training_data_y_nd_array, random_forest_predicted_class_labels_from_training)
f1_score_for_testing_data_random_forest = f1_score(testing_data_y_nd_array, random_forest_predicted_class_labels_from_testing)

print("F-Measure for Random Forest tested with training data: " + str(f1_score_for_training_data_random_forest))
print("F-Measure for Random Forest tested with testing data: " + str(f1_score_for_testing_data_random_forest))

