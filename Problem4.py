# Load, process the data, and create training and testing samples:
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split

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

# May need to adjust numpy arrays.

# Part a - Support Vector Machine with the linear kernel and default parameters:

# Part b - Support Vector Machine with the RBF kernel and default parameters:

# Part c - Random Forest with Default Parameters:
