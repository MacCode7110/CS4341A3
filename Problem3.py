from sklearn.datasets import load_digits
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics


# Classes and helpers
class Cluster:
    def __init__(self, cluster_identifier, digit_matrix, predicted_label):
        self.cluster_identifier = cluster_identifier
        self.digit_matrix = digit_matrix
        self.predicted_label = predicted_label


# Load the digit dataset

digit_data, actual_labels = load_digits(return_X_y=True)
(number_of_samples, number_of_features), number_of_digits = digit_data.shape, np.unique(actual_labels).size

# Structure of confusion matrix (10 X 10):
#                      Predicted Label
#                   | | | | | | | | | | |
#                   ---------------------
# Actual Label      . . . . . . . . . . .
#                   ---------------------
#                   | | | | | | | | | | |

# Part a

print("KMeans Clustering Results:\n")

# Assumption for K-Means Clustering: K = 10

# K-Means Algorithm using sklearn:
number_of_kmeans_clusters = 10
kmeans = KMeans(n_clusters=number_of_kmeans_clusters, init='random',
                n_init=10, max_iter=300, random_state=0)

# Perform the kmeans clustering through the fit method (implementation):
kmeans_results = kmeans.fit(digit_data)
labels_produced_by_clustering = kmeans.labels_
centroids = kmeans.cluster_centers_

# Cluster labels are assigned AFTER clustering is done for the current algorithm.
# i. Assigning cluster labels based on the clustered data above. Each cluster is defined by the digit that represents the majority of the current cluster:
cluster_list = []
kmeans_predicted_labels_per_sample = []

for cluster_index in range(number_of_kmeans_clusters):
    obtained_digits_for_current_cluster_index = digit_data[
        np.where(labels_produced_by_clustering == cluster_index)]  # Returns 2D numpy array
    flattened_obtained_digits_for_current_cluster_index = obtained_digits_for_current_cluster_index.flatten()
    unique_digits, counts = np.unique(flattened_obtained_digits_for_current_cluster_index, return_counts=True)
    most_frequently_occurring_digit_for_current_cluster = unique_digits[counts.argmax()]
    cluster_list.append(Cluster(cluster_index, obtained_digits_for_current_cluster_index,
                                most_frequently_occurring_digit_for_current_cluster))

# Gather the cluster predictions per sample into one list to prepare for confusion matrix creation:

for cluster_list_index in range(len(cluster_list)):
    for row_index in range(len(cluster_list[cluster_list_index].digit_matrix)):
        kmeans_predicted_labels_per_sample.append(cluster_list[cluster_list_index].predicted_label)

# ii. Computing the confusion matrix:
print("Number of samples in actual labels for KMeans Clustering: " + str(len(actual_labels)))
print("Number of samples in predicted labels for KMeans Clustering: " + str(len(kmeans_predicted_labels_per_sample)) + "\n")

kmeans_confusion_matrix = metrics.confusion_matrix(actual_labels, kmeans_predicted_labels_per_sample)

# Determine the total number of unique predicted clusters and the clusters that were not predicted at all. Adjust the confusion matrix accordingly:
kmeans_unique_predicted_digits = []
kmeans_not_predicted_digits = []

for digit in kmeans_predicted_labels_per_sample:
    if digit not in kmeans_unique_predicted_digits:
        kmeans_unique_predicted_digits.append(digit)

if len(kmeans_unique_predicted_digits) < 10:
    for cluster_number in range(0, number_of_kmeans_clusters):
        if cluster_number not in kmeans_unique_predicted_digits:
            kmeans_not_predicted_digits.append(cluster_number)

    # Here, if an actual label does not exist in the predicted labels, then we denote that the actual label was not predicted by setting
    # the position in the confusion matrix corresponding to the same actual label and predicted label values to -1.
    for row_label in range(len(kmeans_confusion_matrix)):
        for col_label in range(len(kmeans_confusion_matrix[row_label])):
            if (row_label in kmeans_not_predicted_digits) and row_label == col_label:
                kmeans_confusion_matrix[row_label][col_label] = -1


print("10 X 10 confusion matrix produced for KMeans Clustering:")
print(kmeans_confusion_matrix)

# iii. Computing the Fowlkes-Mallows index for the actual labels and predicted labels:
print("\nHere is the Fowlkes Mallows Score for KMeans Clustering: " + str(metrics.fowlkes_mallows_score(actual_labels, kmeans_predicted_labels_per_sample)))

# Part b

print("\nAgglomerative Clustering with Ward Linkage Results:\n")

