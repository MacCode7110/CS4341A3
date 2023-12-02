from sklearn.datasets import load_digits
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn import metrics


# Helper functions to carry out all three steps in the protocol for each clustering algorithm:

def find_majority_digit_in_cluster(current_cluster_indices, ground_truth_labels):
    # Strategy for determining the current cluster: We already have the current cluster indices from the assign cluster labels function, which all correspond to the cluster that matches the current cluster label/index (in class range 0-9). Here, we map the current cluster indices back to the ground truth labels to find the samples that got assigned to the current cluster.
    cluster = ground_truth_labels[current_cluster_indices]
    unique_digits, counts = np.unique(cluster, return_counts=True)
    most_frequently_occurring_digit_for_current_cluster = unique_digits[counts.argmax()]
    return most_frequently_occurring_digit_for_current_cluster


def assign_cluster_labels_based_on_majority_digit(labels_produced_by_clustering, number_of_clusters, ground_truth_labels):
    predicted_labels_per_sample = np.zeros(np.shape(labels_produced_by_clustering))  # Here, we ensure that the predicted labels are calculated per sample by having the data structure take the shape of the labels produced by clustering.
    for cluster_index in range(number_of_clusters):
        # Obtain the indices of the cluster that matches the current cluster index/label:
        current_cluster_indices = np.where(labels_produced_by_clustering == cluster_index)
        predicted_labels_per_sample[current_cluster_indices] = find_majority_digit_in_cluster(current_cluster_indices, ground_truth_labels)
    return predicted_labels_per_sample


def compute_confusion_matrix(ground_truth_labels, predicted_labels_per_sample):
    return metrics.confusion_matrix(ground_truth_labels, predicted_labels_per_sample)


def manipulate_confusion_matrix(number_of_clusters, predicted_labels, confusion_matrix):
    # Determine the total number of unique predicted clusters and the clusters that were not predicted at all. Adjust the confusion matrix accordingly:
    unique_predicted_digits = []
    digits_not_predicted = []
    confusion_matrix_copy = confusion_matrix.copy()

    for digit in predicted_labels:
        if digit not in unique_predicted_digits:
            unique_predicted_digits.append(digit)

    if len(unique_predicted_digits) < number_of_clusters:  # Number of clusters is always 10 for all assumptions.
        for cluster_number in range(0, number_of_clusters):
            if cluster_number not in unique_predicted_digits:
                digits_not_predicted.append(cluster_number)

    # Here, if an actual label does not exist in the predicted labels, then we denote that the actual label was not predicted by setting
    # the position in the confusion matrix corresponding to the same actual label and predicted label values to -1.
    for row_label in range(len(confusion_matrix_copy)):
        for col_label in range(len(confusion_matrix_copy[row_label])):
            if (row_label in digits_not_predicted) and row_label == col_label:
                confusion_matrix_copy[row_label][col_label] = -1

    return confusion_matrix_copy


def get_fowlkes_mallows_score(ground_truth_labels, predicted_labels_per_sample):
    return metrics.fowlkes_mallows_score(ground_truth_labels, predicted_labels_per_sample)


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
# Assumption for KMeans Clustering: K = 10

print("KMeans Clustering Results:\n")

# K-Means Algorithm using sklearn:
kmeans = KMeans(n_clusters=10, init='random',
                n_init=10, max_iter=300, random_state=5)

# Perform the kmeans clustering through the fit_predict method (implementation):
kmeans_resulting_labels = kmeans.fit_predict(digit_data)
kmeans_resulting_total_cluster_number = len(np.unique(kmeans_resulting_labels))

# Cluster labels are assigned AFTER clustering is done for the current algorithm.
# i. Assigning cluster labels based on the clustered data above. Each cluster is defined by the digit that represents the majority of the current cluster:
kmeans_predicted_labels = assign_cluster_labels_based_on_majority_digit(kmeans_resulting_labels, kmeans_resulting_total_cluster_number, actual_labels)

# ii. Computing the confusion matrix:
print("Number of samples in actual labels for KMeans Clustering: " + str(len(actual_labels)))
print("Number of samples in predicted labels for KMeans Clustering: " + str(
    len(kmeans_predicted_labels)) + "\n")

kmeans_confusion_matrix = compute_confusion_matrix(actual_labels, kmeans_predicted_labels)
updated_kmeans_confusion_matrix = manipulate_confusion_matrix(kmeans_resulting_total_cluster_number,
                                                              kmeans_predicted_labels,
                                                              kmeans_confusion_matrix)

print("10 X 10 confusion matrix produced for KMeans Clustering:")
print(updated_kmeans_confusion_matrix)

# iii. Computing the Fowlkes-Mallows index for the actual labels and predicted labels:
kmeans_clustering_fowlkes_mallows_score = get_fowlkes_mallows_score(actual_labels, kmeans_predicted_labels)
print("\nHere is the Fowlkes Mallows Score for KMeans Clustering: " + str(kmeans_clustering_fowlkes_mallows_score))

# Part b
# Assumption for Agglomerative Clustering: K = 10


print("\nAgglomerative Clustering with Ward Linkage Results:\n")


# Agglomerative Clustering algorithm implementation (through fit method) using sklearn:
agglomerative_clustering = AgglomerativeClustering(n_clusters=10, linkage='ward')
agglomerative_resulting_labels = agglomerative_clustering.fit_predict(digit_data)
agglomerative_resulting_total_cluster_number = len(np.unique(agglomerative_resulting_labels))


# Cluster labels are assigned AFTER clustering is done for the current algorithm.
# i. Assigning cluster labels based on the clustered data above. Each cluster is defined by the digit that represents the majority of the current cluster:
agglomerative_predicted_labels = assign_cluster_labels_based_on_majority_digit(agglomerative_resulting_labels, agglomerative_resulting_total_cluster_number, actual_labels)


# ii. Computing the confusion matrix:
print("Number of samples in actual labels for Agglomerative Clustering: " + str(len(actual_labels)))
print("Number of samples in predicted labels for Agglomerative Clustering: " + str(
   len(agglomerative_predicted_labels)) + "\n")


agglomerative_confusion_matrix = compute_confusion_matrix(actual_labels, agglomerative_predicted_labels)
updated_agglomerative_confusion_matrix = manipulate_confusion_matrix(agglomerative_resulting_total_cluster_number,
                                                                     agglomerative_predicted_labels,
                                                                     agglomerative_confusion_matrix)


print("10 X 10 confusion matrix produced for Agglomerative Clustering:")
print(updated_agglomerative_confusion_matrix)


# iii. Computing the Fowlkes-Mallows index for the actual labels and predicted labels:
agglomerative_clustering_fowlkes_mallows_score = get_fowlkes_mallows_score(actual_labels,
                                                                          agglomerative_predicted_labels)
print("\nHere is the Fowlkes Mallows Score for Agglomerative Clustering: " + str(
   agglomerative_clustering_fowlkes_mallows_score))


# Part c
# Assumption for Affinity Propagation: K = 10


print("\nAffinity Propagation Results:\n")


# Affinity Propagation algorithm implementation (through fit method) using sklearn:
affinity_propagation = AffinityPropagation(max_iter=200, convergence_iter=15, damping=0.5, copy=True, preference=None,
                                          affinity='euclidean', verbose=False, random_state=None)
affinity_propagation_resulting_labels = affinity_propagation.fit_predict(digit_data)
affinity_propagation_resulting_total_cluster_number = len(np.unique(affinity_propagation_resulting_labels))


# Cluster labels are assigned AFTER clustering is done for the current algorithm.
# i. Assigning cluster labels based on the clustered data above. Each cluster is defined by the digit that represents the majority of the current cluster:
affinity_propagation_predicted_labels = assign_cluster_labels_based_on_majority_digit(affinity_propagation_resulting_labels, affinity_propagation_resulting_total_cluster_number, actual_labels)


# ii. Computing the confusion matrix:
print("Number of samples in actual labels for Affinity Propagation: " + str(len(actual_labels)))
print("Number of samples in predicted labels for Affinity Propagation: " + str(
   len(affinity_propagation_predicted_labels)) + "\n")


affinity_propagation_confusion_matrix = compute_confusion_matrix(actual_labels, affinity_propagation_predicted_labels)
updated_affinity_propagation_confusion_matrix = manipulate_confusion_matrix(
    affinity_propagation_resulting_total_cluster_number,
    affinity_propagation_predicted_labels,
    affinity_propagation_confusion_matrix)


print("10 X 10 confusion matrix produced for Affinity Propagation:")
print(updated_affinity_propagation_confusion_matrix)


# iii. Computing the Fowlkes-Mallows index for the actual labels and predicted labels:
affinity_propagation_fowlkes_mallows_score = get_fowlkes_mallows_score(actual_labels,
                                                                      affinity_propagation_predicted_labels)
print("\nHere is the Fowlkes Mallows Score for Affinity Propagation: " + str(
   affinity_propagation_fowlkes_mallows_score))


