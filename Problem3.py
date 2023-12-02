from sklearn.datasets import load_digits
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn import metrics


# Classes and helper functions to carry out all three steps in the protocol for each clustering algorithm:
class Cluster:
    def __init__(self, cluster_identifier, digit_matrix, predicted_label):
        self.cluster_identifier = cluster_identifier
        self.digit_matrix = digit_matrix
        self.predicted_label = predicted_label


def assign_cluster_labels(labels_produced_by_clustering, number_of_clusters):
    cluster_list = []
    for cluster_index in range(number_of_clusters):
        obtained_digits_for_current_cluster_index = digit_data[
            np.where(labels_produced_by_clustering == cluster_index)]  # Returns 2D numpy array
        flattened_obtained_digits_for_current_cluster_index = obtained_digits_for_current_cluster_index.flatten()
        unique_digits, counts = np.unique(flattened_obtained_digits_for_current_cluster_index, return_counts=True)
        most_frequently_occurring_digit_for_current_cluster = unique_digits[counts.argmax()]
        cluster_list.append(Cluster(cluster_index, obtained_digits_for_current_cluster_index,
                                    most_frequently_occurring_digit_for_current_cluster))
    return cluster_list


def gather_cluster_predictions_per_sample(cluster_list):
    predicted_labels_per_sample = []
    for cluster_list_index in range(len(cluster_list)):
        for row_index in range(len(cluster_list[cluster_list_index].digit_matrix)):
            predicted_labels_per_sample.append(cluster_list[cluster_list_index].predicted_label)
    return predicted_labels_per_sample


def compute_confusion_matrix(ground_truth_labels, predicted_labels_per_sample):
    return metrics.confusion_matrix(ground_truth_labels, predicted_labels_per_sample)


def manipulate_confusion_matrix(number_of_clusters, predicted_labels_per_sample, confusion_matrix):
    # Determine the total number of unique predicted clusters and the clusters that were not predicted at all. Adjust the confusion matrix accordingly:
    unique_predicted_digits = []
    digits_not_predicted = []
    confusion_matrix_copy = confusion_matrix.copy()

    for digit in predicted_labels_per_sample:
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
number_of_kmeans_clusters = 10
kmeans = KMeans(n_clusters=number_of_kmeans_clusters, init='random',
                n_init=10, max_iter=300, random_state=5)

# Perform the kmeans clustering through the fit method (implementation):
kmeans_results = kmeans.fit(digit_data)
labels_produced_by_kmeans_clustering = kmeans.labels_
kmeans_centroids = kmeans.cluster_centers_

# Cluster labels are assigned AFTER clustering is done for the current algorithm.
# i. Assigning cluster labels based on the clustered data above. Each cluster is defined by the digit that represents the majority of the current cluster:
kmeans_cluster_list = assign_cluster_labels(labels_produced_by_kmeans_clustering, number_of_kmeans_clusters)

# Gather the cluster predictions per sample into one list to prepare for confusion matrix creation:
kmeans_predicted_labels_per_sample = gather_cluster_predictions_per_sample(kmeans_cluster_list)

# ii. Computing the confusion matrix:
print("Number of samples in actual labels for KMeans Clustering: " + str(len(actual_labels)))
print("Number of samples in predicted labels for KMeans Clustering: " + str(
    len(kmeans_predicted_labels_per_sample)) + "\n")

kmeans_confusion_matrix = compute_confusion_matrix(actual_labels, kmeans_predicted_labels_per_sample)
updated_kmeans_confusion_matrix = manipulate_confusion_matrix(number_of_kmeans_clusters,
                                                              kmeans_predicted_labels_per_sample,
                                                              kmeans_confusion_matrix)

print("10 X 10 confusion matrix produced for KMeans Clustering:")
print(updated_kmeans_confusion_matrix)

# iii. Computing the Fowlkes-Mallows index for the actual labels and predicted labels:
kmeans_clustering_fowlkes_mallows_score = get_fowlkes_mallows_score(actual_labels, kmeans_predicted_labels_per_sample)
print("\nHere is the Fowlkes Mallows Score for KMeans Clustering: " + str(kmeans_clustering_fowlkes_mallows_score))

# Part b
# Assumption for Agglomerative Clustering: K = 10

print("\nAgglomerative Clustering with Ward Linkage Results:\n")

# Agglomerative Clustering algorithm implementation (through fit method) using sklearn:
number_of_agglomerative_clusters = 10
agglomerative_clustering = AgglomerativeClustering(n_clusters=number_of_agglomerative_clusters, linkage='ward')
agglomerative_clustering_results = agglomerative_clustering.fit(digit_data)
labels_produced_by_agglomerative_clustering = agglomerative_clustering.labels_

# Cluster labels are assigned AFTER clustering is done for the current algorithm.
# i. Assigning cluster labels based on the clustered data above. Each cluster is defined by the digit that represents the majority of the current cluster:
agglomerative_cluster_list = assign_cluster_labels(labels_produced_by_agglomerative_clustering, number_of_agglomerative_clusters)

# Gather the cluster predictions per sample into one list to prepare for confusion matrix creation:
agglomerative_clustering_labels_per_sample = gather_cluster_predictions_per_sample(agglomerative_cluster_list)

# ii. Computing the confusion matrix:
print("Number of samples in actual labels for Agglomerative Clustering: " + str(len(actual_labels)))
print("Number of samples in predicted labels for Agglomerative Clustering: " + str(
    len(agglomerative_clustering_labels_per_sample)) + "\n")

agglomerative_confusion_matrix = compute_confusion_matrix(actual_labels, agglomerative_clustering_labels_per_sample)
updated_agglomerative_confusion_matrix = manipulate_confusion_matrix(number_of_agglomerative_clusters,
                                                                     agglomerative_clustering_labels_per_sample,
                                                                     agglomerative_confusion_matrix)

print("10 X 10 confusion matrix produced for Agglomerative Clustering:")
print(updated_agglomerative_confusion_matrix)

# iii. Computing the Fowlkes-Mallows index for the actual labels and predicted labels:
agglomerative_clustering_fowlkes_mallows_score = get_fowlkes_mallows_score(actual_labels,
                                                                           agglomerative_clustering_labels_per_sample)
print("\nHere is the Fowlkes Mallows Score for Agglomerative Clustering: " + str(
    agglomerative_clustering_fowlkes_mallows_score))

# Part c
# Assumption for Affinity Propagation: K = 10

print("\nAffinity Propagation Results:\n")

# Affinity Propagation algorithm implementation (through fit method) using sklearn:
desired_number_of_affinity_propagation_clusters = 10
affinity_propagation = AffinityPropagation(max_iter=200, convergence_iter=15, damping=0.5, copy=True, preference=None,
                                           affinity='euclidean', verbose=False, random_state=None)
affinity_propagation_results = affinity_propagation.fit(digit_data)
labels_produced_by_affinity_propagation = affinity_propagation.labels_

# Cluster labels are assigned AFTER clustering is done for the current algorithm.
# i. Assigning cluster labels based on the clustered data above. Each cluster is defined by the digit that represents the majority of the current cluster:
affinity_propagation_cluster_list = assign_cluster_labels(labels_produced_by_affinity_propagation, desired_number_of_affinity_propagation_clusters)

# Gather the cluster predictions per sample into one list to prepare for confusion matrix creation:
affinity_propagation_labels_per_sample = gather_cluster_predictions_per_sample(affinity_propagation_cluster_list)

# ii. Computing the confusion matrix:
print("Number of samples in actual labels for Affinity Propagation: " + str(len(actual_labels)))
print("Number of samples in predicted labels for Affinity Propagation: " + str(
    len(affinity_propagation_labels_per_sample)) + "\n")

affinity_propagation_confusion_matrix = compute_confusion_matrix(actual_labels, affinity_propagation_labels_per_sample)
updated_affinity_propagation_confusion_matrix = manipulate_confusion_matrix(
    desired_number_of_affinity_propagation_clusters,
    affinity_propagation_labels_per_sample,
    affinity_propagation_confusion_matrix)

print("10 X 10 confusion matrix produced for Affinity Propagation:")
print(updated_affinity_propagation_confusion_matrix)

# iii. Computing the Fowlkes-Mallows index for the actual labels and predicted labels:
affinity_propagation_fowlkes_mallows_score = get_fowlkes_mallows_score(actual_labels,
                                                                       affinity_propagation_labels_per_sample)
print("\nHere is the Fowlkes Mallows Score for Affinity Propagation: " + str(
    affinity_propagation_fowlkes_mallows_score))

# Fixes:
# When taking majority digit for each cluster, if a digit is already selected as the majority digit of a previous cluster, need to take the next highest majority digit for the current cluster.
# Affinity Propagation:
# The samples with insanely high labels (outside the class range 0-9) are most likely outliers. Put these outliers into a different bin (bin 10), and exclude them from the computation of the confusion matrix. Only use samples assigned bins/classes 0-9 to calculate the confusion matrix.
