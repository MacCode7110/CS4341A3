from sklearn.datasets import load_digits
import numpy as np
from sklearn.cluster import KMeans


# Part a

# Assumption for K-Means Clustering: K = 10

class Cluster:
    def __init__(self, cluster_number, actual_label):
        self.cluster_number = cluster_number
        self.actual_label = actual_label
        self.predicted_label = None
        self.digit_list = []

    def assign_predicted_label(self, predicted_label):
        self.predicted_label = predicted_label

    def append_digit_to_digit_list(self, digit):
        self.digit_list.append(digit)


digit_data, actual_labels = load_digits(return_X_y=True)
(number_of_samples, number_of_features), number_of_digits = digit_data.shape, np.unique(actual_labels).size

# K-Means Algorithm using sklearn:
number_of_clusters = 10
kmeans = KMeans(n_clusters=number_of_clusters, init='random',
                n_init=10, max_iter=300, random_state=0)

# Perform the kmeans clustering through the fit method (implementation):
kmeans_results = kmeans.fit(digit_data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Cluster labels are assigned AFTER clustering is done for the current algorithm.
# i. Assigning cluster labels based on the clustered data above. Each cluster is defined by the digit that represents the majority of the current cluster:
cluster_list = []
predicted_k_means_cluster_labels = []

for cluster_index in range(number_of_clusters):
    cluster_list.append(Cluster(cluster_index, None))
    obtained_digits = digit_data[np.where(labels == cluster_index)]

