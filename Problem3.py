# Part a

# Assumption for K-Means Clustering: K = 10

from sklearn.datasets import load_digits
import numpy as np
from sklearn.cluster import KMeans

digit_data, actual_labels = load_digits(return_X_y=True)
(number_of_samples, number_of_features), number_of_digits = digit_data.shape, np.unique(actual_labels).size

# K-Means Algorithm using sklearn:

kmeans = KMeans(n_clusters=10, init='random',
    n_init=10, max_iter=300, random_state=0)

# Compute cluster centers and predict cluster index for each sample.
# fit_predict returns index of the cluster each sample belongs to.
kmeans_results = kmeans.fit(digit_data)

# Cluster labels are assigned AFTER clustering is done for each algorithm.
# i. Assigning cluster labels based on the clustered data above. Each cluster is defined by the digit that represents the majority of the current cluster:


