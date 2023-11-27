# Assumption for K-Means Clustering: K = 10

from sklearn.datasets import load_digits
import numpy as np
from sklearn.cluster import KMeans

digit_data, labels = load_digits(return_X_y=True)
(number_of_samples, number_of_features), number_of_digits = digit_data.shape, np.unique(labels).size

# K-Means Algorithm using sklearn:

kmeans = KMeans(n_clusters=10, init='random',
    n_init=10, max_iter=300, random_state=0)

# Compute cluster centers and predict cluster index for each sample.
# fir predict returns index of the cluster each sample belongs to.
kmeans_label_predictions = kmeans.fit_predict(digit_data)
print(kmeans_label_predictions)

