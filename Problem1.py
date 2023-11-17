# Problem 1 - Parts a and b combined

# K-means clustering is an unsupervised machine learning algorithm that takes an unlabeled dataset and groups it into different clusters.
# I referenced the following resource to help me better understand the steps involved in the k-means algorithm: https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/?
# The steps go as follows:
# 1. Set the number of clusters that will be generated
# 2. Randomly select the centroids for the dataset
# 3. Assign each data point to the closest cluster centroid
# 4. Recalculate the centroids of each cluster
# 5. Repeat #3 and #4 (assign all data points to the closest cluster centroid and recalculate the centroids of each cluster)

# The k-means algorithm stops execution when all centroids stay unchanged for two successive iterations.

import pandas as pd  # pandas is used for data manipulation and analysis.
import numpy as np  # numpy is used to perform a wide variety of mathematical operations on arrays.
import random
import matplotlib.pyplot as plt  # matplotlib is used for creating graphs, which is required in part b


class KMeansClusteringAlgorithm:

    def __init__(self, x_coordinate_list, y_coordinate_list, cluster_number):
        self.x_coordinate_list = x_coordinate_list
        self.y_coordinate_list = y_coordinate_list
        self.cluster_number = cluster_number  # Set the number of clusters that will be generated
        self.total_change_in_centroid_positions = 1

    def randomly_select_x_and_y_centroids(self):
        centroid_list_x = []
        centroid_list_y = []
        x_coordinate_list_copy = self.x_coordinate_list.copy()
        y_coordinate_list_copy = self.y_coordinate_list.copy()
        for i1 in range(self.cluster_number):
            random_index = random.randint(0, len(x_coordinate_list_copy) - 1)
            centroid_list_x.append(x_coordinate_list_copy[random_index])
            centroid_list_y.append(y_coordinate_list_copy[random_index])
            x_coordinate_list_copy.remove(x_coordinate_list_copy[random_index])
            y_coordinate_list_copy.remove(y_coordinate_list_copy[random_index])
        return centroid_list_x, centroid_list_y

    def generate_plot(self, centroid_list_x, centroid_list_y):
        plt.scatter(self.x_coordinate_list, self.y_coordinate_list, c='blue')  # coordinates are denoted as blue dots
        plt.scatter(centroid_list_x, centroid_list_y, c='red')  # centroids are denoted as red dots
        plt.xlabel('Length')
        plt.ylabel('Width')
        plt.show()

    def assign_all_data_points_to_a_cluster(self):
        self

    def recalculate_centroids_of_every_cluster(self):
        self

    def run_algorithm(self):
        centroid_list_x, centroid_list_y = self.randomly_select_x_and_y_centroids()
        self.generate_plot(centroid_list_x, centroid_list_y)


row_number, x_coordinates, y_coordinates = np.loadtxt("cluster_data.txt", delimiter=None, unpack=True)
k_means_user_1 = KMeansClusteringAlgorithm(x_coordinates.tolist(), y_coordinates.tolist(), 3)
k_means_user_1.run_algorithm()
