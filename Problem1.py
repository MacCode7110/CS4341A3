# Problem 1 - Parts a and b combined

# K-means clustering is an unsupervised machine learning algorithm that takes an unlabeled dataset and groups it into different clusters.
# I referenced the following resource to help me better understand the steps involved in the k-means algorithm: https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/?
# The steps go as follows:
# 1. Set the number of clusters that will be generated
# 2. Randomly select the centroids for the dataset
# 3. Assign each data point to the closest cluster centroid
# 4. Recalculate the centroids of each cluster
# 5. Repeat #3 and #4 (assign all data points to the closest cluster centroid and recalculate the centroids of each cluster)

# The k-means algorithm stops execution when the total change in centroid positions remains unchanged.
# Here, the closest cluster centroid for each vector is computed using the euclidean distance.

# Each cluster contains a centroid and points assigned to that cluster.
# All points assigned to a cluster have a distinct, randomly chosen shape, and a distinct, unifying color that is also randomly chosen.
# The centroid in a cluster has a distinct color and the same shape as the points assigned to that cluster.

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
        plt.scatter(self.x_coordinate_list, self.y_coordinate_list, c='blue')
        plt.scatter(centroid_list_x, centroid_list_y, c='red')
        plt.xlabel('Length')
        plt.ylabel('Width')
        plt.show()

    def assign_all_data_points_to_closest_cluster_and_create_cluster_objects(self, centroid_list_x, centroid_list_y):
        cluster_object_list = []
        for i1 in range(len(self.x_coordinate_list)):
            euclidean_distance_from_centroid = 0
            shortest_euclidean_distance_from_a_centroid = 0
            for j1 in range(len(centroid_list_x)):
                euclidean_distance_from_centroid = np.sqrt(np.square(1))
        return cluster_object_list

    def recalculate_centroids_of_every_cluster(self, cluster_object_list, centroid_list_x, centroid_list_y):
        self

    def run_algorithm(self):
        centroid_list_x, centroid_list_y = self.randomly_select_x_and_y_centroids()
        while self.total_change_in_centroid_positions != 0:
            cluster_object_list = self.assign_all_data_points_to_closest_cluster_and_create_cluster_objects(
                centroid_list_x, centroid_list_y)
            self.recalculate_centroids_of_every_cluster(cluster_object_list, centroid_list_x, centroid_list_y)
        # self.generate_plot(centroid_list_x, centroid_list_y)


class ColorRecord:
    def __init__(self):
        self.color_list = ["#d14338", "#e3c51e", "#b0e610", "#64eb10", "#667869", "#58f5db", "#0095ff", "#6600ff"]

    def get_available_color(self):
        random_index = random.randint(0, len(self.color_list) - 1)
        available_color = self.color_list[random_index]
        self.update_available_colors(random_index)
        return available_color

    def update_available_colors(self, index):
        self.color_list.remove(self.color_list[index])


class ShapeRecord:
    def __init__(self):
        self.shape_list = ["v", "o", "x", "s"]

    def get_available_shape(self):
        random_index = random.randint(0, len(self.shape_list) - 1)
        available_shape = self.shape_list[random_index]
        self.update_available_shapes(random_index)
        return available_shape

    def update_available_shapes(self, index):
        self.shape_list.remove(self.shape_list[index])


class Cluster:
    def __init__(self, x_coordinate_assignees, y_coordinate_assignees, centroid_x, centroid_y, coordinate_shape,
                 assigned_coordinate_color, centroid_color):
        self.x_coordinate_assignees = x_coordinate_assignees
        self.y_coordinate_assignees = y_coordinate_assignees
        self.centroid_x = centroid_x
        self.centroid_y = centroid_y
        self.coordinate_shape = coordinate_shape
        self.assigned_coordinate_color = assigned_coordinate_color
        self.centroid_color = centroid_color


row_number, x_coordinates, y_coordinates = np.loadtxt("cluster_data.txt", delimiter=None, unpack=True)
k_means_user_1 = KMeansClusteringAlgorithm(x_coordinates.tolist(), y_coordinates.tolist(), 4)
k_means_user_1.run_algorithm()
