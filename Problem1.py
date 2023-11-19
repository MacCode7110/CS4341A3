# Problem 1 - Parts a and b combined

# K-means clustering is an unsupervised machine learning algorithm that takes an unlabeled dataset and groups it into different clusters.
# I referenced the following s to help me better understand the steps involved in the k-means algorithm:
# https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/?
# https://www.jeremyjordan.me/grouping-data-points-with-k-means-clustering/
# The steps go as follows:
# 1. Set the number of clusters that will be generated
# 2. Randomly select the centroids for the dataset
# 3. Assign each data point to the closest cluster centroid
# 4. Recalculate the centroids of each cluster
# 5. Repeat #3 and #4 (assign all data points to the closest cluster centroid and recalculate the centroids of each cluster)

# The k-means algorithm stops execution when the change in centroid positions remains unchanged. This is determined by computing the sum of all differences in centroid positions after each time all the centroid positions are recalculated.
# For step 3, the closest cluster centroid for each vector is computed using the euclidean distance. A vector is assigned to the cluster that it is the lowest euclidean distance away from compared to all other clusters.
# For step 4, the centroids of each cluster are recalculated as follows; the mean data point of each cluster becomes the new centroid of that cluster. Given that we are working with a scatter plot, we compute the mean value of a cluster just as we would the mean value of a scatter plot or subset of a scatter plot. The mean data point takes the form of (mean value of x-coordinates in current cluster, mean value of y-coordinates in current cluster).

# Each cluster contains a centroid and points assigned to that cluster.
# All points assigned to a cluster have a distinct, randomly chosen shape, and a distinct, unifying color that is also randomly chosen.
# The centroid in a cluster has a distinct color but the same shape as the points assigned to that cluster.

import statistics as st  # Perform statistical operations in data.
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

    def generate_scatter_plot(self, cluster_object_list):
        cluster_sample_point_list = []
        for i1 in range(len(cluster_object_list)):
            cluster_coordinate_assignees_color = ColorRecord.get_available_color()
            cluster_centroid_color = ColorRecord.get_available_color()
            cluster_coordinate_shape = ShapeRecord.get_available_shape()
            for j1 in range(len(cluster_object_list[i1].x_coordinate_assignees)):
                point = plt.scatter(cluster_object_list[i1].x_coordinate_assignees[j1], cluster_object_list[i1].y_coordinate_assignees[j1], c=cluster_coordinate_assignees_color, marker=cluster_coordinate_shape)
                if j1 == 0:
                    cluster_sample_point_list.append(point)
            plt.scatter(cluster_object_list[i1].centroid_x, cluster_object_list[i1].centroid_y, c=cluster_centroid_color, marker=cluster_coordinate_shape)
        plt.legend(cluster_sample_point_list, ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"], loc="lower right")
        plt.xlabel('Length')
        plt.ylabel('Width')
        plt.show()

    def create_cluster_objects_and_assign_all_data_points_to_closest_cluster(self, centroid_list_x, centroid_list_y):
        cluster_object_list = []
        euclidean_distances_from_centroids_list_for_current_point = []

        for i1 in range(len(self.x_coordinate_list)):
            closest_centroid_x_and_closest_centroid_y_have_been_found_in_cluster_list_for_current_x_and_y_coordinates = False
            euclidean_distances_from_centroids_list_for_current_point.clear()
            x_coordinate_placeholder_list = []
            y_coordinate_placeholder_list = []

            for j1 in range(len(centroid_list_x)):
                euclidean_distances_from_centroids_list_for_current_point.append(np.sqrt(np.square((self.x_coordinate_list[i1] - centroid_list_x[j1])) + np.square((self.y_coordinate_list[i1] - centroid_list_y[j1]))))
            index_of_minimum_euclidean_distance_for_current_point = euclidean_distances_from_centroids_list_for_current_point.index(min(euclidean_distances_from_centroids_list_for_current_point))
            closest_centroid_x_coordinate = centroid_list_x[index_of_minimum_euclidean_distance_for_current_point]
            closest_centroid_y_coordinate = centroid_list_y[index_of_minimum_euclidean_distance_for_current_point]

            if len(cluster_object_list) == 0:
                x_coordinate_placeholder_list.clear()
                y_coordinate_placeholder_list.clear()
                x_coordinate_placeholder_list.append(self.x_coordinate_list[i1])
                y_coordinate_placeholder_list.append(self.y_coordinate_list[i1])
                cluster_object_list.append(Cluster(x_coordinate_placeholder_list, y_coordinate_placeholder_list,
                closest_centroid_x_coordinate, closest_centroid_y_coordinate))
            else:
                for k1 in range(len(cluster_object_list)):
                    if closest_centroid_x_coordinate == cluster_object_list[k1].centroid_x and closest_centroid_y_coordinate == cluster_object_list[k1].centroid_y:
                        cluster_object_list[k1].append_x_coordinate_assignees(self.x_coordinate_list[i1])
                        cluster_object_list[k1].append_y_coordinate_assignees(self.y_coordinate_list[i1])
                        closest_centroid_x_and_closest_centroid_y_have_been_found_in_cluster_list_for_current_x_and_y_coordinates = True
                if not closest_centroid_x_and_closest_centroid_y_have_been_found_in_cluster_list_for_current_x_and_y_coordinates:
                    x_coordinate_placeholder_list.clear()
                    y_coordinate_placeholder_list.clear()
                    x_coordinate_placeholder_list.append(self.x_coordinate_list[i1])
                    y_coordinate_placeholder_list.append(self.y_coordinate_list[i1])
                    cluster_object_list.append(Cluster(x_coordinate_placeholder_list, y_coordinate_placeholder_list,
                    closest_centroid_x_coordinate, closest_centroid_y_coordinate))

        return cluster_object_list

    def recalculate_centroids_of_every_cluster(self, cluster_object_list):
        new_centroid_list_x = []
        new_centroid_list_y = []
        for i1 in range(len(cluster_object_list)):
            centroid_x_placeholder_list = []
            centroid_y_placeholder_list = []
            centroid_x_placeholder_list.append(cluster_object_list[i1].centroid_x)
            centroid_y_placeholder_list.append(cluster_object_list[i1].centroid_y)
            all_x_coordinates_for_current_cluster = cluster_object_list[i1].x_coordinate_assignees + centroid_x_placeholder_list
            all_y_coordinates_for_current_cluster = cluster_object_list[i1].y_coordinate_assignees + centroid_y_placeholder_list
            mean_x_coordinate = st.mean(all_x_coordinates_for_current_cluster)
            mean_y_coordinate = st.mean(all_y_coordinates_for_current_cluster)
            new_centroid_list_x.append(mean_x_coordinate)
            new_centroid_list_y.append(mean_y_coordinate)
        return new_centroid_list_x, new_centroid_list_y

    def update_total_change_in_centroid_positions(self, old_centroid_list_x, old_centroid_list_y, updated_centroid_list_x, updated_centroid_list_y):
        sum_of_differences_in_centroid_positions = sum([a - b for a, b in zip(updated_centroid_list_x, old_centroid_list_x)]) + sum([a - b for a, b in zip(updated_centroid_list_y, old_centroid_list_y)])
        self.total_change_in_centroid_positions = sum_of_differences_in_centroid_positions

    def run_algorithm(self):
        centroid_list_x, centroid_list_y = self.randomly_select_x_and_y_centroids()
        while self.total_change_in_centroid_positions != 0:
            cluster_object_list = self.create_cluster_objects_and_assign_all_data_points_to_closest_cluster(
                centroid_list_x, centroid_list_y)
            new_centroid_list_x, new_centroid_list_y = self.recalculate_centroids_of_every_cluster(cluster_object_list)
            self.update_total_change_in_centroid_positions(centroid_list_x, centroid_list_y, new_centroid_list_x, new_centroid_list_y)
            centroid_list_x = new_centroid_list_x.copy()
            centroid_list_y = new_centroid_list_y.copy()
        self.generate_scatter_plot(cluster_object_list)


class ColorRecord:

    color_list = ["#d14338", "#e3c51e", "#b0e610", "#64eb10", "#667869", "#58f5db", "#0095ff", "#6600ff"]

    @classmethod
    def get_available_color(cls):
        random_index = random.randint(0, len(cls.color_list) - 1)
        available_color = cls.color_list[random_index]
        cls.update_available_colors(random_index)
        return available_color

    @classmethod
    def update_available_colors(cls, index):
        ColorRecord.color_list.remove(ColorRecord.color_list[index])


class ShapeRecord:

    shape_list = ["v", "o", "x", "s"]

    @classmethod
    def get_available_shape(cls):
        random_index = random.randint(0, len(cls.shape_list) - 1)
        available_shape = cls.shape_list[random_index]
        cls.update_available_shapes(random_index)
        return available_shape

    @classmethod
    def update_available_shapes(cls, index):
        cls.shape_list.remove(cls.shape_list[index])


class Cluster:
    def __init__(self, x_coordinate_assignees, y_coordinate_assignees, centroid_x, centroid_y):
        self.x_coordinate_assignees = x_coordinate_assignees
        self.y_coordinate_assignees = y_coordinate_assignees
        self.centroid_x = centroid_x
        self.centroid_y = centroid_y

    def append_x_coordinate_assignees(self, new_x_coordinate):
        self.x_coordinate_assignees.append(new_x_coordinate)

    def append_y_coordinate_assignees(self, new_y_coordinate):
        self.y_coordinate_assignees.append(new_y_coordinate)


row_number, x_coordinates, y_coordinates = np.loadtxt("cluster_data.txt", delimiter=None, unpack=True)
k_means_user_1 = KMeansClusteringAlgorithm(x_coordinates.tolist(), y_coordinates.tolist(), 4)
k_means_user_1.run_algorithm()
