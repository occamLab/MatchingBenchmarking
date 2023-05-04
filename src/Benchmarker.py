import os
import pickle
import cv2
from scipy.linalg import inv
import numpy as np


class Benchmarker:
    """
    Benchmarker class for comparing the performance of different matching
    algorithms. This class contains general methods for loading the Session objects
    from a pickle file, and depth map projection and plotting to be used in
    different types benchmarkers in the benchmarkers directory.

    Instance Attributes:
        sessions (list): A list of Session objects.
        Algorithms (list): MatchingAlgorithm objects to be benchmarked.
        sweep_values (list): Quantile values to be used in the ratio test for
            the OpenCV matching algorithms.
    """

    def __init__(self, algorithms, values):
        self.sessions = self.get_sessions()
        self.algorithms = algorithms
        self.sweep_values = values

    def get_sessions(self):
        """
        Loads the Session objects from a pickle file.

        Returns: A list of Session objects.
        """
        sessions_path = (
            f"{os.path.dirname(os.path.dirname(__file__))}/session_data/sessions.pkl"
        )
        with open(sessions_path, "rb") as sessions_file:
            sessions_data = pickle.load(sessions_file)
        return sessions_data.sessions

    def project_depth_onto_image(self, query_depth_feature_points, focal_length, offset_x, offset_y):
        """
        Projects depth map onto query image using camera intrinsics.

        Args:
            query_depth_feature_points (numpy.ndarray): A numpy array 
                containing the depth map's feature points.
            focal_length (float): The focal length of the camera.
            offset_x (float): The x offset of the camera.
            offset_y (float): The y offset of the camera.

        Returns: A numpy array containing the projected feature points.
        """
        pixels = []
        for row in query_depth_feature_points:
            pixel_x = row[0] * focal_length / row[2] + offset_x
            pixel_y = row[1] * focal_length / row[2] + offset_y
            pixels.append((pixel_x, pixel_y))
        return pixels

    def plot_depth_map(self, query_image_depth_map, pixels, image):
        """
        Plots the depth map onto the query image as black and white dots
        varying by depth.

        Args:
            query_image_depth_map (numpy.ndarray): A numpy array containing LIDAR
                data about the query image.
            pixels (numpy.ndarray): A numpy array containing the projected
                feature points of the depth map.
            image (numpy.ndarray): The query image.

        Returns: The query image with the depth map plotted on it.
        """
        for i, pixel in enumerate(pixels):
            output = cv2.circle(
                image,
                (int(pixel[0]), int(pixel[1])),
                2,
                (
                    query_image_depth_map[i][3] * 255,
                    query_image_depth_map[i][3] * 255,
                    query_image_depth_map[i][3] * 0,
                ),
                -1,
            )
        return output

    def draw_circle(self, image, keypoint, color):
        return cv2.circle(
            image,
            (keypoint[0], keypoint[1]),
            20,
            color,
            -1,
        )

    def convert_depth_vectors(self, depth_data):
        """
        Converts the depth map data from homogeneous coordinates to cartesian
        coordinates.

        Args:
            depth_data (list): A list of depth map data.

        Returns: A numpy array containing the transformed depth map data.
        """
        lidar_depths = []
        for row in depth_data:
            x = row[0] * row[3]
            y = row[1] * row[3]
            z = row[2] * row[3]
            lidar_depths.append([x, y, z])
        depth_data = np.array(lidar_depths)
        depth_data = np.hstack(
            (depth_data, np.ones((depth_data.shape[0], 1)))
        ).T
        return depth_data
