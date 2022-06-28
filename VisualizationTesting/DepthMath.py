from re import I
import sys
import os
import cv2
from cv2 import ROTATE_90_COUNTERCLOCKWISE
import numpy as np
from scipy.linalg import inv
import math as m

# import pyvista as pv

# setting path to src files
sys.path.append(f"{os.path.dirname(os.path.dirname(__file__))}/src/")
from Benchmarker import Benchmarker
from MatchingAlgorithm import OrbMatcher, SiftMatcher


session = Benchmarker("Temp").sessions[0]
bundle = session.bundles[0]
new_sift_matcher = SiftMatcher().get_matches(bundle.query_image, bundle.query_image)


def map_depth(bundle):

    focal_length = bundle.query_image_intrinsics[0]
    offset_x = bundle.query_image_intrinsics[6]
    offset_y = bundle.query_image_intrinsics[7]

    lidar_depth = []
    for row in bundle.query_image_depth_map:
        x = row[0] * row[3]
        y = row[1] * row[3]
        z = row[2] * row[3]
        lidar_depth.append([x, y, z])

    depth_data = np.array(lidar_depth)
    depth_data = np.hstack((depth_data, np.ones((depth_data.shape[0], 1)))).T
    depth_fp = np.array((depth_data[0], -depth_data[1], -depth_data[2])).T

    # calculate depths and pixels of feature points
    pixels = []
    for row in depth_fp:
        pixel_x = row[0] * focal_length / row[2] + offset_x
        pixel_y = row[1] * focal_length / row[2] + offset_y
        pixels.append((pixel_x, pixel_y))

    final_image = bundle.query_image

    for i, pixel in enumerate(pixels):
        img = cv2.circle(
            final_image,
            (int(pixel[0]), int(pixel[1])),
            10,
            (
                bundle.query_image_depth_map[i][3] * 255,
                bundle.query_image_depth_map[i][3] * 255,
                bundle.query_image_depth_map[i][3] * 255,
            ),
            -1,
        )
    cv2.imwrite("query_img.png", img)

    focal_length = bundle.train_image_intrinsics[0]
    offset_x = bundle.train_image_intrinsics[6]
    offset_y = bundle.train_image_intrinsics[7]

    lidar_depth = []
    for row in bundle.train_image_depth_map:
        x = row[0] * row[3]
        y = row[1] * row[3]
        z = row[2] * row[3]
        lidar_depth.append([x, y, z])

    depth_data = np.array(lidar_depth)
    depth_data = np.hstack((depth_data, np.ones((depth_data.shape[0], 1)))).T
    depth_fp = np.array((depth_data[0], -depth_data[1], -depth_data[2])).T

    # calculate depths and pixels of feature points
    pixels = []
    for row in depth_fp:
        pixel_x = row[0] * focal_length / row[2] + offset_x
        pixel_y = row[1] * focal_length / row[2] + offset_y
        pixels.append((pixel_x, pixel_y))

    final_image = bundle.train_image

    for i, pixel in enumerate(pixels):
        img = cv2.circle(
            final_image,
            (int(pixel[0]), int(pixel[1])),
            10,
            (
                bundle.train_image_depth_map[i][3] * 255,
                bundle.train_image_depth_map[i][3] * 255,
                bundle.train_image_depth_map[i][3] * 255,
            ),
            -1,
        )
    cv2.imwrite("train_img.png", img)


map_depth(session.bundles[0])
