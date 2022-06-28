from re import I
import sys
import os
import cv2
import numpy as np
from scipy.linalg import inv
import math as m

# import pyvista as pv

# setting path to src files
sys.path.append(f"{os.path.dirname(os.path.dirname(__file__))}/src/")
from Benchmarker import Benchmarker
from MatchingAlgorithm import OrbMatcher, SiftMatcher


session = Benchmarker("Temp").sessions[1]
bundle = session.bundles[0]
new_sift_matcher = SiftMatcher().get_matches(bundle.query_image, bundle.train_image)

intrinsics_dict = {
    "query": bundle.query_image_intrinsics,
    "train": bundle.train_image_intrinsics,
}
depth_dict = {
    "query": bundle.query_image_depth_map,
    "train": bundle.train_image_depth_map,
}

def map_depth(bundle, image_type):
    focal_length = intrinsics_dict[image_type][0]
    offset_x = intrinsics_dict[image_type][6]
    offset_y = intrinsics_dict[image_type][7]

    lidar_depth = []
    for row in depth_dict[image_type]:
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

    if image_type == "query":
        final_image = bundle.query_image
    else:
        final_image = bundle.train_image
    for i, pixel in enumerate(pixels):
        img = cv2.circle(
            final_image,
            (int(pixel[0]), int(pixel[1])),
            10,
            (
                depth_dict[image_type][i][3] * 255,
                depth_dict[image_type][i][3] * 255,
                depth_dict[image_type][i][3] * 255,
            ),
            -1,
        )
    cv2.imwrite(f"{image_type}.png", img)


map_depth(session.bundles[0], "train")
map_depth(session.bundles[0], "query")
