from re import I
import sys
import os
import cv2
import numpy as np
from scipy.linalg import inv
import math as m

sys.path.append(f"{os.path.dirname(os.path.dirname(__file__))}/src/")
from Benchmarker import Benchmarker
from MatchingAlgorithm import OrbMatcher, SiftMatcher


session = Benchmarker("Temp").sessions[0]
bundle = session.bundles[11]
new_sift_matcher = SiftMatcher().get_matches(bundle.query_image, bundle.train_image)


def map_depth(bundle):
    focal_length = bundle.query_image_intrinsics[0]
    offset_x = bundle.query_image_intrinsics[6]
    offset_y = bundle.query_image_intrinsics[7]

    query_lidar_depths = []
    for row in bundle.query_image_depth_map:
        x = row[0] * row[3]
        y = row[1] * row[3]
        z = row[2] * row[3]
        query_lidar_depths.append([x, y, z])

    query_depth_data = np.array(query_lidar_depths)
    query_depth_data = np.hstack(
        (query_depth_data, np.ones((query_depth_data.shape[0], 1)))
    ).T
    query_depth_feature_points = np.array(
        (query_depth_data[0], -query_depth_data[1], -query_depth_data[2])
    ).T

    # calculate depths and pixels of feature points
    pixels = []
    for row in query_depth_feature_points:
        pixel_x = row[0] * focal_length / row[2] + offset_x
        pixel_y = row[1] * focal_length / row[2] + offset_y
        pixels.append((pixel_x, pixel_y))

    query_image = bundle.query_image

    for i, pixel in enumerate(pixels):
        final_query_image = cv2.circle(
            query_image,
            (int(pixel[0]), int(pixel[1])),
            1,
            (
                255,
                255,
                255,
            ),
            -1,
        )

    # Represents the keypoint
    test_pixel = (900, 500)
    final_query_image = cv2.circle(
        query_image,
        (test_pixel[0], test_pixel[1]),
        5,
        (
            255,
            255,
            255,
        ),
        -1,
    )

    # Converts pixels of keypoint to depth point index
    corresponding_depth_index = round(test_pixel[0] / 7.5) * 192 + round(
        test_pixel[1] / 7.5
    )

    # Plots corresponding depth point
    final_query_image = cv2.circle(
        query_image,
        (
            int(pixels[corresponding_depth_index][0]),
            int(pixels[corresponding_depth_index][1]),
        ),
        5,
        (
            0,
            0,
            0,
        ),
        -1,
    )
    cv2.imwrite(f"query.png", final_query_image)

    focal_length = bundle.train_image_intrinsics[0]
    offset_x = bundle.train_image_intrinsics[6]
    offset_y = bundle.train_image_intrinsics[7]

    query_pose = np.array(bundle.query_image_pose).reshape(4, 4).T
    train_pose = np.array(bundle.train_image_pose).reshape(4, 4).T

    pose_difference = inv(query_pose) @ train_pose

    query_depth_data_projected_on_train = inv(pose_difference) @ query_depth_data
    projected_depth_feature_points = np.array(
        (
            query_depth_data_projected_on_train[0],
            -query_depth_data_projected_on_train[1],
            -query_depth_data_projected_on_train[2],
        )
    ).T

    pixels = []
    for row in projected_depth_feature_points:
        pixel_x = row[0] * focal_length / row[2] + offset_x
        pixel_y = row[1] * focal_length / row[2] + offset_y
        pixels.append((pixel_x, pixel_y))

    train_image = bundle.train_image

    for i, pixel in enumerate(pixels):
        final_train_image = cv2.circle(
            train_image,
            (int(pixel[0]), int(pixel[1])),
            1,
            (
                bundle.query_image_depth_map[i][3] * 255,
                bundle.query_image_depth_map[i][3] * 255,
                bundle.query_image_depth_map[i][3] * 255,
            ),
            -1,
        )

    # Plots corresponding depth point from query image on trian image
    final_train_image = cv2.circle(
        train_image,
        (
            int(pixels[corresponding_depth_index][0]),
            int(pixels[corresponding_depth_index][1]),
        ),
        5,
        (
            0,
            0,
            0,
        ),
        -1,
    )
    cv2.imwrite("train.png", final_train_image)


map_depth(bundle)
