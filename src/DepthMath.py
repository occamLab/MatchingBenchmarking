import sys
import os
import cv2
import numpy as np
from scipy.linalg import inv
from copy import copy

# sys.path.append(f"{os.path.dirname(os.path.dirname(__file__))}/src/")
from Benchmarker import Benchmarker
from MatchingAlgorithm import OrbMatcher, SiftMatcher, SuperGlueMatcher


def map_depth(bundle, keypoints, query_image, train_image):
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

    final_query_image = query_image

    # Plot depth map
    for i, pixel in enumerate(pixels):
        final_query_image = cv2.circle(
            final_query_image,
            (int(pixel[0]), int(pixel[1])),
            1,
            (
                255,
                255,
                255,
            ),
            -1,
        )

    # Converts pixels of keypoint to depth point index
    # # Plots corresponding depth point
    # final_query_image = cv2.circle(
    #     query_image,
    #     (
    #         int(pixels[corresponding_depth_index][0]),
    #         int(pixels[corresponding_depth_index][1]),
    #     ),
    #     10,
    #     (
    #         0,
    #         0,
    #         0,
    #     ),
    #     -1,
    # )

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

    final_train_image = train_image

    # Plot depth map
    for i, pixel in enumerate(pixels):
        final_train_image = cv2.circle(
            final_train_image,
            (int(pixel[0]), int(pixel[1])),
            1,
            (
                bundle.query_image_depth_map[i][3] * 255,
                bundle.query_image_depth_map[i][3] * 255,
                bundle.query_image_depth_map[i][3] * 255,
            ),
            -1,
        )

    for unimatch in keypoints:
        matched_query_keypoint = (int(unimatch.queryPt.x), int(unimatch.queryPt.y))
        matched_train_keypoint = (int(unimatch.trainPt.x), int(unimatch.trainPt.y))

        corresponding_depth_index = round(
            matched_query_keypoint[0] / 7.5
        ) * 192 + round(matched_query_keypoint[1] / 7.5)

        # Represents the keypoint
        final_query_image = cv2.circle(
            final_query_image,
            (matched_query_keypoint[0], matched_query_keypoint[1]),
            20,
            (
                0,
                0,
                0,
            ),
            -1,
        )

        # Represents the matched train keypoint
        final_train_image = cv2.circle(
            final_train_image,
            (matched_train_keypoint[0], matched_train_keypoint[1]),
            20,
            (
                0,
                0,
                0,
            ),
            -1,
        )

        # Plots corresponding depth point from query image on train image
        final_train_image = cv2.circle(
            final_train_image,
            (
                int(pixels[corresponding_depth_index][0]),
                int(pixels[corresponding_depth_index][1]),
            ),
            20,
            (
                255,
                255,
                255,
            ),
            -1,
        )

    cv2.imwrite("query.png", final_query_image)
    cv2.imwrite("train.png", final_train_image)
    userinput = input("d")


session = Benchmarker("Temp").sessions[2]

for bundle in session.bundles:

    query_image = copy(bundle.query_image)
    train_image = copy(bundle.train_image)

    new_superglue_matcher = SuperGlueMatcher().get_matches(query_image, train_image)

    map_depth(bundle, new_superglue_matcher[:10], query_image, train_image)
