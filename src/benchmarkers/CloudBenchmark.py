"""
This file runs the benchmarking code for the cloud anchors. It is used to
project 2 different recording of the same session onto each other using a matching cloud anchor.
We use the phone poses associated with the matching cloud anchors to find a pose difference
and apply that to the lidar data from one session to project it onto the other.
Currently the code only takes the first pair of matching cloud anchors and uses that
to compare the sessions. For future implementations, some kind of heuristic should be
used to determine which pair of matching cloud anchors to use.
"""
from scipy.linalg import inv
import cv2
import numpy as np
from Benchmarker import Benchmarker
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
sys.path.append("..")


def compare_cloud_matches(benchmarker, cloud_to_s1, cloud_to_s2, s1_bundle, s2_bundle, i):
    print(f"new image:{i}")
    focal_length = s1_bundle.query_image_intrinsics[0]
    offset_x = s1_bundle.query_image_intrinsics[6]
    offset_y = s1_bundle.query_image_intrinsics[7]

    query_depth_data = benchmarker.convert_depth_vectors(
        s1_bundle.query_image_depth_map)

    # Actual depth feature points, with magnitude removed from the vector.
    query_depth_feature_points = np.array(
        (query_depth_data[0], -query_depth_data[1], -query_depth_data[2])
    ).T

    # calculate depths and pixels of feature points
    pixels = benchmarker.project_depth_onto_image(
        query_depth_feature_points, focal_length, offset_x, offset_y)

    final_query_image = benchmarker.plot_depth_map(
        s1_bundle.query_image_depth_map, pixels, s1_bundle.query_image)

    # Find the pose difference between the two sessions
    s2_to_s1 = cloud_to_s1 @ np.linalg.inv(cloud_to_s2)

    query_depth_data_projected_on_train = inv(
        s2_to_s1) @ query_depth_data

    projected_depth_feature_points = np.array(
        (
            query_depth_data_projected_on_train[0],
            -query_depth_data_projected_on_train[1],
            -query_depth_data_projected_on_train[2],
        )
    ).T

    pixels = []
    pixels = benchmarker.project_depth_onto_image(
        projected_depth_feature_points, focal_length, offset_x, offset_y)

    count = 0
    for pixel in pixels:
        if int(pixel[0]) in range(0, s2_bundle.query_image.shape[0]) and int(pixel[1]) in range(0, s2_bundle.query_image.shape[1]):
            count += 1

    print(
        f"count: {count}, pixels: {len(pixels)}, percentage: {count/len(pixels)}")

    final_train_image = benchmarker.plot_depth_map(
        s1_bundle.query_image_depth_map, pixels, s2_bundle.query_image)

    overlap_img = cv2.addWeighted(
        final_query_image, 0.5, final_train_image, 0.5, 0)
    cv2.imwrite("overlap.png", overlap_img)
    cv2.imwrite("query.png", final_query_image)
    cv2.imwrite("train.png", final_train_image)

    # create figure
    fig = plt.figure(figsize=(10, 7))

    # setting values to rows and column variables
    rows = 1
    columns = 2

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)

    # showing image
    plt.imshow(final_query_image)
    plt.axis('off')
    plt.title("First")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)

    # showing image
    plt.imshow(final_train_image)
    plt.axis('off')
    plt.title("Second")

    plt.show()

    # d = input("d")

    return final_query_image, final_train_image

    # take timestamps from garAnchorTimestamps and match each garAnchor to a corresponding cloudAnchor to get the timestamp of that cloudAnchor


def cloud_anchor_pose_test(benchmarker, num_runs=None):
    first_session = benchmarker.sessions[0]
    second_session = benchmarker.sessions[1]

    print(len(first_session.bundles))
    print(len(second_session.bundles))

    first_session_cloud = first_session.all_metadata["cloudAnchorsForAlignment"]
    second_session_cloud = second_session.all_metadata["cloudAnchorsForAlignment"]

    cloud_to_s1 = np.array(
        first_session_cloud[0]["anchorTransform"]).reshape(4, 4).T
    cloud_to_s2 = np.array(
        second_session_cloud[0]["anchorTransform"]).reshape(4, 4).T

    if num_runs is None:
        cloud_session = first_session_cloud
    else:
        cloud_session = first_session_cloud[:num_runs]

    for i, anchor in enumerate(cloud_session):
        first_session_bundle = first_session.bundles[i]
        second_session_bundle = second_session.bundles[i]

        img1, img2 = compare_cloud_matches(benchmarker,
                                           cloud_to_s1, cloud_to_s2, first_session_bundle, second_session_bundle, i)


def run_benchmark(algorithms, values, num_runs=None):
    benchmarker = Benchmarker(algorithms, values)
    cloud_anchor_pose_test(benchmarker, num_runs)
