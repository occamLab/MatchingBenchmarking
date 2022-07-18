import cv2
import numpy as np
from scipy.linalg import inv
from copy import copy
import matplotlib.pyplot as plt
from scipy import stats

from Benchmarker import Benchmarker
from MatchingAlgorithm import OrbMatcher, SiftMatcher, AkazeMatcher

def convert_unit_depth_vectors(bundle):
    """
    Multiplies unit depth vector by it's magnitude.
    """
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
    return query_depth_data
    
def project_depth_onto_image(query_depth_feature_points, focal_length, offset_x, offset_y):
    pixels = []
    for row in query_depth_feature_points:
        pixel_x = row[0] * focal_length / row[2] + offset_x
        pixel_y = row[1] * focal_length / row[2] + offset_y
        pixels.append((pixel_x, pixel_y))
    return pixels    

def plot_depth_map(pixels, image):
    for i, pixel in enumerate(pixels):
        output = cv2.circle(
            image,
            (int(pixel[0]), int(pixel[1])),
            1,
            (
                bundle.query_image_depth_map[i][3] * 255,
                bundle.query_image_depth_map[i][3] * 255,
                bundle.query_image_depth_map[i][3] * 255,
            ),
            -1,
        )
    return output

def draw_circle(image, keypoint, color):
    return cv2.circle(
            image,
            (keypoint[0], keypoint[1]),
            20,
            color,
            -1,
        )


def map_depth(bundle, keypoints, query_image, train_image):
    ## Depth map of query image
    focal_length = bundle.query_image_intrinsics[0]
    offset_x = bundle.query_image_intrinsics[6]
    offset_y = bundle.query_image_intrinsics[7]

    query_depth_data = convert_unit_depth_vectors(bundle)
    
    # Actual depth feature points, with magnitude removed from the vector.
    query_depth_feature_points = np.array(
        (query_depth_data[0], -query_depth_data[1], -query_depth_data[2])
    ).T

    # calculate depths and pixels of feature points
    pixels = project_depth_onto_image(query_depth_feature_points, focal_length, offset_x, offset_y)

    final_query_image = plot_depth_map(pixels, query_image)

    ## Depth map of train image

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
    pixels = project_depth_onto_image(projected_depth_feature_points, focal_length, offset_x, offset_y)

    final_train_image = plot_depth_map(pixels, train_image)

    depth_point_to_algo_point_distances = []
    ## Project corresponding query image keypoints onto train image which are matched using depth map data
    for unimatch in keypoints:
        matched_query_keypoint = (int(unimatch.queryPt.x), int(unimatch.queryPt.y))
        matched_train_keypoint = (int(unimatch.trainPt.x), int(unimatch.trainPt.y))

        # get corresponding depth map index for each keypoint. 
            # Keypoints in a rectangular area around a depth index are matched to same index.
            # This is done since the resolution of depth map is lower than the resolution of the image.
        corresponding_depth_index = round(
            matched_query_keypoint[0] / 7.5
        ) * 192 + round(matched_query_keypoint[1] / 7.5)

        # Draw query image keypoints
        final_query_image = draw_circle(final_query_image, matched_query_keypoint, (0, 0, 0))


        algo_matched_point = np.array((matched_train_keypoint[0], matched_train_keypoint[1]))
        depth_matched_point = np.array((int(pixels[corresponding_depth_index][0]), int(pixels[corresponding_depth_index][1])))
        
        # Draw train image keypoints, matched using the algorithm
        final_train_image = draw_circle(final_train_image, algo_matched_point, (0, 0, 0))

        # Plots corresponding depth point from query image on train image, matched using the depth data
        final_train_image  = draw_circle(final_train_image, depth_matched_point, (255,255,255))
        
        # draw line between algo matched point and depth matched point
        final_train_image = cv2.line(
            final_train_image,
            algo_matched_point,
            depth_matched_point,
            (
                255,
                255,
                255,
            ),
            1,
        )

        depth_point_to_algo_point_distances.append(np.linalg.norm(algo_matched_point - depth_matched_point))

    # plt.scatter(depth_point_to_algo_point_distances, range(len(depth_point_to_algo_point_distances)))
    # plt.boxplot(depth_point_to_algo_point_distances)

    # kde = stats.gaussian_kde(depth_point_to_algo_point_distances)
    # x = np.linspace(0, max(depth_point_to_algo_point_distances), 100)
    # p = kde(x)
    # plt.plot(x, p)

    filtered_points = [x for x in depth_point_to_algo_point_distances if x < 100]

    plt.hist(filtered_points)
    depth_point_to_algo_point_distances = []
    plt.xlabel("Depth point to algo point distance")
    plt.ylabel("No. of Points")
    plt.savefig('depth_point_to_algo_point_distances.png')
    plt.clf()
    cv2.imwrite("query.png", final_query_image)
    cv2.imwrite("train.png", final_train_image)
    userinput = input("d")


session = Benchmarker("Temp").sessions[1]

for bundle in session.bundles:

    query_image = copy(bundle.query_image)
    train_image = copy(bundle.train_image)

    new_superglue_matcher = SiftMatcher().get_matches(query_image, train_image)

    map_depth(bundle, new_superglue_matcher, query_image, train_image)
    