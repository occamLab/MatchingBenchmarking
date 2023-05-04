import cv2
import numpy as np
from scipy.linalg import inv
from copy import copy
import matplotlib.pyplot as plt
from scipy import stats

from Benchmarker import Benchmarker
from MatchingAlgorithm import OrbMatcher, SiftMatcher, AkazeMatcher

def convert_query_depth_vectors(bundle):
    """
    Multiplies unit depth vector by it's magnitude.
    """
    query_lidar_depths = []
    print(bundle.query_image_confidence_map[-100:])
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

def convert_train_depth_vectors(bundle):
    """
    Multiplies unit depth vector by it's magnitude.
    """
    train_lidar_depths = []
    for row in bundle.train_image_depth_map:
        x = row[0] * row[3]
        y = row[1] * row[3]
        z = row[2] * row[3]
        train_lidar_depths.append([x, y, z]) 
    train_depth_data = np.array(train_lidar_depths)
    train_depth_data = np.hstack(
        (train_depth_data, np.ones((train_depth_data.shape[0], 1)))
    ).T
    return train_depth_data
    
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

# def svd_rotation(query, train):
#     """
#     Calculates the rotation matrix using SVD.
#     """
#     # Calculate the SVD of the matrix.
#     u, s, vh = np.linalg.svd(query.T.dot(train))
#     # Calculate the rotation matrix.
#     rotation = vh.T.dot(u.T)
#     return rotation

def svd_rotation(query, train):
    """
    Calculates the rotation matrix that rotates A onto B.
    """
    # calc center of mass or mean of data
    com_query = np.mean(query, axis=1)
    com_train = np.mean(train, axis=1)
    query_centered = np.array([np.subtract(row, com_query) for row in query.T])
    train_centered = np.array([np.subtract(row, com_train) for row in train.T])
    print("centered", query_centered.shape, train_centered.shape)
    W = query_centered.T @ train_centered  
    U, S, V_t = np.linalg.svd(W)
    R = np.dot(U, V_t)
    t = com_query - np.dot(R, com_train)
    return R, t

def map_depth(bundle, keypoints, query_image, train_image):
    ## Depth map of query image
    focal_length = bundle.query_image_intrinsics[0]
    offset_x = bundle.query_image_intrinsics[6]
    offset_y = bundle.query_image_intrinsics[7]

    query_depth_data = convert_query_depth_vectors(bundle)

    print("query_depth_data: ", query_depth_data.shape)

    train_depth_data = convert_train_depth_vectors(bundle)
    
    # Actual depth feature points, with magnitude removed from the vector.
    query_depth_feature_points = np.array(
        (query_depth_data[0], -query_depth_data[1], -query_depth_data[2])
    ).T

    train_depth_feature_points = np.array(
        (train_depth_data[0], -train_depth_data[1], -train_depth_data[2])
    ).T

    matched_query_depth_matrix = []
    matched_train_depth_matrix = []

    # 3d plot of point1 and point 2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    query_pose = np.array(bundle.query_image_pose).reshape(4, 4).T
    train_pose = np.array(bundle.train_image_pose).reshape(4, 4).T

    pose_difference = inv(query_pose) @ train_pose


    for unimatch in keypoints:
        matched_query_keypoint = (int(unimatch.queryPt.x), int(unimatch.queryPt.y))
        matched_train_keypoint = (int(unimatch.trainPt.x), int(unimatch.trainPt.y))

        corresponding_query_depth_index = round(
            matched_query_keypoint[0] / 7.5
        ) * 192 + round(matched_query_keypoint[1] / 7.5)
        corresponding_train_depth_index = round(
            matched_train_keypoint[0] / 7.5
        ) * 192 + round(matched_train_keypoint[1] / 7.5)

        matched_query_depth_matrix.append(np.array(query_depth_data[:,corresponding_query_depth_index]).reshape(4,1))
        matched_train_depth_matrix.append(np.array(train_depth_data[:,corresponding_train_depth_index]).reshape(4,1))

        # matched_query_depth_matrix.append(np.array(query_depth_feature_points[corresponding_query_depth_index]).reshape(3,1))
        # matched_train_depth_matrix.append(np.array(train_depth_feature_points[corresponding_train_depth_index]).reshape(3,1))
        if len(matched_query_depth_matrix) > 40:
            matched_query_depth_matrix = np.concatenate(matched_query_depth_matrix, axis=1)
            matched_train_depth_matrix = np.concatenate(matched_train_depth_matrix, axis=1)
            # inv_of_pose = matched_train_depth_matrix @ inv(matched_query_depth_matrix)

            # pose_rotated_points = inv(pose_difference) @ matched_query_depth_matrix
            # rotated_points = inv_of_pose @ matched_query_depth_matrix
            R, t = svd_rotation(matched_train_depth_matrix, matched_query_depth_matrix)
            print(R)
            print("R", R.shape)
            print("t", t.shape)
            print(matched_query_depth_matrix.shape)
            print(R.shape)
            rotation_applied = R @ matched_query_depth_matrix
            svd_rotated = np.array([np.add(row, t) for row in rotation_applied.T])
            # svd_rotated = R @ matched_train_depth_matrix, t
            print("final", svd_rotated.shape)
            

            for query_point in matched_query_depth_matrix.T:
                ax.plot(query_point[0], -query_point[1], -query_point[2], 'o', color='red')
            for query_point in matched_train_depth_matrix.T:
                ax.plot(query_point[0], -query_point[1], -query_point[2], 'o', color='blue')
                # ax.plot(matched_train_depth_matrix[i][0], -matched_train_depth_matrix[i][1], -matched_train_depth_matrix[i][2], 'o', color='blue')
            # for point in rotated_points:
            #     ax.plot(point[0], -point[1], -point[2], '*', color='green')
            # for point in pose_rotated_points:
            #     ax.plot(point[0], -point[1], -point[2], 'x', color='yellow')
            for point in svd_rotated:
                ax.plot(point[0], -point[1], -point[2], '^', color='black')

            # print("pose: ", inv(inv_of_pose))
            print("sensor pose: ", inv(pose_difference))
            # img_out = np.concatenate((query_image, train_image), axis=1)
            # cv2.imshow("img_out", img_out)
            # cv2.waitKey(0)
            matched_query_depth_matrix = []
            matched_train_depth_matrix = []
            plt.show()
            inp = input("d")






    # calculate depths and pixels of feature points
    pixels = project_depth_onto_image(query_depth_feature_points, focal_length, offset_x, offset_y)

    final_query_image = plot_depth_map(pixels, query_image)


session = Benchmarker([OrbMatcher(), SiftMatcher(), AkazeMatcher()], [0.01, 0.25, 0.45, 0.85]).sessions[1]

for bundle in session.bundles:
    query_image = copy(bundle.query_image)
    train_image = copy(bundle.train_image)

    new_superglue_matcher = AkazeMatcher().get_matches(query_image, train_image, 0.45)

    map_depth(bundle, new_superglue_matcher, query_image, train_image)

    