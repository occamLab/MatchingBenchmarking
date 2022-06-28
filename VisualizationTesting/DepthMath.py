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


session = Benchmarker("Temp").sessions[0]
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
                depth_dict[image_type][i][3] * 256,
                depth_dict[image_type][i][3] * 256,
                depth_dict[image_type][i][3] * 256,
            ),
            -1,
        )
    cv2.imwrite(f"{image_type}.png", img)


# map_depth(session.bundles[0], "train")
# map_depth(session.bundles[0], "query")
map_depth(bundle, "query")

focal_length = intrinsics_dict["query"][0]
offset_x = intrinsics_dict["query"][6]
offset_y = intrinsics_dict["query"][7]

lidar_depth = []
for row in depth_dict["query"]:
    x = row[0] * row[3]
    y = row[1] * row[3]
    z = row[2] * row[3]
    lidar_depth.append([x, y, z])

query_pose = np.array(bundle.query_image_pose).reshape(4, 4).T
train_pose = np.array(bundle.train_image_pose).reshape(4, 4).T
query_pose = np.array(
    [
        [-2.01407932e-02, 9.99681711e-01, -1.51819335e-02, 7.53196888e-03],
        [-9.34941351e-01, -1.34520307e-02, 3.54547024e-01, -1.56493671e-03],
        [3.54230016e-01, 2.13350747e-02, 9.34914768e-01, 4.04297607e-05],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)
train_pose = np.array(
    [
        [0.03808993, 0.97807384, 0.20474519, -0.04228871],
        [-0.94415671, -0.03188249, 0.32795048, -0.00612328],
        [0.32728761, -0.20580319, 0.92224056, 0.01524482],
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    ]
)

print(query_pose)
print(train_pose)

relative_pose = inv(query_pose) @ train_pose

depth_data = np.array(lidar_depth)
depth_data = np.hstack((depth_data, np.ones((depth_data.shape[0], 1)))).T

depth_data_projected = relative_pose @ depth_data
depth_fp = np.array(
    (depth_data_projected[0], -depth_data_projected[1], -depth_data_projected[2])
).T

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
            depth_dict["query"][i][3] * 256,
            depth_dict["query"][i][3] * 256,
            depth_dict["query"][i][3] * 256,
        ),
        -1,
    )
cv2.imwrite(f"train1.png", img)
