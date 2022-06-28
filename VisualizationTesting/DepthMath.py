from re import I
import sys
import os
import cv2
import numpy as np
import pyvista as pv

# setting path to src files
sys.path.append(f"{os.path.dirname(os.path.dirname(__file__))}/src/")
from Benchmarker import Benchmarker
from MatchingAlgorithm import OrbMatcher, SiftMatcher


session = Benchmarker("A").sessions[0]
bundle = session.bundles[0]
new_sift_matcher = SiftMatcher().get_matches(bundle.query_image[0], bundle.train_image[0])

query_depth_data = np.array(bundle.query_image_depth_map)

# scale LiDAR data
lidar_depth = []
for row in query_depth_data:
    x = row[0] * row[3]
    y = row[1] * row[3]
    z = row[2] * row[3]
    lidar_depth.append([x,y,z])
# lidar_depth = np.array(lidar_depth)
img = cv2.circle(bundle.query_image[0], (int(lidar_depth[10000][0])*-100,int(lidar_depth[10000][1])*100), 30, (0, 0, 255), -1)
cv2.imshow("img",img)
cv2.waitKey(0)
# visualize a 3D point cloud of the LiDAR depth data
# pv_point_cloud = pv.PolyData(lidar_depth)
# pv_point_cloud.plot(render_points_as_spheres=True)
# cv2.imshow("img", bundle.query_image[0])
# cv2.waitKey(0)