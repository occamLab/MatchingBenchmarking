import os
import pickle
import cv2
from scipy.linalg import inv
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

class Benchmarker:
    """
    Validates the matches of user specified matching algorithms.

    Using data about the query and train images (stored as a bundle inside a session), it is
    possible to map a keypoint from the query image to a keypoint on the train
    image (and vice versa). First, we let the matching algorithm produce a list
    of matches which contain keypoints in the query and train images. Using data
    about the query and train images from the bundle, we map every keypoint
    (generated by the matching algorithm) from the query image to the train image.
    Finally, by noting how far the train keypoints produced by the matching algorithm
    are from our correctly mapped keypoints, we can score each algorithm based on how
    many of its keypoints fell within a certain distance threshold from the correctly
    mapped keypoints. A better score means that more of the keypoints generated by the
    algorithm in the train image were close to the keypoints in the train image that
    we generated based off of the bundle data.

    Instance Attributes:
        sessions (list): A list of Session objects.
        Algorithms (list): A list of MatchingAlgorithm objects to be benchmarked.
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
            sessions = pickle.load(sessions_file)
        return sessions

    def convert_unit_depth_vectors(self, bundle):
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
        
    def project_depth_onto_image(self, query_depth_feature_points, focal_length, offset_x, offset_y):
        pixels = []
        for row in query_depth_feature_points:
            pixel_x = row[0] * focal_length / row[2] + offset_x
            pixel_y = row[1] * focal_length / row[2] + offset_y
            pixels.append((pixel_x, pixel_y))
        return pixels    

    def plot_depth_map(self, bundle, pixels, image):
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

    def draw_circle(self, image, keypoint, color):
        return cv2.circle(
                image,
                (keypoint[0], keypoint[1]),
                20,
                color,
                -1,
            )


    def compare_matches(self, bundle, matches, query_image, train_image):
        """
        Compares the matches produced by the algorithm with the matches produced
        by the depth map data.

        Args: 
            bundle (Bundle): A Bundle object containing the data about the query and train images.
            matches (list): A list of UniMatch objects produced by the algorithm.
            query_image (numpy.ndarray): The query image.
            train_image (numpy.ndarray): The train image.

        Returns: A list of distances between the matches produced by the algorithm
        and the matches produced by the depth map data.
        """
        ## Depth map of query image
        focal_length = bundle.query_image_intrinsics[0]
        offset_x = bundle.query_image_intrinsics[6]
        offset_y = bundle.query_image_intrinsics[7]

        query_depth_data = self.convert_unit_depth_vectors(bundle)
        
        # Actual depth feature points, with magnitude removed from the vector.
        query_depth_feature_points = np.array(
            (query_depth_data[0], -query_depth_data[1], -query_depth_data[2])
        ).T

        # calculate depths and pixels of feature points
        pixels = self.project_depth_onto_image(query_depth_feature_points, focal_length, offset_x, offset_y)

        final_query_image = self.plot_depth_map(bundle, pixels, query_image)

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
        pixels = self.project_depth_onto_image(projected_depth_feature_points, focal_length, offset_x, offset_y)

        final_train_image = self.plot_depth_map(bundle, pixels, train_image)

        depth_point_to_algo_point_distances = []
        ## Project corresponding query image keypoints onto train image which are matched using depth map data
        for unimatch in matches:
            matched_query_keypoint = (int(unimatch.queryPt.x), int(unimatch.queryPt.y))
            matched_train_keypoint = (int(unimatch.trainPt.x), int(unimatch.trainPt.y))

            # get corresponding depth map index for each keypoint. 
                # Keypoints in a rectangular area around a depth index are matched to same index.
                # This is done since the resolution of depth map is lower than the resolution of the image.
            corresponding_depth_index = round(
                matched_query_keypoint[0] / 7.5
            ) * 192 + round(matched_query_keypoint[1] / 7.5)

            # Draw query image keypoints
            # final_query_image = self.draw_circle(final_query_image, matched_query_keypoint, (255, 255, 255))


            algo_matched_point = np.array((matched_train_keypoint[0], matched_train_keypoint[1]))
            depth_matched_point = np.array((int(pixels[corresponding_depth_index][0]), int(pixels[corresponding_depth_index][1])))
            
            # Draw train image keypoints, matched using the algorithm
            # final_train_image = self.draw_circle(final_train_image, algo_matched_point, (0, 0, 0))

            # Plots corresponding depth point from query image on train image, matched using the depth data
            # final_train_image  = self.draw_circle(final_train_image, depth_matched_point, (255,255,255))
            
            # draw line between algo matched point and depth matched point
            # final_train_image = cv2.line(
            #     final_train_image,
            #     algo_matched_point,
            #     depth_matched_point,
            #     (
            #         255,
            #         255,
            #         255,
            #     ),
            #     1,
            # )

            depth_point_to_algo_point_distances.append(np.linalg.norm(algo_matched_point - depth_matched_point))

        return depth_point_to_algo_point_distances, final_query_image, final_train_image
    def benchmark(self):
        """
        Scores each algorithm based on how close their matches are with matches
        produced by the bundle data for each image pair.
        """
        total_points_less_than_100 = []
        total_points = []
        correct_vs_incorrect_for_one_algo = []
        for algorithm in self.algorithms:
            for ratio in self.sweep_values:
                for session in self.sessions:
                    print(repr(algorithm), session, ratio)
                    for bundle in session.bundles:
                        query_image = copy(bundle.query_image)
                        train_image = copy(bundle.train_image)

                        matches = algorithm.get_matches(query_image, train_image, ratio)

                        try:
                            distances, final_query_image, final_train_image = self.compare_matches(bundle, matches, query_image, train_image)
                        except Exception as e:
                            print(e)
                        # plt.scatter(depth_point_to_algo_point_distances, range(len(depth_point_to_algo_point_distances)))
                        # plt.boxplot(depth_point_to_algo_point_distances)

                        # kde = stats.gaussian_kde(depth_point_to_algo_point_distances)
                        # x = np.linspace(0, max(depth_point_to_algo_point_distances), 100)
                        # p = kde(x)
                        # plt.plot(x, p)
                        
                        filtered_points = [x for x in distances if x < 100]
                        try:
                            filtered_points = [x for x in filtered_points if x < np.quantile(copy(filtered_points), 0.25, axis=0)]
                            print(len(filtered_points))
                        except:
                            continue
                        total_points_less_than_100.append(len(filtered_points))
                        total_points.append(len(distances))
                        # plt.hist(filtered_points)
                        # plt.xlabel("Depth point to algo point distance")
                        # plt.ylabel("No. of Points")
                        # plt.savefig('depth_point_to_algo_point_distances.png')
                        # plt.clf()
                        # cv2.imwrite("query.png", final_query_image)
                        # cv2.imwrite("train.png", final_train_image)
                        # userinput = input("d")
                try:
                    correct_vs_incorrect_for_one_algo.append((ratio, sum(total_points_less_than_100) / sum(total_points), algorithm))
                except:
                    print(len(total_points_less_than_100))
                # TODO: add quantile i.e. take bottom 10% of points
        print("total", len(correct_vs_incorrect_for_one_algo))
        for x in correct_vs_incorrect_for_one_algo:
            if repr(x[-1]) == "Orb":
                plt.plot(x[0], x[1], 'o', color='red', label="Orb")
            elif repr(x[-1]) == "Sift":
                plt.plot(x[0], x[1], '^', color='green', label="Sift")
            elif repr(x[-1]) == "Akaze":
                plt.plot(x[0], x[1], '*', color ='blue', label="Akaze")
        plt.xticks(self.sweep_values)
        plt.xlabel("Quantile values")
        plt.ylabel("Ratio of correct/total matches")
        label_orb = mpatches.Patch(color='red', label='Orb')
        label_sift = mpatches.Patch(color='green', label='Sift')
        label_akaze = mpatches.Patch(color='blue', label='Akaze')
        plt.legend(handles=[label_orb, label_sift, label_akaze])
        plt.savefig("Ratio test.png")



