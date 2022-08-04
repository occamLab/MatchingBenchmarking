from abc import ABC, abstractmethod
import cv2
import matplotlib.pyplot as plt
from UniMatch import UNIMatch
import numpy as np

# from SuperGlue import get_superglue_matches, draw_superglue_matches


class MatchingAlgorithm(ABC):
    """
    Abstract class representing a Matching Algorithm object.

    Instance Attributes:
        query_image (numpy.ndarray): The query image passed into the algorithm.
        train_image (numpy.ndarray): The train image passed into the algorithm.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_matches(self):
        """
        Runs the algorithm on the input images and returns a list of matches.

        Returns:
            matches (list): A list of matches.
            query_keypoints (list): A list of keypoints in the query image.
            train_keypoints (list): A list of keypoints in the train image.
        """
        pass

    @abstractmethod
    def matches_to_unimatches(self):
        """
        Creates a list of UNIMatch objects representing the matches between the query and train images.

        Returns:
            uniMatches (list): A list of UNIMatch objects.
        """
        pass

    def __repr__(self):
        pass

class OpenCVAlgorithm(MatchingAlgorithm):
    """
    Class representing an OpenCV Algorithm.
    """
    def __init__(self):
        super().__init__()

    def get_matches(self, query_image, train_image, algorithm, quantile=0.25, ratio_not_quantile=False):
        # Find the keypoints and descriptors with SIFT
        query_keypoints, query_descriptors = algorithm.detectAndCompute(query_image, None)
        train_keypoints, train_descriptors = algorithm.detectAndCompute(train_image, None)

        # Get matches from keypoints and descriptors
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(query_descriptors, train_descriptors, k=2)

        if ratio_not_quantile:
            ratio = quantile
        else:
            ratio_tests = []
            for match, nearest_neighbor in matches:
                ratio_tests.append(match.distance / nearest_neighbor.distance)
            
            ratio = np.quantile(ratio_tests, quantile)
        # Apply ratio test (currently set to 1 so it does nothing)
        filtered_matches = [
            match
            for match, nearest_neighbor in matches
            if match.distance < ratio * nearest_neighbor.distance
        ]

        return self.matches_to_unimatches(
            filtered_matches, query_keypoints, train_keypoints
        )

    def matches_to_unimatches(self, matches, query_keypoints, train_keypoints):
        # Creates
        coord_array = []
        for match in matches:
            x1, y1 = query_keypoints[match.queryIdx].pt
            x2, y2 = train_keypoints[match.trainIdx].pt
            coord_array.extend([x1, y1, x2, y2])

        # Create list of UNIMatch objects
        uniMatches = [
            UNIMatch(
                coord_array[i],
                coord_array[i + 1],
                coord_array[i + 2],
                coord_array[i + 3],
            )
            for i in range(0, len(coord_array), 4)
        ]

        return uniMatches
    
    def __repr__(self):
        pass

# class SuperGlueMatcher(MatchingAlgorithm):
#     """
#     Matching Algorithm that uses SuperGlue to match keypoints between two images.
#     """

#     def __init__(self):
#         pass

#     def get_matches(self, query_image, train_image):
#         return self.matches_to_unimatches(*get_superglue_matches(query_image, train_image))

#     def draw_matches(self):
#         return draw_superglue_matches()

#     def matches_to_unimatches(self, matches, mkpts0, mkpts1):
#         # Create list of UNIMatch objects
#         uniMatches = []
#         for i in range(0, len(matches), 4):
#             uniMatches.append(UNIMatch(
#                 matches[i],
#                 matches[i + 1],
#                 matches[i + 2],
#                 matches[i + 3],
#             ))

#         return uniMatches
#     def __repr__(self):
#         return "SuperGlueMatcher"


class OrbMatcher(OpenCVAlgorithm):
    """
    Matching Algorithm that uses ORB to match keypoints between two images.
    """

    algorithm = cv2.ORB_create()

    def __init__(self):
        pass

    def get_matches(self, query_image, train_image, ratio, ratio_not_quantile=False):
        return super().get_matches(query_image, train_image, self.algorithm, ratio, ratio_not_quantile)

    def matches_to_unimatches(self, matches, query_keypoints, train_keypoints):
        return super().matches_to_unimatches(matches, query_keypoints, train_keypoints)
    
    def __repr__(self):
        return "Orb"


class SiftMatcher(OpenCVAlgorithm):
    """
    Matching Algorithm that uses SIFT to match keypoints between two images.
    """

    algorithm = cv2.SIFT_create()

    def __init__(self):
        pass

    def get_matches(self, query_image, train_image, ratio, ratio_not_quantile=False):
        return super().get_matches(query_image, train_image, self.algorithm, ratio, ratio_not_quantile)

    def matches_to_unimatches(self, matches, query_keypoints, train_keypoints):
        return super().matches_to_unimatches(matches, query_keypoints, train_keypoints)
    
    def __repr__(self):
        return "Sift"

class AkazeMatcher(OpenCVAlgorithm):
    """
    Matching Algorithm that uses AKAZE to match keypoints between two images.
    """

    algorithm = cv2.AKAZE_create()

    def __init__(self):
        pass

    def get_matches(self, query_image, train_image, ratio, ratio_not_quantile=False):
        return super().get_matches(query_image, train_image, self.algorithm, ratio, ratio_not_quantile)

    def matches_to_unimatches(self, matches, query_keypoints, train_keypoints):
        return super().matches_to_unimatches(matches, query_keypoints, train_keypoints)
    
    def __repr__(self):
        return "Akaze"