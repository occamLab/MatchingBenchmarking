from abc import ABC, abstractmethod
import cv2
import matplotlib.pyplot as plt
from UniMatch import UNIMatch

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


class SuperGlueMatcher(MatchingAlgorithm):
    """
    Matching Algorithm that uses SuperGlue to match keypoints between two images.
    """

    def __init__(self, query_image, train_image):
        pass

    def get_matches(self, query_image, train_image):
        return self.get_uni_matches(get_superglue_matches(query_image, train_image))

    def draw_matches(self):
        return draw_superglue_matches()

    def get_uni_matches(self, matches):
        # Create list of UNIMatch objects
        uniMatches = [
            UNIMatch(
                matches[i + 1],
                matches[i],
                matches[i + 2],
                matches[i + 3],
            )
            for i in range(0, len(matches), 4)
        ]

        return uniMatches


class OrbMatcher(MatchingAlgorithm):
    """
    Matching Algorithm that uses ORB to match keypoints between two images.
    """

    def __init__(self):
        pass

    def get_matches(self, query_image, train_image):
        # Initiate ORB algorithm
        orb = cv2.ORB_create()

        # Find the keypoints and descriptors with ORB
        query_keypoints, query_descriptors = orb.detectAndCompute(query_image, None)
        train_keypoints, train_descriptors = orb.detectAndCompute(train_image, None)

        # Get matches from keypoints and descriptors
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(query_descriptors, train_descriptors)

        # TODO: Add ratio test for ORB algorithm
        return self.matches_to_unimatches(
            sorted(matches, key=lambda x: x.distance),
            query_keypoints,
            train_keypoints,
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


class SiftMatcher(MatchingAlgorithm):
    """
    Matching Algorithm that uses SIFT to match keypoints between two images.
    """

    def __init__(self):
        pass

    def get_matches(self, query_image, train_image):
        # Initiate SIFT algorithm
        sift = cv2.SIFT_create()

        # Find the keypoints and descriptors with SIFT
        query_keypoints, query_descriptors = sift.detectAndCompute(query_image, None)
        train_keypoints, train_descriptors = sift.detectAndCompute(train_image, None)

        # Get matches from keypoints and descriptors
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(query_descriptors, train_descriptors, k=2)

        # Apply ratio test (currently set to 1 so it does nothing)
        filtered_matches = [
            match
            for match, nearest_neighbor in matches
            if match.distance < 1 * nearest_neighbor.distance
        ]
        return self.matches_to_unimatches(
            filtered_matches, query_keypoints, train_keypoints
        )

    def matches_to_unimatches(self, matches, query_keypoints, train_keypoints):

        coord_array = []
        for match in matches:
            x1, y1 = query_keypoints[match.queryIdx].pt
            x2, y2 = train_keypoints[match.trainIdx].pt
            coord_array.extend([x1, y1, x2, y2])

        # Create list of UNIMatch objects
        uniMatches = [
            UNIMatch(
                coord_array[i + 1],
                coord_array[i],
                coord_array[i + 2],
                coord_array[i + 3],
            )
            for i in range(0, len(coord_array), 4)
        ]

        return uniMatches
