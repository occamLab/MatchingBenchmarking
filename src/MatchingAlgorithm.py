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
    def __init__(self, query_image, train_image):
        # TODO remove images from constructor parameters, don't need to create a new object for each pair.
        self.query_image = query_image
        self.train_image = train_image

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
    def draw_matches(self):
        """
        Visualizes the matches between the query and train images.

        Returns:
            (None): uses matplotlib to show the matches on the images.
        """
        pass

    @abstractmethod
    def get_uni_matches(self):
        """
        Creates a list of UNIMatch objects representing the matches between the query and train images.

        Returns:
            uniMatches (list): A list of UNIMatch objects.
        """
        pass


# class SuperGlueMatcher(MatchingAlgorithm):
#     """
#     Matching Algorithm that uses SuperGlue to match keypoints between two images.
#     """
#     def __init__(self, query_image, train_image):
#         super().__init__(query_image, train_image)
#         (
#             self.matches,
#             self.query_keypoints,
#             self.train_keypoints,
#         ) = get_superglue_matches(self.query_image, self.train_image)

#     def get_matches(self):
#         return get_superglue_matches(self.query_image, self.train_image)

#     def draw_matches(self):
#         return draw_superglue_matches()

#     def get_uni_matches(self):
#         # Create list of UNIMatch objects
#         uniMatches = [
#             UNIMatch(
#                 self.matches[i],
#                 self.matches[i + 1],
#                 self.matches[i + 2],
#                 self.matches[i + 3],
#             )
#             for i in range(0, len(self.matches), 4)
#         ]

#         return uniMatches


class OrbMatcher(MatchingAlgorithm):
    """
    Matching Algorithm that uses ORB to match keypoints between two images.
    """
    def __init__(self, query_image, train_image):
        super().__init__(query_image, train_image)
        self.matches, self.query_keypoints, self.train_keypoints = self.get_matches()

    def get_matches(self):
        # Initiate ORB algorithm
        orb = cv2.ORB_create()

        # Find the keypoints and descriptors with ORB
        query_keypoints, query_descriptors = orb.detectAndCompute(
            self.query_image, None
        )
        train_keypoints, train_descriptors = orb.detectAndCompute(
            self.train_image, None
        )

        # Get matches from keypoints and descriptors
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(query_descriptors, train_descriptors)

        # TODO: Add ratio test for ORB algorithm
        return (
            sorted(matches, key=lambda x: x.distance),
            query_keypoints,
            train_keypoints,
        )

    def draw_matches(self):
        # Draw first 10 matches
        img3 = cv2.drawMatches(
            self.query_image,
            self.query_keypoints,
            self.train_image,
            self.train_keypoints,
            self.matches[:50],
            self.train_image,
            flags=2,
        )
        plt.imshow(img3)
        plt.show(img3)

    def get_uni_matches(self):
        # Creates
        coord_array = []
        for match in self.matches:
            x1, y1 = self.query_keypoints[match.queryIdx].pt
            x2, y2 = self.train_keypoints[match.trainIdx].pt
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
    def __init__(self, query_image, train_image):
        super().__init__(query_image, train_image)
        (
            self.filtered_matches,
            self.query_keypoints,
            self.train_keypoints,
        ) = self.get_matches()

    def get_matches(self):
        # Initiate SIFT algorithm
        sift = cv2.SIFT_create()

        # Find the keypoints and descriptors with SIFT
        query_keypoints, query_descriptors = sift.detectAndCompute(
            self.query_image, None
        )
        train_keypoints, train_descriptors = sift.detectAndCompute(
            self.train_image, None
        )

        # Get matches from keypoints and descriptors
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(query_descriptors, train_descriptors, k=2)

        # Apply ratio test (currently set to 1 so it does nothing)
        filtered_matches = [
            match
            for match, nearest_neighbor in matches
            if match.distance < 1 * nearest_neighbor.distance
        ]
        return filtered_matches, query_keypoints, train_keypoints

    def draw_matches(self):
        # Draw first 10 matches
        img3 = cv2.drawMatchesKnn(
            self.query_image,
            self.query_keypoints,
            self.train_image,
            self.train_keypoints,
            self.filtered_matches,
            None,
            flags=2,
        )
        plt.imshow(img3), plt.show()

    def get_uni_matches(self):

        coord_array = []
        for match in self.filtered_matches:
            x1, y1 = self.query_keypoints[match.queryIdx].pt
            x2, y2 = self.train_keypoints[match.trainIdx].pt
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
