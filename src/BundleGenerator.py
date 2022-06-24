import os
import pickle


class Bundle:
    """
    A data structure that represents the data about two images captured from a
    RealityKit session. Both images (query and train) have a depth map,
    confidence map, pose, and intrinsics associated with them. Using this data,
    it is possible to know how a pixel in the query image maps to the train
    image (and vice versa).

    Instance Attributes:
        query_image: A grayscale and cropped JPEG image that represents the
            first image of the query and train image pair.
        query_image_depth_map: The depth map of the query_image. (Need Type)
        query_image_confidence_map: The confidence map of the query_image. (Need Type)
        query_image_pose: The pose of the query_image. (Need Type)
        query_image_intrinsics: The intrinsics of the camera when the query
            image was captured in the RealityKit session. (Need Type)
        train_image: A grayscale and cropped JPEG image that represents the
            first image of the query and train image pair.
        train_image_depth_map: The depth map of the train_image. (Need Type)
        train_image_confidence_map: The confidence map of the train_image. (Need Type)
        train_image_pose: The pose of the train_image. (Need Type)
        train_image_intrinsics: The intrinsics of the camera when the train
            image was captured in the RealityKit session. (Need Type)

    """

    def __init__(self, bundle_data):
        (
            self.query_image,
            self.query_image_depth_map,
            self.query_image_confidence_map,
            self.query_image_pose,
            self.query_image_intrinsics,
            self.train_image,
            self.train_image_depth_map,
            self.train_image_confidence_map,
            self.train_image_pose,
            self.train_image_intrinsics,
        ) = bundle_data


class BundleGenerator:
    """
    Handles creating bundle objects from data fetched from the firebase server
    and saving these bundle objects to a pickle file (images/bundles.pkl) where
    they can be utilized in the benchmarking operations.

    Class Atrributes:
        images_path (string): Represents the path of the images folder based
            on the absolute path of the BundleGenerator.py file.

    Instance Attributes:
        bundles (list): A list of bundle objects.
    """

    images_path = f"{os.path.dirname(os.path.dirname(__file__))}\\bundle_data\\"

    def __init__(self, images_data):
        self.bundles = self.generate_bundles(images_data)

    def generate_bundles(self, sessions_data):
        """
        Generates a list bundle objects from data fetched from the firebase
        server.

        Arguments:
            sessions_data (list): A list of lists which contain data about an image.

        Returns:
            A list of lists, each list represents a session and contains a
            number of bundles.
        """
        sessions_bundle_data = []
        for images_data in sessions_data:
            session_bundle_data = []
            for i in range(0, len(images_data) - 1):
                session_bundle_data.append(Bundle(images_data[i] + images_data[i + 1]))
            sessions_bundle_data.append(session_bundle_data)
        print(sessions_bundle_data)
        return sessions_bundle_data

    def save_bundles(self):
        """
        Saves a list of bundle objects to a pickle file (images/bundles.pkl).

        Arguments:
            bundles (list): A list of bundle objects.
        """
        bundles_file_path = f"{self.images_path}bundles.pkl"
        with open(bundles_file_path, "wb") as bundles_file:
            pickle.dump(self.bundles, bundles_file, pickle.HIGHEST_PROTOCOL)
