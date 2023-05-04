import os
import pickle
import json


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
            self.query_image_timestamp,
            self.train_image,
            self.train_image_depth_map,
            self.train_image_confidence_map,
            self.train_image_pose,
            self.train_image_intrinsics,
            self.train_image_timestamp,
        ) = bundle_data


class Session:
    """
    A data structure that represents the data about a session. A session is
    a collection of images that were captured from a single Firebase session.

    Instance Attributes:
        bundles (list): A list of bundles that represents the data about pairs
            of images captured in the same session.
    """

    def __init__(self, all_metadata, session_data):
        self.all_metadata = all_metadata
        self.bundles = session_data


class SessionGenerator:
    """
    Handles creating bundle objects from data fetched from the firebase server
    and saving these bundle objects to a pickle file (images/bundles.pkl) where
    they can be utilized in the benchmarking operations.

    Class Atrributes:
        images_path (string): Represents the path of the images folder based
            on the absolute path of the BundleGenerator.py file.

    Instance Attributes:
        sessions (list): A list of session objects.
    """

    sessions_path = f"{os.path.dirname(os.path.dirname(__file__))}/session_data/"

    def __init__(self, sessions_data):
        self.sessions = self.generate_sessions(sessions_data)

    def generate_sessions(self, sessions_data):
        """
        Generates a list of session objects from data fetched from the firebase
        server.

        Arguments:
            sessions_data (list): A list of session lists which contain data about all images
                captured in a single session.

        Returns:
            A list of session objects, each session contains a
            number of bundles.
        """

        sessions_bundle_data = []
        for all_metadata, session_data in sessions_data:
            session_bundle_data = []
            for i in range(0, len(session_data) - 1):
                session_bundle_data.append(
                    Bundle(session_data[i] + session_data[i + 1]))
            sessions_bundle_data.append(
                Session(all_metadata, session_bundle_data))
        return sessions_bundle_data

    def save_sessions(self):
        """
        Saves a list of session objects to a pickle file (session_data/sessions.pkl).
        """
        sessions_file_path = f"{self.sessions_path}sessions.pkl"
        with open(sessions_file_path, "wb") as sessions_file:
            print("saved!")
            pickle.dump(self, sessions_file, pickle.HIGHEST_PROTOCOL)
