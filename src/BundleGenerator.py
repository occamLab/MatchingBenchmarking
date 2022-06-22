import os
import pickle


class Bundle:
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
        ) = bundle_data


class BundleGenerator:

    images_path = f"{os.path.dirname(os.path.dirname(__file__))}\images\\"

    def __init__(self):
        self.bundles = self.generate_bundles()

    def generate_bundles(self):
        bundles_data = zip(os.listdir(self.images_path))
        return [Bundle(bundle_data) for bundle_data in bundles_data]

    def save_bundles(self, bundles):
        bundles_file_path = f"{self.images_path}bundles.pkl"
        with open(bundles_file_path, "wb") as bundles_file:
            pickle.dump(bundles, bundles_file, pickle.HIGHEST_PROTOCOL)


# Predict where the point on the train image should be based off of camera pose and depth data, compare that to match given by matching algorithm
