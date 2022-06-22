import os
import pickle
from tkinter import image_names

class BundleGenerator:

    def __init__(self):
        self.path = self.get_images_path()
    
    def get_images_path(self):
        return f"{os.path.dirname(os.path.dirname(__file__))}\images\\"

    def generate_bundles(self):
        # query_path
        bundles = []
        zip(os.listdir(self.path))

    
    def save_bundles(self):
        bundles_file_path = f"{self.path}bundles.pkl"
        with open(bundles_file_path, 'wb') as bundles_file:
            pickle.dump(Bundle(), bundles_file, pickle.HIGHEST_PROTOCOL)

class Bundle:

    def __init__(self, query_image, train_image):
        self.query_image = "a"
        self.train_image = "b"
        # depth_maps of both iamges
        # confidence_maps of bot image
        # poses of both images 
        # intrinsics of both images

# Predict where the point on the train image should be based off of camera pose and depth data, compare that to match given by matching algorithm

print(BundleGenerator().save_bundles())
