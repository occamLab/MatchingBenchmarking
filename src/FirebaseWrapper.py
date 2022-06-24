"""
Wrapper class for pulling data from firebase storage.
"""
import firebase_admin
from firebase_admin import credentials, storage
import cv2
import subprocess
import os
import json

# sync firebase storage data with local file system
subprocess.call(['sh', "sync_firebase_storage.sh"])

cred = credentials.Certificate("serviceAccountKey.json")
app = firebase_admin.initialize_app(
    cred, {"storageBucket": "depthbenchmarking.appspot.com"}
)

class FirebaseDataGatherer:
    def __init__(self):
        self.bucket = storage.bucket()

    def get_images_data(self):
        directory = f"{os.path.dirname(os.path.dirname(__file__))}/image_data/"
        images_data = []
        for root, _, files in os.walk(directory):
            image_data = []
            for data_file in files:
                # ignore hidden files
                if data_file[0] != ".":
                    # get file path
                    data = os.path.join(root, data_file)
                    if data_file[-3:] == "jpg":
                        # read image
                        image_data.append([cv2.imread(data)])
                    else:
                        # read json
                        with open(data, "r") as f:
                            json_data = json.load(f)
                            image_data.extend(
                                [
                                    json_data["depthData"],
                                    json_data["confData"],
                                    json_data["pose"],
                                    json_data["intrinsics"],
                                ]
                            )
            if len(image_data) > 0:
                images_data.append(image_data)
        return images_data