"""
Wrapper class for pulling data from firebase storage.
"""
from cv2 import ROTATE_90_CLOCKWISE
import firebase_admin
from firebase_admin import credentials, storage
import cv2
import subprocess
import os
from glob import glob
import json

# sync firebase storage data with local file system
# subprocess.call(["sh", "sync_firebase_storage.sh"])

cred = credentials.Certificate("serviceAccountKey.json")
app = firebase_admin.initialize_app(
    cred, {"storageBucket": "depthbenchmarking.appspot.com"}
)


class FirebaseDataGatherer:
    def __init__(self):
        self.bucket = storage.bucket()

    def get_images_data(self):
        path = f"{os.path.dirname(os.path.dirname(__file__))}/image_data/50lyHYG52VTfIB2OWMmCBA5elaC3/"
        sessions_data = []
        for session in os.listdir(path):
            session_path = f"{path}{session}/"
            session_data = []
            for frame in os.listdir(session_path):
                frame_data = []
                frame_path = f"{session_path}{frame}"
                image_data_files = glob(f"{frame_path}/*")
                if image_data_files[0][-3:] == "jpg":
                    image_file, json_file = image_data_files
                else:
                    json_file, image_file = image_data_files
                formatted_image = cv2.imread(image_file, 0)
                # formatted_image = cv2.rotate(image, ROTATE_90_CLOCKWISE)
                frame_data.append(formatted_image)

                # read json
                with open(json_file, "r") as f:
                    json_data = json.load(f)
                    frame_data.extend(
                        [
                            # TODO convert to numpy arrays
                            json_data["depthData"],
                            json_data["confData"],
                            json_data["pose"],
                            json_data["intrinsics"],
                        ]
                    )
                session_data.append(frame_data)
            sessions_data.append(session_data)
        return sessions_data
