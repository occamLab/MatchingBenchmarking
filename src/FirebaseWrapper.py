"""
Wrapper class for pulling data from firebase storage.
"""
import firebase_admin
from firebase_admin import credentials, storage
import cv2
import subprocess
import os
from glob import glob
import json

# sync firebase storage data with local file system
subprocess.call(["sh", "sync_firebase_storage.sh"])

cred = credentials.Certificate("serviceAccountKey.json")
app = firebase_admin.initialize_app(
    cred, {"storageBucket": "depthbenchmarking.appspot.com"}
)


class FirebaseDataGatherer:
    def __init__(self):
        self.bucket = storage.bucket()

    def get_images_data(self):
        path = f"{os.path.dirname(os.path.dirname(__file__))}/image_data/tqO5JKPW1yN66yjjuYiw8cQvvh72/"
        sessions_data = []
        for session in os.listdir(path):
            session_path = f"{path}{session}/"
            session_data = []
            for frame in os.listdir(session_path):
                frame_data = []
                frame_path = f"{session_path}{frame}"
                image, json_file = glob(f"{frame_path}/*")
                frame_data.append([cv2.imread(image)])
                # read json
                with open(json_file, "r") as f:
                    json_data = json.load(f)
                    frame_data.extend(
                        [
                            json_data["depthData"],
                            json_data["confData"],
                            json_data["pose"],
                            json_data["intrinsics"],
                        ]
                    )
                session_data.append(frame_data)
            sessions_data.append(session_data)
        return sessions_data