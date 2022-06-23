"""
Wrapper class for pulling data from firebase storage.
"""
import firebase_admin
from firebase_admin import credentials, storage
import numpy as np
import cv2
from ast import literal_eval

cred = credentials.Certificate("src/serviceAccountKey.json")
app = firebase_admin.initialize_app(
    cred, {"storageBucket": "depthbenchmarking.appspot.com"}
)


class FirebaseDataGatherer:
    def __init__(self):
        self.bucket = storage.bucket()

    def get_images_data(self):
        directory = "visual_alignment_benchmarking/tqO5JKPW1yN66yjjuYiw8cQvvh72/FF972FAC-329A-4834-8156-EF19ADCB9598"
        images_data = []
        i = 1
        while True:
            try:
                image_blob = self.bucket.get_blob(f"{directory}/000{i}/frame.jpg")
                arr = np.frombuffer(image_blob.download_as_string(), np.uint8)
                image = cv2.imdecode(arr, cv2.COLOR_BGR2GRAY)
                json_blob = self.bucket.get_blob(
                    f"{directory}/000{i}/framemetadata.json"
                )
                # decodes bytes to valid json
                json_data = literal_eval(json_blob.download_as_bytes().decode("utf8"))

                images_data.append(
                    [
                        image,
                        json_data["depthData"],
                        json_data["confData"],
                        json_data["pose"],
                        json_data["intrinsics"],
                    ]
                )
            except AttributeError:
                break
            i += 1

        return images_data
