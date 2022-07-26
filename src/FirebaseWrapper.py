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
import mesh_pb2
from google.protobuf.json_format import MessageToJson
from progressbar import ProgressBar

# sync firebase storage data with local file system
# subprocess.call(["sh", "sync_firebase_storage.sh"])


class FirebaseDataGatherer:
    def __init__(self):
        pass
    def get_images_data(self):
        path = f"{os.path.dirname(os.path.dirname(__file__))}/image_data/wwhzbcAKGZZ7gsVHOfpDoKyWil53/"
        sessions_data = []
        for session in os.listdir(path):
            if session.endswith(".DS_Store"):
                print("!DS_Store")
                continue
            session_path = f"{path}{session}/"
            session_data = []
            pbar = ProgressBar()
            for frame in pbar(os.listdir(session_path)):
                frame_data = []
                frame_path = f"{session_path}{frame}"
                image_data_files = glob(f"{frame_path}/*")
                for file in image_data_files:
                    if file.endswith(".json"):
                        json_file = file
                    elif file.endswith(".jpg"):
                        image_file = file
                    elif file.endswith(".pb"):
                        protobuf_file = file
                formatted_image = cv2.imread(image_file, 0)
                # formatted_image = cv2.rotate(image, ROTATE_90_CLOCKWISE)
                frame_data.append(formatted_image)

                # read protobuf file
                with open(protobuf_file, 'rb') as f:
                    read_mesh = mesh_pb2.Points()
                    read_mesh.ParseFromString(f.read())
                    protobuf_json = json.loads(MessageToJson(read_mesh))
                    try:
                        protobuf_points = [list(x.values()) for x in protobuf_json['points']]
                        protobuf_conf = protobuf_json['confidences']
                    except KeyError:
                        print("No protobuf data" + str(protobuf_file)[-3:])
                        continue

                # read json
                with open(json_file, "r") as f:
                    json_data = json.load(f)
                    frame_data.extend(
                        [
                            # TODO convert to numpy arrays
                            protobuf_points,
                            protobuf_conf,
                            json_data["pose"],
                            json_data["intrinsics"],
                        ]
                    )
                session_data.append(frame_data)
            sessions_data.append(session_data)
        return sessions_data
