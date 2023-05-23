"""
Wrapper class for pulling data from firebase storage.
"""
import cv2
import subprocess
import os
from glob import glob
import json
import mesh_pb2
from google.protobuf.json_format import MessageToJson
from progressbar import ProgressBar

# run these once:
#--------------------------------------------
# sync firebase storage data with local file system
# subprocess.call(["sh", "sync_firebase_storage.sh"])

# untar all files into correct directories
# subprocess.call(["sh", "untar.sh"])
# --------------------------------------------

class FirebaseDataGatherer:
    def __init__(self):
        pass

    def sort_metadata_jsons(self):
        route_names_path = f"{os.path.dirname(os.path.dirname(__file__))}/image_data_good/NcsslHgGt7OcPalBU90raZOBcqP2/"
        metadata_jsons_path = f"{os.path.dirname(os.path.dirname(__file__))}/image_data_good_logs/NcsslHgGt7OcPalBU90raZOBcqP2/"

        for json_file in os.listdir(metadata_jsons_path):
            if json_file.endswith(".DS_Store"):
                print("!DS_Store")
                continue
            if json_file.endswith("metadata.json"):
                with open(f"{metadata_jsons_path}{json_file}", "r") as f:
                    json_data = json.load(f)
                    try:
                        ARDataDir = json_data["ARLoggerDataDir"].split("/")[1:]
                        route_name = "/".join(ARDataDir)
                        pathdata_json = json_file[:-(len("-0_metadata.json"))]+"_pathdata.json"

                        # move metadata, pathdata jsons to route folder
                        # subprocess.call(            
                        #     [
                        #         "cp",
                        #         f"{metadata_jsons_path}{json_file}",
                        #         f"{metadata_jsons_path}{pathdata_json}",
                        #         f"{route_names_path}{route_name}/",
                        #     ]
                        # )
                    except KeyError:
                        print("No ARLoggerDataDir")
                        continue

        

    def get_sessions_data(self):
        # image_data/wwhzbcAKGZZ7gsVHOfpDoKyWil53/
        # image_data_good/NcsslHgGt7OcPalBU90raZOBcqP2/ Confident water fountain 
        path = f"{os.path.dirname(os.path.dirname(__file__))}/image_data_good/NcsslHgGt7OcPalBU90raZOBcqP2/ Confident water fountain /NcsslHgGt7OcPalBU90raZOBcqP2/"
        sessions_data = []
        for session in os.listdir(path):
            if session.endswith(".DS_Store"):
                print("!DS_Store")
                continue
            session_path = f"{path}{session}/"
            session_data = []
            all_metadata = {}
            for files in os.listdir(session_path):
                if files.endswith("metadata.json"):
                    with open(f"{session_path}{files}", "r") as f:
                        metadata = json.load(f)
                        for key in metadata:
                            all_metadata[key] = metadata[key]
                elif files.endswith("pathdata.json"):
                    with open(f"{session_path}{files}", "r") as f:
                        pathdata = json.load(f)
                        for key in pathdata:
                            all_metadata[key] = pathdata[key]
                elif files.endswith("log.json"):
                    with open(f"{session_path}{files}", "r") as f:
                        logdata = json.load(f)
                        all_metadata["cloudAnchorResolvedTimestamp"] = logdata
            pbar = ProgressBar()
            sorted_frames = []
            for frame in os.listdir(session_path):
                if frame.endswith(".DS_Store"):
                    print("!DS_Store")
                    continue
                sorted_frames.append(glob(f"{session_path}{frame}")[0])
            for frame_path in pbar(sorted(sorted_frames)):
                frame_data = []
                image_data_files = glob(f"{frame_path}/*")
                for file in image_data_files:
                    if file.endswith(".json"):
                        json_file = file
                    elif file.endswith(".jpg"):
                        image_file = file
                    elif file.endswith(".pb"):
                        protobuf_file = file
                
                # read image
                formatted_image = cv2.imread(image_file, 0)
                frame_data.append(formatted_image)

                # read protobuf file
                with open(protobuf_file, 'rb') as f:
                    read_mesh = mesh_pb2.Points()
                    read_mesh.ParseFromString(f.read())
                    protobuf_json = json.loads(MessageToJson(read_mesh))
                    try:
                        protobuf_points = [list(x.values())
                                           for x in protobuf_json['points']]
                        protobuf_conf = protobuf_json['confidences']
                    except KeyError:
                        print("No protobuf data" + str(protobuf_file))
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
                            json_data["timestamp"],
                        ]
                    )
                session_data.append(frame_data)
            sessions_data.append([all_metadata, session_data])
        return sessions_data
