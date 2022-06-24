import os
from glob import glob

path = "C:/Users/meftimie/Desktop/Grind/MatchingBenchmarking/image_data/tqO5JKPW1yN66yjjuYiw8cQvvh72/"

for session in os.listdir(path):
    session_path = f"{path}{session}/"
    for frame in os.listdir(session_path):
        frame_path = f"{session_path}{frame}"
        print(glob(f"{frame_path}/*"))

    print("new session")
