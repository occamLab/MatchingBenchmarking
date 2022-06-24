from FirebaseWrapper import FirebaseDataGatherer
from SessionGenerator import SessionGenerator
from Benchmarker import Benchmarker

fireBaseDataGatherer = FirebaseDataGatherer()
images_data = fireBaseDataGatherer.get_images_data()

sessionGenerator = SessionGenerator(images_data)
sessionGenerator.save_sessions()

benchmarker = Benchmarker("A")
sessions = benchmarker.get_sessions()

for session in sessions:
    for bundle in session.bundles:
        print(bundle.query_image_intrinsics)
