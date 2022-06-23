from FirebaseWrapper import FirebaseDataGatherer
from BundleGenerator import BundleGenerator, Bundle
from Benchmarker import Benchmarker

fireBaseDataGatherer = FirebaseDataGatherer()
images_data = fireBaseDataGatherer.get_images_data()

bundleGenerator = BundleGenerator(images_data)
bundleGenerator.save_bundles()

benchmarker = Benchmarker("A")
bundles = benchmarker.get_bundles()
