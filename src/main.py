from FirebaseWrapper import FirebaseDataGatherer
from SessionGenerator import SessionGenerator
from Benchmarker import Benchmarker
from MatchingAlgorithm import OrbMatcher, SiftMatcher, AkazeMatcher

# fireBaseDataGatherer = FirebaseDataGatherer()
# images_data = fireBaseDataGatherer.get_images_data()

# sessionGenerator = SessionGenerator(images_data)
# sessionGenerator.save_sessions()

benchmarker = Benchmarker([OrbMatcher(), SiftMatcher(), AkazeMatcher()])
benchmarker.benchmark()

