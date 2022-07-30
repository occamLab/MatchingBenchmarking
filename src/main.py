from FirebaseWrapper import FirebaseDataGatherer
from SessionGenerator import SessionGenerator
from Benchmarker import Benchmarker
from MatchingAlgorithm import OrbMatcher, SiftMatcher, AkazeMatcher
# fireBaseDataGatherer = FirebaseDataGatherer()
# images_data = fireBaseDataGatherer.get_images_data()

# sessionGenerator = SessionGenerator(images_data)
# sessionGenerator.save_sessions()

benchmarker = Benchmarker([OrbMatcher(), SiftMatcher()], [0.25, 0.45])
# benchmarker.benchmark()
benchmarker.cross_benchmark(benchmarker.sessions[1], benchmarker.sessions[2], [OrbMatcher(), SiftMatcher()], [0.25, 0.45])
