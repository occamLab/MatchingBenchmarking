import os
from FirebaseWrapper import FirebaseDataGatherer
from SessionGenerator import SessionGenerator
from Benchmarker import Benchmarker
from MatchingAlgorithm import OrbMatcher, SiftMatcher, AkazeMatcher
import benchmarkers.CloudBenchmark as cbm
import benchmarkers.PairBenchmark as pbm

# run these lines once per dataset in Firebase 🙏🏼
############################################
# fireBaseDataGatherer = FirebaseDataGatherer()
# fireBaseDataGatherer.sort_metadata_jsons()
# sessions_data = fireBaseDataGatherer.get_sessions_data()

# sessionGenerator = SessionGenerator(sessions_data)
# sessionGenerator.save_sessions()
############################################

algorithms = [OrbMatcher()]
values = [0.5]
# cbm.run_benchmark(algorithms, values)
pbm.run_benchmark(algorithms, values)
