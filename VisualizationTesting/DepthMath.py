import sys
import os
import cv2
import numpy as np
# setting path to src files
sys.path.append(f"{os.path.dirname(os.path.dirname(__file__))}/src/")
from Benchmarker import Benchmarker
from MatchingAlgorithm import SiftMatcher


session = Benchmarker("A").sessions[0]
bundle = session.bundles[0]
new_sift_matcher = SiftMatcher(bundle.query_image[0], bundle.train_image[0])
new_sift_matcher.draw_matches()