import sys
import os
# setting path to src files
sys.path.append(f"{os.path.dirname(os.path.dirname(__file__))}/src/")
from Benchmarker import Benchmarker


session = Benchmarker("A").sessions[0]
bundle = session.bundles[0]

print(bundle)
