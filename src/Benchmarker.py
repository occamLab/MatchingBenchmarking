from abc import ABC, abstractmethod
import os
import pickle
from BundleGenerator import Bundle

# from MatchingAlgorithm import MatchingAlgorithm


class Benchmarker:
    def __init__(self, algorithms):
        self.bundles = self.get_bundles()
        self.algorithms = algorithms

    def get_bundles(self):
        bundles_path = (
            f"{os.path.dirname(os.path.dirname(__file__))}\\images\\bundles.pkl"
        )
        with open(bundles_path, "rb") as bundles_file:
            bundles = pickle.load(bundles_file)
        return bundles

    def benchmark(self):
        for algorithm in self.algorithms:
            for bundle in self.bundles:
                pass
