from abc import ABC, abstractmethod

class Benchmarcer:
    def __init__(self, algorithms):
        self.algorithms = algorithms

    @abstractmethod
    def score(self, query):
        """
        Returns a score for the type of query on
        given list of algorithms.
        """
        pass

