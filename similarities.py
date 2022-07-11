from typing import Set, List
from math import sqrt

def jaccard_distance(a: Set[str], b: Set[str]) -> float:
    """Calculates jaccard distance of two sets of strings."""

    return len(a.intersection(b)) / len(a.union(b))

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculates cosine similarity of two vectors denoted as lists of floats."""

    return _dot(a, b) / (_norm(a) * _norm(b))

# calculate dot product of two vectors
def _dot(a: List[float], b: List[float]) -> float:

    return sum((i * j for i, j in zip(a, b)))

# calculate L2 norm of vector
def _norm(a: List[float]) -> float:

    return sqrt(_dot(a, a))