from typing import Set, List
from math import sqrt

def jaccard_distance(a: Set[str], b: Set[str]) -> float:

    return len(a.intersection(b)) / len(a.union(b))

def cosine_similarity(a: List[float], b: List[float]) -> float:

    return _dot(a, b) / (_norm(a) * _norm(b))

def _dot(a: List[float], b: List[float]) -> float:

    return sum((i * j for i, j in zip(a, b)))

def _norm(a: List[float]) -> float:

    return sqrt(_dot(a, a))