from typing import List
import math


def dot(x: List[float], y: List[float]) -> float:
    return sum((a * b for a, b in zip(x, y)))


def norm(x: List[float]) -> float:
    return math.sqrt(sum((a * a for a in x)))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    na = norm(a)
    nb = norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)


def euclidean_distance(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return float("inf")
    return math.sqrt(sum(((x - y) ** 2 for x, y in zip(a, b))))
