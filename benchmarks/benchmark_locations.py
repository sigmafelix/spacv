"""Run with python benchmarks/benchmark_locations.py from the repository root.

Compare clustering all observations with clustering unique locations and
broadcasting, including deduplication cost. Reports median times; no timing
threshold is imposed because machine load and panel structure affect results.
"""
from time import perf_counter
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def run(coords, deduplicate):
    start = perf_counter()
    if deduplicate:
        locations, inverse = np.unique(coords, axis=0, return_inverse=True)
    else:
        locations = coords
    labels = MiniBatchKMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(locations)
    if deduplicate:
        labels = labels[inverse]
    return perf_counter() - start


if __name__ == '__main__':
    rng = np.random.RandomState(42)
    locations = rng.normal(size=(1000, 2))
    coords = np.repeat(locations, 100, axis=0)
    for deduplicate in (False, True):
        times = [run(coords, deduplicate) for _ in range(5)]
        print('{}: {:.4f}s median (100,000 rows, 1,000 sites)'.format(
            'unique locations' if deduplicate else 'all observations', np.median(times)))
