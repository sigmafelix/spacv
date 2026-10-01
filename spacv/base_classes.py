from abc import abstractmethod
import numpy as np
from sklearn.model_selection import BaseCrossValidator
from .utils import convert_geoseries, convert_geodataframe


class BaseSpatialCV(BaseCrossValidator):
    """Base class for partitioning-based spatial cross-validation."""

    def split(self, X, y=None, groups=None):
        """Yield positional train/test indices; groups may hold geometries.

        Pass geometries as X, or pass a GeoSeries/GeoDataFrame as groups
        when X contains only model features.
        """
        XYs = convert_geoseries(X if groups is None else groups)
        if y is not None and len(y) != len(XYs):
            raise ValueError("y and geometries must have the same length.")
        if groups is not None and len(X) != len(XYs):
            raise ValueError("X and groups must have the same length.")
        if not np.isfinite(self.buffer_radius) or self.buffer_radius < 0:
            raise ValueError("buffer_radius must be finite and nonnegative.")
        for test_indices, train_excluded in self._iter_test_indices(XYs):
            mask = np.ones(len(XYs), dtype=bool)
            mask[test_indices] = False
            mask[train_excluded] = False
            train_index = np.flatnonzero(mask)
            if not len(train_index):
                raise ValueError("Training set is empty. Try lowering buffer_radius.")
            yield train_index, np.asarray(test_indices, dtype=int)

    def _remove_buffered_indices(self, XYs, test_indices, buffer_radius, geometry_buffer):
        if buffer_radius > 0:
            # Query the full index to preserve original observation positions.
            geometry_buffer = convert_geodataframe(geometry_buffer)
            excluded = XYs.sindex.query(geometry_buffer.geometry, predicate='intersects')[1]
            return test_indices, np.unique(excluded)
        return test_indices, np.empty(0, dtype=int)

    @abstractmethod
    def _iter_test_indices(self, XYs):
        """Yield test positions and buffered positions to exclude."""

    def get_n_splits(self, X=None, y=None, groups=None):
        if X is not None or groups is not None:
            return sum(1 for _ in self.split(X, y, groups))
        if not hasattr(self, 'n_splits'):
            raise ValueError("X or groups is required to count observed folds.")
        return self.n_splits
