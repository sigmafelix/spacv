"""Label-based spatial and spatiotemporal cross-validation."""
import numbers
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_random_state
from .utils import convert_geodataframe, geometry_to_2d


class STCV(BaseCrossValidator):
    """Configure spatial, temporal, or location-time cell holdouts.

    Parameters
    ----------
    mode : str, default='lblo'
        lolo: unique location; loto: unique time; lolto: unique location-time
        pair; lblo: spatial block; lbto: temporal block; lblto: spatial and
        temporal block pair; loo: individual row; random: shuffled rows.
    n_splits : int, default=5
        Number of blocks for lblo/lbto or folds for random.
    spatial_splits, temporal_splits : int, default=5
        Block counts for lblto.
    time_col : str or None
        Temporal field. None detects time, datetime, date, timestamp, or ts
        (case insensitive) when a temporal mode is requested.
    blocks : str or None
        Existing spatial block column, used instead of clustering.
    random_state : int, RandomState or None
        Seed for clustering or random folds.

    Temporal blocks partition sorted unique times into contiguous groups of
    nearly equal size; gaps and subdaily timestamps are preserved. Combined
    modes hold out observed cells, with all other cells available for training.
    They do not enforce future-only prediction or unseen locations AND times.
    Pass a GeoDataFrame as X or as groups alongside numeric model features.
    Fold labels are zero-based, and input frames are never modified.
    """

    def __init__(self, mode='lblo', n_splits=5, spatial_splits=5,
                 temporal_splits=5, time_col=None, blocks=None, random_state=None):
        self.mode = mode
        self.n_splits = n_splits
        self.spatial_splits = spatial_splits
        self.temporal_splits = temporal_splits
        self.time_col = time_col
        self.blocks = blocks
        self.random_state = random_state

    @staticmethod
    def _count(value, available, name):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not 2 <= value <= available:
            raise ValueError("{} must be an integer between 2 and {}.".format(name, available))
        return value

    def fold_labels(self, X):
        """Return one integer fold label per observation in original order."""
        modes = ('lolo', 'loto', 'lolto', 'lblo', 'lbto', 'lblto', 'loo', 'random')
        if self.mode not in modes:
            raise ValueError("mode must be one of {}.".format(modes))
        frame = convert_geodataframe(X)
        if len(frame) < 2:
            raise ValueError("At least two observations are required.")
        temporal = self.mode in ('loto', 'lolto', 'lbto', 'lblto')
        if temporal:
            column = self.time_col
            if column is None:
                column = next((c for name in ('time', 'datetime', 'date', 'timestamp', 'ts')
                               for c in frame.columns if str(c).lower() == name), None)
            if column is None or column not in frame.columns:
                raise ValueError("A temporal field is required; specify time_col.")
            times = frame[column]
            if times.isna().any():
                raise ValueError("Temporal values must not be missing.")
            if not (pd.api.types.is_numeric_dtype(times) or pd.api.types.is_datetime64_any_dtype(times)):
                raise ValueError("Temporal field must be numeric or datetime; convert dates with pandas.to_datetime.")
            if pd.api.types.is_numeric_dtype(times) and not np.isfinite(times.to_numpy()).all():
                raise ValueError("Temporal values must be finite.")
            time_labels, unique_times = pd.factorize(times, sort=True)
            if self.mode in ('lbto', 'lblto'):
                count = self.n_splits if self.mode == 'lbto' else self.temporal_splits
                self._count(count, len(unique_times), 'temporal splits')
                mapping = np.repeat(np.arange(count), [len(a) for a in np.array_split(unique_times, count)])
                time_labels = mapping[time_labels]
        if self.mode in ('lolo', 'lolto', 'lblo', 'lblto'):
            if frame.geometry.isna().any() or frame.geometry.is_empty.any():
                raise ValueError("Geometries must not be missing or empty.")
            coords = geometry_to_2d(frame.geometry)
            if not np.isfinite(coords).all():
                raise ValueError("Coordinates must be finite.")
            locations, spatial_labels = np.unique(coords, axis=0, return_inverse=True)
            if self.mode in ('lblo', 'lblto'):
                if self.blocks is not None:
                    if self.blocks not in frame or frame[self.blocks].isna().any():
                        raise ValueError("blocks must name a column without missing values.")
                    spatial_labels = pd.factorize(frame[self.blocks], sort=True)[0]
                    assignments = pd.DataFrame({'location': np.unique(coords, axis=0, return_inverse=True)[1],
                                                'block': spatial_labels})
                    if (assignments.groupby('location').block.nunique() > 1).any():
                        raise ValueError("Repeated locations must have the same spatial block.")
                else:
                    count = self.n_splits if self.mode == 'lblo' else self.spatial_splits
                    self._count(count, len(locations), 'spatial splits')
                    labels = MiniBatchKMeans(n_clusters=count, random_state=self.random_state,
                                            n_init=10).fit_predict(locations)
                    if len(np.unique(labels)) != count:
                        raise ValueError("Clustering produced fewer spatial blocks than requested.")
                    spatial_labels = labels[spatial_labels]
        if self.mode in ('lolo', 'lblo'):
            labels = spatial_labels
        elif self.mode in ('loto', 'lbto'):
            labels = time_labels
        elif self.mode in ('lolto', 'lblto'):
            labels = pd.factorize(pd.MultiIndex.from_arrays([spatial_labels, time_labels]), sort=True)[0]
        elif self.mode == 'loo':
            labels = np.arange(len(frame))
        else:
            count = self._count(self.n_splits, len(frame), 'n_splits')
            labels = np.empty(len(frame), dtype=int)
            labels[check_random_state(self.random_state).permutation(len(frame))] = np.arange(len(frame)) % count
        if len(np.unique(labels)) < 2:
            raise ValueError("Configuration must produce at least two nonempty folds.")
        return np.asarray(labels, dtype=int)

    def split(self, X, y=None, groups=None):
        frame = X if groups is None else groups
        if (groups is not None and len(X) != len(frame)) or (y is not None and len(y) != len(frame)):
            raise ValueError("X, y, and groups must have the same length.")
        labels = self.fold_labels(frame)
        order = np.argsort(labels, kind='stable')
        boundaries = np.flatnonzero(np.diff(labels[order])) + 1
        for test in np.split(order, boundaries):
            mask = np.ones(len(labels), dtype=bool)
            mask[test] = False
            yield np.flatnonzero(mask), np.sort(test)

    def get_n_splits(self, X=None, y=None, groups=None):
        if X is None and groups is None:
            raise ValueError("X or groups is required to count observed folds.")
        return len(np.unique(self.fold_labels(X if groups is None else groups)))
