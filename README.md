# `spacv`: spatial cross-validation in Python

`spacv` is a small Python 3 (3.9 and above) package for cross-validation of models
that assess generalization performance to datasets with spatial dependence. `spacv` provides
a familiar sklearn-like API to expose a suite of tools useful for points-based spatial prediction tasks.
See the notebook `spacv_guide.ipynb` for usage. For a complete, executable tour
of all supported CV methods with 2.5D fold visualizations, see
[`spacv_guide_v2.ipynb`](spacv_guide_v2.ipynb).

<p align="center">
<img src="demo_viz_buffer.gif" width="300" height="250"/>
</p>

## Dependencies

* `numpy`
* `matplotlib`
* `pandas`
* `geopandas`
* `shapely`
* `scikit-learn`
* `scipy`

## Installation and usage

To install use pip:

    $ pip install spacv

Then build quick spatial cross-validation workflows with `sklearn` as:

```python
import spacv
import geopandas as gpd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

df = gpd.read_file('data/baltim.geojson')

XYs = df['geometry']
X = df[['NROOM', 'BMENT', 'NBATH', 'PRICE', 'LOTSZ', 'SQFT']]
y = df['PATIO']

# Build fold indices as a generator
skcv = spacv.SKCV(n_splits=4, buffer_radius=10).split(XYs)

svc = SVC()

cross_val_score(svc,       # Model 
                X,         # Features
                y,         # Labels
                cv = skcv) # Fold indices
```

## Spatial and spatiotemporal configuration

`STCV` computes a fold label for every observation and yields positional indices
using the scikit-learn cross-validator interface. Spatial modes work without a
temporal field. Temporal modes accept numeric or pandas datetime columns; use
`pd.to_datetime` to convert date strings first. Set `time_col` explicitly, or
allow detection of `time`, `datetime`, `date`, `timestamp`, or `ts`, in that
priority order (case insensitive). No temporal column is used implicitly by
spatial modes.

```python
from spacv import STCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# df is a GeoDataFrame with geometry, timestamp, features, and target.
cv = STCV(mode='lblto', spatial_splits=5, temporal_splits=4,
          time_col='timestamp', random_state=42)
labels = cv.fold_labels(df)  # original row order; df is not modified
scores = cross_val_score(LinearRegression(), df[['feature']], df['target'],
                         cv=cv, groups=df)
# Alternatively, pass cv=cv.split(df) to sklearn, as with existing spacv APIs.
```

| Mode | Test fold | Configuration |
| --- | --- | --- |
| `lolo` | All observations at one location | No temporal field required |
| `lblo` | One cluster of unique locations | `n_splits`, or `blocks='column_name'` |
| `loto` | All observations at one time | Optional `time_col` |
| `lbto` | One contiguous group of unique times | `n_splits`, optional `time_col` |
| `lolto` | One location-time pair, including duplicate rows | Optional `time_col` |
| `lblto` | One spatial-block / temporal-block cell | `spatial_splits`, `temporal_splits`; optional `blocks` |
| `loo` | One observation | No temporal field required |
| `random` | A shuffled group of observations | `n_splits`, `random_state` |

Temporal blocks divide sorted distinct times into nearly equal groups without
splitting ties. They preserve irregular gaps and subdaily resolution. Only
occupied cells produce folds, so `lblto` yields at most
`spatial_splits * temporal_splits` folds. Use `cv.get_n_splits(df)` to count
observed folds. A configuration with fewer than two nonempty folds is rejected.
Missing temporal values and empty/missing spatial geometries are rejected.

Combined modes train on the complement of the held-out cell. Locations and times
in that cell can occur in other training cells. These modes assess missing-cell
prediction; they do not enforce forecasting or simultaneous unseen-location and
unseen-time generalization. `lolo` holds out locations across all their times;
`loto` holds out times across all locations.

Locations are exact coordinate pairs (polygon centroids for clustering).
Repeated locations are clustered once and their labels broadcast to all rows,
so unequal temporal sampling does not weight the spatial partition. Snap or
round jittered station coordinates before splitting. A supplied block column
must assign the same block to every observation at a repeated location.
Spatial clustering uses coordinate units directly; use an appropriate projected
CRS for distance-based spatial partitions and legacy spatial buffers.

Existing `HBLOCK`, `SKCV`, `RepeatedSKCV`, and `UserDefinedSCV` remain available
for geometric grids, custom polygons, and spatial exclusion buffers. `STCV`
does not add buffer zones. Existing splitters also accept `y` and spatial data
through `groups`; using a precomputed split iterable remains supported.
