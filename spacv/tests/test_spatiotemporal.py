import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from spacv import STCV, SKCV, HBLOCK, RepeatedSKCV, UserDefinedSCV
from shapely.geometry import box


@pytest.fixture
def panel():
    sites = np.repeat(np.arange(6), 4)
    return gpd.GeoDataFrame(
        {'date': np.tile(pd.to_datetime(['2020-01-01', '2020-01-01 01:00',
                                       '2020-03-01', '2022-01-01'], format='mixed'), 6),
         'block': sites // 2, 'value': np.arange(24)},
        geometry=gpd.points_from_xy(sites * 100, sites % 2 * 100),
        index=np.repeat(np.arange(12) + 100, 2), crs='EPSG:3857')


@pytest.mark.parametrize('mode,count', [('lolo', 6), ('loto', 4), ('lolto', 24),
                                       ('lblo', 3), ('lbto', 2), ('lblto', 6),
                                       ('loo', 24), ('random', 3)])
def test_modes(panel, mode, count):
    original = panel.copy()
    cv = STCV(mode=mode, n_splits=2 if mode == 'lbto' else 3,
              spatial_splits=3, temporal_splits=2, random_state=3)
    labels = cv.fold_labels(panel)
    splits = list(cv.split(panel))
    assert cv.get_n_splits(panel) == len(splits) == count
    np.testing.assert_array_equal(np.sort(np.concatenate([te for _, te in splits])), np.arange(len(panel)))
    for tr, te in splits:
        assert not np.intersect1d(tr, te).size
        assert len(tr) + len(te) == len(panel)
        assert len(np.unique(labels[te])) == 1
    pd.testing.assert_frame_equal(panel, original)


def test_temporal_blocks(panel):
    labels = STCV('lbto', n_splits=2).fold_labels(panel)
    np.testing.assert_array_equal(labels[:4], [0, 0, 1, 1])
    numeric = panel.assign(date=np.tile([0, .01, 100, 5000], 6))
    np.testing.assert_array_equal(STCV('lbto', n_splits=2).fold_labels(numeric), labels)


def test_location_sampling_invariance(panel):
    cv = STCV('lblo', n_splits=3, random_state=1)
    labels = cv.fold_labels(panel)
    expanded = pd.concat([panel, panel.iloc[:4]] * 2)
    np.testing.assert_array_equal(cv.fold_labels(expanded)[:len(panel)], labels)
    for site in range(6):
        assert len(np.unique(labels[site*4:site*4+4])) == 1


def test_custom_blocks_and_cells(panel):
    cv = STCV('lblto', blocks='block', temporal_splits=2)
    labels = cv.fold_labels(panel)
    assert len(np.unique(labels)) == 6
    assert labels[0] == labels[1] == labels[4]
    assert labels[0] != labels[2]
    inconsistent = panel.copy()
    inconsistent.iloc[0, inconsistent.columns.get_loc('block')] = 99
    with pytest.raises(ValueError, match='Repeated locations'):
        cv.fold_labels(inconsistent)


def test_sklearn_integration(panel):
    X = panel[['value']].to_numpy()
    y = np.arange(len(panel)) * 2 + 1
    cv = STCV('lblto', blocks='block', temporal_splits=2)
    scores = cross_val_score(LinearRegression(), X, y, cv=cv, groups=panel)
    np.testing.assert_allclose(scores, 1)
    search = GridSearchCV(LinearRegression(), {'fit_intercept': [True, False]}, cv=cv)
    search.fit(X, y, groups=panel)
    assert search.best_params_['fit_intercept']


@pytest.mark.parametrize('change,kwargs', [
    ({'date': pd.NaT}, {'mode': 'loto'}),
    ({'date': 'bad'}, {'mode': 'lbto'}),
    ({'date': np.inf}, {'mode': 'loto'}),
    ({}, {'mode': 'invalid'}),
    ({}, {'mode': 'lbto', 'n_splits': 5}),
    ({}, {'mode': 'lblo', 'n_splits': 7}),
    ({}, {'mode': 'loto', 'time_col': 'missing'}),
])
def test_validation(panel, change, kwargs):
    with pytest.raises(ValueError):
        STCV(**kwargs).fold_labels(panel.assign(**change))


def test_spatial_without_time(panel):
    spatial = panel.drop(columns='date')
    assert len(list(STCV('lblo', n_splits=3).split(spatial))) == 3
    with pytest.raises(ValueError, match='temporal field'):
        list(STCV('loto').split(spatial))


def test_buffer_positions():
    frame = gpd.GeoDataFrame(geometry=gpd.points_from_xy([0, 5, 6, 20], [0, 0, 0, 0]),
                             index=[9, 9, 100, 101])
    splits = list(SKCV(n_splits=4, buffer_radius=2).split(frame))
    np.testing.assert_array_equal(splits[1][0], [0, 3])
    np.testing.assert_array_equal(splits[2][0], [0, 3])


def test_repeated_and_legacy_sklearn(panel):
    cv = RepeatedSKCV(n_splits=3, n_repeats=2, random_state=5)
    assert cv.get_n_splits() == 6
    first, second = list(cv.split(panel)), list(cv.split(panel))
    for (tr1, te1), (tr2, te2) in zip(first, second):
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(te1, te2)
    scores = cross_val_score(LinearRegression(), panel[['value']].to_numpy(),
                             panel.value.to_numpy(), cv=SKCV(3, random_state=4), groups=panel)
    assert len(scores) == 3


def test_user_defined_and_grouped_buffer():
    frame = gpd.GeoDataFrame(geometry=gpd.points_from_xy([1, 2, 11, 12, 21, 22], [1]*6))
    polygons = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 2), box(10, 0, 20, 2), box(20, 0, 30, 2)])
    original = polygons.copy()
    cv = UserDefinedSCV(polygons)
    assert cv.get_n_splits(frame) == 3
    pd.testing.assert_frame_equal(polygons, original)
    cv = HBLOCK(tiles_x=3, tiles_y=1, method='systematic', buffer_radius=.1)
    assert cv.get_n_splits(frame) == len(list(cv.split(frame)))


def test_polygons_and_temporal_ties():
    frame = gpd.GeoDataFrame({'time': [0, 0, 1, 1]},
                             geometry=[box(i*10, 0, i*10+2, 2) for i in range(4)],
                             index=[3, 5, 7, 9])
    cv = SKCV(n_splits=4, buffer_radius=1)
    for train, test in cv.split(frame):
        assert len(train) == 3 and len(test) == 1
    assert len(np.unique(STCV('lolo').fold_labels(frame))) == 4
    np.testing.assert_array_equal(STCV('loto').fold_labels(frame), [0, 0, 1, 1])


def test_sparse_cells(panel):
    sparse = panel.iloc[[0, 1, 8, 9, 22, 23]]
    cv = STCV('lblto', blocks='block', temporal_splits=2)
    assert cv.get_n_splits(sparse) == 3


def test_random_preserves_global_rng(panel):
    np.random.seed(12)
    before = np.random.get_state()
    STCV('random', random_state=5).fold_labels(panel)
    list(HBLOCK(tiles_x=3, tiles_y=2, method='random', n_groups=2,
                random_state=4).split(panel))
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
