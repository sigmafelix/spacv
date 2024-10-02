import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, MultiPolygon

def geometry_to_2d(geometry):
    """
    Convert geometries to 2D coordinates suitable for clustering algorithms.
    For Points, use the x and y coordinates directly.
    For Polygons, use the centroid's x and y coordinates.
    
    Parameters
    ----------
    geometry : GeoSeries or array-like
        Series of shapely geometries (Points or Polygons).
    
    Returns
    -------
    coords : ndarray
        Numpy array of shape (n_samples, 2) containing x and y coordinates.
    """
    def get_xy(geom):
        if geom.geom_type == 'Point':
            return (geom.x, geom.y)
        elif geom.geom_type in ['Polygon', 'MultiPolygon']:
            centroid = geom.centroid
            return (centroid.x, centroid.y)
        else:
            # Handle other geometry types if needed
            raise ValueError(f"Unsupported geometry type: {geom.geom_type}")
    return np.array([get_xy(g) for g in geometry])

def convert_geoseries(XYs):
    """
    Convert various types of inputs to a GeoSeries.
    If input is a GeoDataFrame with Polygon geometries, convert to centroids.
    
    Parameters
    ----------
    XYs : GeoDataFrame, ndarray, or other geometry types
        Input geometries.
    
    Returns
    -------
    XYs : GeoSeries
        GeoSeries containing Point geometries.
    """
    if isinstance(XYs, gpd.GeoDataFrame):
        if any(XYs.geom_type.isin(['Polygon', 'MultiPolygon'])):
            XYs = XYs.geometry.centroid
        else:
            XYs = XYs.geometry
    elif isinstance(XYs, np.ndarray):
        XYs = gpd.GeoSeries(gpd.points_from_xy(XYs[:, 0], XYs[:, 1]))
    elif isinstance(XYs, (Point, Polygon, LineString, MultiPolygon)):
        XYs = gpd.GeoSeries([XYs])
    elif isinstance(XYs, list):
        XYs = gpd.GeoSeries(XYs)
    return XYs.reset_index(drop=True)

def convert_geodataframe(XYs):
    """
    Convert various types of inputs to a GeoDataFrame.
    
    Parameters
    ----------
    XYs : GeoSeries, ndarray, or other geometry types
        Input geometries.
    
    Returns
    -------
    XYs : GeoDataFrame
        GeoDataFrame containing geometries.
    """
    if isinstance(XYs, gpd.GeoSeries):
        XYs = gpd.GeoDataFrame(geometry=XYs.reset_index(drop=True))
    elif isinstance(XYs, np.ndarray):
        XYs = gpd.GeoDataFrame(geometry=gpd.points_from_xy(XYs[:, 0], XYs[:, 1]))
    elif isinstance(XYs, (Point, Polygon, LineString, MultiPolygon)):
        XYs = gpd.GeoDataFrame(geometry=[XYs])
    elif isinstance(XYs, list):
        XYs = gpd.GeoDataFrame(geometry=XYs)
    return XYs.reset_index(drop=True)

def convert_numpy(X):
    """
    Convert various data types to a numpy array.
    
    Parameters
    ----------
    X : DataFrame, Series, GeoDataFrame, or array-like
        Input data.
    
    Returns
    -------
    X : ndarray
        Numpy array representation of the input data.
    """
    if isinstance(X, (pd.DataFrame, pd.Series, gpd.GeoDataFrame)):
        return X.values
    else:
        return np.array(X)

def load_custom_polygon(custom_poly):
    """
    Load custom polygons from a file or accept GeoDataFrame/GeoSeries.
    
    Parameters
    ----------
    custom_poly : str, GeoDataFrame, or GeoSeries
        File path to the custom polygon shapefile or a GeoDataFrame/GeoSeries.
    
    Returns
    -------
    custom_poly : GeoDataFrame
        GeoDataFrame containing the custom polygons.
    """
    if isinstance(custom_poly, (gpd.GeoDataFrame, gpd.GeoSeries)):
        return custom_poly.reset_index(drop=True)
    elif isinstance(custom_poly, str):
        return gpd.read_file(custom_poly).reset_index(drop=True)
    else:
        raise ValueError("custom_polygons must be a file path or a GeoDataFrame/GeoSeries.")
