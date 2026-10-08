"""GLIMS glacier inventory utilities for the Samudra Tapu glacier segmentation pipeline.

Provides:
  - download_glims(glims_dir)    : fetch/cache the GLIMS WFS GeoJSON for Samudra Tapu
  - load_and_filter_glims(path)  : load and filter the cached GeoJSON
  - rasterize_glims(gdf, ref, out): reproject and rasterize the GLIMS polygon
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GLIMS_CACHE_NAME = "samudra_tapu_glims.geojson"
GLIMS_GLACIER_ID = "G077425E32510N"
GLIMS_GLACIER_NAME = "Samudra Tapu"

GLIMS_WFS_URL = (
    "https://www.glims.org/geoserver/GLIMS/ows"
    "?service=WFS"
    "&version=1.0.0"
    "&request=GetFeature"
    "&typeName=GLIMS:GLIMS_Glacier_Outlines"
    "&outputFormat=application/json"
    f"&CQL_FILTER=glac_id='{GLIMS_GLACIER_ID}'"
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_glims(glims_dir: Path) -> Path:
    """Download (or use cached) GLIMS WFS data for Samudra Tapu Glacier.

    Parameters
    ----------
    glims_dir : Path
        Directory where the GeoJSON cache should be stored / already exists.

    Returns
    -------
    Path
        Absolute path to the cached ``samudra_tapu_glims.geojson`` file.

    Raises
    ------
    RuntimeError
        If the HTTP request fails or returns a non-200 status, with the URL
        and instructions for manual file placement.
    """
    glims_dir = Path(glims_dir)
    cache_path = glims_dir / GLIMS_CACHE_NAME

    # Requirement 1.2: return cached file immediately, no HTTP call
    if cache_path.exists():
        return cache_path

    # Requirement 1.1: fetch from GLIMS WFS endpoint
    try:
        response = requests.get(GLIMS_WFS_URL, timeout=60)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to reach the GLIMS WFS endpoint.\n"
            f"URL: {GLIMS_WFS_URL}\n"
            f"Error: {exc}\n\n"
            f"To resolve this manually:\n"
            f"  1. Download the GLIMS GeoJSON for glacier '{GLIMS_GLACIER_ID}' "
            f"({GLIMS_GLACIER_NAME}) from https://www.glims.org/\n"
            f"  2. Save it as: {cache_path}"
        ) from exc

    # Requirement 1.3: raise descriptive RuntimeError on non-200 status
    if response.status_code != 200:
        raise RuntimeError(
            f"GLIMS WFS request returned HTTP {response.status_code}.\n"
            f"URL: {GLIMS_WFS_URL}\n\n"
            f"To resolve this manually:\n"
            f"  1. Download the GLIMS GeoJSON for glacier '{GLIMS_GLACIER_ID}' "
            f"({GLIMS_GLACIER_NAME}) from https://www.glims.org/\n"
            f"  2. Save it as: {cache_path}"
        )

    glims_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(response.text, encoding="utf-8")
    return cache_path


def load_and_filter_glims(geojson_path: Path) -> gpd.GeoDataFrame:
    """Load the cached GLIMS GeoJSON and filter to Samudra Tapu Glacier.

    Parameters
    ----------
    geojson_path : Path
        Path to the cached ``samudra_tapu_glims.geojson`` file.

    Returns
    -------
    gpd.GeoDataFrame
        Filtered GeoDataFrame containing only Samudra Tapu rows.

    Raises
    ------
    ValueError
        If no rows match the glacier ID or name.
    """
    gdf = gpd.read_file(geojson_path)

    # Match on glac_id (exact) OR glac_name (case-insensitive substring)
    id_mask = gdf.get("glac_id", pd.Series(dtype=str)) == GLIMS_GLACIER_ID
    name_mask = (
        gdf.get("glac_name", pd.Series(dtype=str))
        .fillna("")
        .str.contains(GLIMS_GLACIER_NAME, case=False, na=False)
    )
    filtered = gdf[id_mask | name_mask].copy()

    if filtered.empty:
        raise ValueError(
            f"No GLIMS polygons found for glacier '{GLIMS_GLACIER_NAME}' "
            f"(ID: {GLIMS_GLACIER_ID}) in {geojson_path}. "
            f"Check that the GeoJSON contains the correct glacier record."
        )

    return filtered


def rasterize_glims(
    gdf: gpd.GeoDataFrame,
    reference_raster: Path,
    out_path: Path,
) -> Path:
    """Reproject and rasterize the GLIMS polygon to match a reference raster.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GLIMS GeoDataFrame (assumed source CRS EPSG:4326).
    reference_raster : Path
        Path to the reference raster (e.g. the NDSI mask for a given year).
    out_path : Path
        Destination path for the output GLIMS_Mask GeoTIFF.

    Returns
    -------
    Path
        ``out_path`` after the file has been written.

    Raises
    ------
    ValueError
        If the rasterized output contains no pixels equal to 1 (CRS mismatch
        or geometry does not overlap the raster extent).
    """
    with rasterio.open(reference_raster) as src:
        raster_crs = src.crs
        transform = src.transform
        width = src.width
        height = src.height

    # Reproject from source CRS (EPSG:4326) to raster CRS
    gdf_reprojected = gdf.to_crs(raster_crs)

    # Burn polygon to uint8 array (1 = inside, 0 = outside)
    shapes = [(geom, 1) for geom in gdf_reprojected.geometry if geom is not None]
    burned = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    if burned.max() == 0:
        raise ValueError(
            f"GLIMS rasterization produced an all-zero mask. "
            f"The GLIMS polygon may not overlap the reference raster extent, "
            f"or there may be a CRS mismatch. "
            f"Reference raster CRS: {raster_crs}, "
            f"GLIMS source CRS: {gdf.crs}"
        )

    # Write output GeoTIFF
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": width,
        "height": height,
        "count": 1,
        "crs": raster_crs,
        "transform": transform,
        "compress": "deflate",
        "nodata": 0,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(burned, 1)

    return out_path
