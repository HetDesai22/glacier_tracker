import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from glims_utils import download_glims, load_and_filter_glims, rasterize_glims
from glacier_pipeline import process_year, save_raster, make_validation_plot, main

# ---------------------------------------------------------------------------
# 9.1 glims_utils download behaviour
# ---------------------------------------------------------------------------

@patch("glims_utils.requests.get")
def test_download_creates_file(mock_get, tmp_path):
    # mock HTTP GET -> file written, path returned
    mock_get.return_value.status_code = 200
    mock_get.return_value.text = '{"type": "FeatureCollection"}'
    
    glims_dir = tmp_path / "glims"
    result = download_glims(glims_dir)
    
    assert result.exists()
    assert result.read_text() == '{"type": "FeatureCollection"}'
    mock_get.assert_called_once()


@patch("glims_utils.requests.get")
def test_download_uses_cache(mock_get, tmp_path):
    # pre-existing file -> no HTTP call made
    glims_dir = tmp_path / "glims"
    glims_dir.mkdir()
    cache_file = glims_dir / "samudra_tapu_glims.geojson"
    cache_file.write_text("cached")
    
    result = download_glims(glims_dir)
    assert result == cache_file
    mock_get.assert_not_called()


@patch("glims_utils.requests.get")
def test_download_error_raises(mock_get, tmp_path):
    # mock HTTP 500 -> RuntimeError containing URL
    mock_get.return_value.status_code = 500
    
    with pytest.raises(RuntimeError) as exc:
        download_glims(tmp_path)
    
    assert "https://www.glims.org" in str(exc.value)


def test_filter_empty_raises(tmp_path):
    # GDF with no matching rows -> ValueError
    gdf = gpd.GeoDataFrame({
        "glac_id": ["G123456E12345N"],
        "glac_name": ["Other Glacier"]
    }, geometry=[None])
    
    geojson_path = tmp_path / "test.geojson"
    gdf.to_file(geojson_path, driver="GeoJSON")
    
    with pytest.raises(ValueError, match="No GLIMS polygons found"):
        load_and_filter_glims(geojson_path)

# ---------------------------------------------------------------------------
# 9.2 rasterization and alignment
# ---------------------------------------------------------------------------

def test_rasterize_zero_coverage_raises(tmp_path):
    # polygon outside raster extent -> ValueError
    from shapely.geometry import Polygon
    
    # Polygon far away
    gdf = gpd.GeoDataFrame(geometry=[Polygon([(0,0), (1,0), (1,1), (0,1)])], crs="EPSG:4326")
    
    ref_raster = tmp_path / "ref.tif"
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": 10,
        "height": 10,
        "count": 1,
        "crs": "EPSG:32643",
        "transform": rasterio.transform.from_origin(100000, 100000, 10, 10)
    }
    with rasterio.open(ref_raster, "w", **profile) as dst:
        dst.write(np.zeros((10, 10), dtype=np.uint8), 1)
        
    out_path = tmp_path / "out.tif"
    with pytest.raises(ValueError, match="all-zero mask"):
        rasterize_glims(gdf, ref_raster, out_path)

def test_alignment_guard(tmp_path):
    # mismatched CRS/transform between two rasters -> ValueError in process_year
    ndsi_dir = tmp_path / "ndsi"
    ndsi_dir.mkdir()
    ndsi_file = ndsi_dir / "Samudra_Tapu_2020_Scientific_mask.tif"
    
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "Samudra_Tapu_2020_Scientific.tif").touch()
    
    # Create NDSI file with one CRS
    profile = {
        "driver": "GTiff", "dtype": "uint8", "width": 10, "height": 10,
        "count": 1, "crs": "EPSG:32643",
        "transform": rasterio.transform.from_origin(0, 0, 10, 10)
    }
    with rasterio.open(ndsi_file, "w", **profile) as dst:
        dst.write(np.ones((10, 10), dtype=np.uint8), 1)

    out_dirs = {
        "glims": tmp_path / "glims_out",
        "candidates": tmp_path / "candidates",
        "final_masks": tmp_path / "final",
        "validation": tmp_path / "val"
    }
    for d in out_dirs.values():
        d.mkdir()

    # Create GLIMS file with different CRS directly to simulate mismatch after rasterize_glims
    # We will mock rasterize_glims so it creates a mismatched file
    def mock_rasterize(gdf, ref, out_path):
        bad_profile = profile.copy()
        bad_profile["crs"] = "EPSG:4326"
        with rasterio.open(out_path, "w", **bad_profile) as dst:
            dst.write(np.ones((10, 10), dtype=np.uint8), 1)
            
    with patch("glims_utils.rasterize_glims", side_effect=mock_rasterize):
        with pytest.raises(ValueError, match="Alignment error"):
            process_year(2020, ndsi_dir, raw_dir, tmp_path, out_dirs, None, 0, 0)

# ---------------------------------------------------------------------------
# 9.3 output format and I/O
# ---------------------------------------------------------------------------

def test_save_raster_metadata(tmp_path):
    out_path = tmp_path / "out.tif"
    profile = {
        "driver": "GTiff", "crs": "EPSG:32643",
        "transform": rasterio.transform.from_origin(0, 0, 10, 10)
    }
    arr = np.ones((5, 5), dtype=np.uint8)
    save_raster(arr, profile, out_path, band_desc="test_desc", nodata=99)
    
    with rasterio.open(out_path) as src:
        assert src.meta["dtype"] == "uint8"
        assert src.meta["nodata"] == 99
        assert src.profile.get("compress", "none") == "deflate" or src.compression.name == "deflate"
        assert src.descriptions[0] == "test_desc"

def test_validation_plot_creates_file(tmp_path):
    # Setup dummy arrays and RGB tif
    rgb_path = tmp_path / "rgb.tif"
    profile = {
        "driver": "GTiff", "dtype": "float32", "width": 5, "height": 5,
        "count": 3, "crs": "EPSG:32643",
        "transform": rasterio.transform.from_origin(0, 0, 10, 10)
    }
    with rasterio.open(rgb_path, "w", **profile) as dst:
        dst.write(np.random.rand(3, 5, 5).astype(np.float32))

    out_path = tmp_path / "val.png"
    arr = np.zeros((5, 5), dtype=np.uint8)
    make_validation_plot(2020, rgb_path, arr, arr, arr, arr, 1.0, out_path)
    
    assert out_path.exists()

# ---------------------------------------------------------------------------
# 9.4 CLI behaviour
# ---------------------------------------------------------------------------

@patch("glacier_pipeline.process_year")
@patch("glacier_pipeline.download_glims")
@patch("glacier_pipeline.load_and_filter_glims")
@patch("sys.argv", ["glacier_pipeline.py"])
def test_cli_default_years(mock_load, mock_download, mock_process, tmp_path):
    # no --year arg -> all years 2018-2026 attempted
    mock_process.return_value = {"year": 2018, "candidate_pixels": 0, "candidate_area_km2": 0.0, "final_pixels": 0, "final_area_km2": 0.0}
    
    # We need to change cwd or mock Path so it can write summary
    with patch("glacier_pipeline.Path") as mock_path:
        # Just mock open to avoid file writing issues
        with patch("builtins.open"):
            main()
    
    assert mock_process.call_count == 9
    years_called = [call.kwargs["year"] for call in mock_process.call_args_list]
    assert years_called == list(range(2018, 2027))

@patch("glacier_pipeline.process_year")
@patch("glacier_pipeline.download_glims")
@patch("glacier_pipeline.load_and_filter_glims")
@patch("sys.argv", ["glacier_pipeline.py", "--year", "2021"])
def test_cli_single_year(mock_load, mock_download, mock_process):
    # --year 2021 -> only 2021 processed
    mock_process.return_value = {"year": 2021, "candidate_pixels": 0, "candidate_area_km2": 0.0, "final_pixels": 0, "final_area_km2": 0.0}
    with patch("builtins.open"):
        main()
    
    assert mock_process.call_count == 1
    assert mock_process.call_args.kwargs["year"] == 2021

@patch("glacier_pipeline.process_year")
@patch("glacier_pipeline.download_glims")
@patch("glacier_pipeline.load_and_filter_glims")
@patch("sys.argv", ["glacier_pipeline.py"])
def test_cli_missing_all_years_exits_nonzero(mock_load, mock_download, mock_process):
    # all NDSI files absent -> exit code 1
    mock_process.return_value = None
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1

@patch("glacier_pipeline.download_glims")
@patch("glacier_pipeline.load_and_filter_glims")
@patch("sys.argv", ["glacier_pipeline.py", "--year", "2020", "--ndsi-dir", "invalid_dir"])
def test_cli_missing_one_year_continues(mock_load, mock_download, caplog):
    # If run through main with missing NDSI, it returns None.
    # To test one year missing continues, we need two years: one missing, one present.
    with patch("sys.argv", ["glacier_pipeline.py"]):
        with patch("glacier_pipeline.process_year") as mock_process:
            mock_process.side_effect = [None, {"year": 2019, "candidate_pixels": 0, "candidate_area_km2": 0, "final_pixels": 0, "final_area_km2": 0}] + [None]*7
            with patch("builtins.open"):
                main()
            assert mock_process.call_count == 9

