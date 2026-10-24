import csv
import os
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from hypothesis import given, settings, strategies as st
from shapely.geometry import Point

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hypothesis import HealthCheck
from glims_utils import download_glims, load_and_filter_glims, GLIMS_CACHE_NAME, GLIMS_GLACIER_ID, GLIMS_GLACIER_NAME
import glacier_pipeline

# Feature: glacier-segmentation-pipeline, Property 1
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.text())
def test_download_idempotence(tmp_path: Path, file_content: str):
    """
    Property 1: Download idempotence
    Validates: Requirements 1.2
    """
    cache_path = tmp_path / GLIMS_CACHE_NAME
    cache_path.write_bytes(file_content.encode("utf-8"))
    
    with patch("glims_utils.requests.get") as mock_get:
        out_path = download_glims(tmp_path)
        
        mock_get.assert_not_called()
        assert out_path == cache_path
        assert out_path.read_bytes() == file_content.encode("utf-8")

# Feature: glacier-segmentation-pipeline, Property 2
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.lists(st.tuples(st.text(), st.text()), max_size=20))
def test_glims_filter_correctness(tmp_path: Path, rows_data):
    """
    Property 2: GLIMS filter correctness
    Validates: Requirements 1.4
    """
    data = [{"glac_id": gid, "glac_name": gname, "geometry": Point(0, 0)} for gid, gname in rows_data]
    
    # Inject matching rows to ensure they are tested
    data.append({"glac_id": GLIMS_GLACIER_ID, "glac_name": "Random Name", "geometry": Point(0, 0)})
    data.append({"glac_id": "Random ID", "glac_name": f"Prefix {GLIMS_GLACIER_NAME.upper()} Suffix", "geometry": Point(0, 0)})
    
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    
    geojson_path = tmp_path / "test.geojson"
    gdf.to_file(geojson_path, driver="GeoJSON")
    
    filtered_gdf = load_and_filter_glims(geojson_path)
    
    for _, row in filtered_gdf.iterrows():
        match_id = row.get("glac_id") == GLIMS_GLACIER_ID
        name = row.get("glac_name")
        match_name = GLIMS_GLACIER_NAME.lower() in str(name).lower() if pd.notna(name) else False
        assert match_id or match_name

# Feature: glacier-segmentation-pipeline, Property 9
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    st.lists(
        st.fixed_dictionaries({
            "year": st.integers(min_value=2018, max_value=2026),
            "candidate_pixels": st.integers(min_value=0),
            "candidate_area_km2": st.floats(min_value=0, allow_nan=False, allow_infinity=False),
            "final_pixels": st.integers(min_value=0),
            "final_area_km2": st.floats(min_value=0, allow_nan=False, allow_infinity=False),
        }),
        unique_by=lambda x: x["year"],
        min_size=1,
        max_size=9
    )
)
def test_csv_completeness_and_ordering(tmp_path: Path, results):
    """
    Property 9: CSV completeness and ordering
    Validates: Requirements 7.2, 7.3
    """
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        results_dict = {r["year"]: r for r in results}
        
        def mock_process_year(year, *args, **kwargs):
            return results_dict.get(year, None)
            
        with patch("glacier_pipeline.process_year", side_effect=mock_process_year), \
             patch("glacier_pipeline.download_glims", return_value=tmp_path / "dummy.geojson"), \
             patch("glacier_pipeline.load_and_filter_glims", return_value=gpd.GeoDataFrame()), \
             patch("sys.argv", ["glacier_pipeline.py"]):
             
             glacier_pipeline.main()
             
        summary_path = Path("data/processed/glacier_area_summary.csv")
        assert summary_path.exists()
        
        with open(summary_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            
        assert len(reader) == len(results)
        
        # Check ascending year order
        years = [int(row["year"]) for row in reader]
        assert years == sorted(years)
        
        # Check correct values
        for row, expected in zip(reader, sorted(results, key=lambda x: x["year"])):
            assert int(row["year"]) == expected["year"]
            assert int(row["candidate_pixels"]) == expected["candidate_pixels"]
            assert float(row["candidate_area_km2"]) == expected["candidate_area_km2"]
            assert int(row["final_pixels"]) == expected["final_pixels"]
            assert float(row["final_area_km2"]) == expected["final_area_km2"]
    finally:
        os.chdir(old_cwd)
