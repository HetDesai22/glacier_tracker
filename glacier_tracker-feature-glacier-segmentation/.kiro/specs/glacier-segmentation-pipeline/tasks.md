# Implementation Plan: Glacier Segmentation Pipeline

## Overview

Implement the two-module glacier segmentation pipeline (`src/glims_utils.py` and
`src/glacier_pipeline.py`) that refines NDSI candidate masks into authoritative
glacier extent masks for Samudra Tapu Glacier (2018–2026), plus a full pytest /
Hypothesis test suite. The implementation follows the stage sequence in the design:
acquire → rasterize → intersect → refine → validate → summarize.

## Tasks

- [x] 1. Update environment dependencies
  - Add `geopandas`, `shapely`, `scipy`, `requests`, and `hypothesis` to
    `requirements.txt` with pinned versions compatible with the existing
    `rasterio==1.5.2`, `numpy==2.5.3`, and `matplotlib==3.11.2`.
  - Do NOT modify `src/ndsi_mask.py` or any existing file under
    `data/processed/ndsi/`.
  - _Requirements: 9.1, 9.2_

- [x] 2. Implement `src/glims_utils.py` — GLIMS download and filter
  - [x] 2.1 Implement `download_glims(glims_dir: Path) -> Path`
    - Check for cached `glims_dir / "samudra_tapu_glims.geojson"`; return path
      immediately if it exists (no HTTP call).
    - If absent, perform HTTP GET to the GLIMS WFS endpoint (URL in design).
    - Save raw GeoJSON response to the cache path; return path.
    - Raise `RuntimeError` with URL and manual-placement instructions on HTTP
      failure or non-200 status.
    - _Requirements: 1.1, 1.2, 1.3_
  - [x] 2.2 Implement `load_and_filter_glims(geojson_path: Path) -> gpd.GeoDataFrame`
    - Load GeoJSON with geopandas.
    - Filter rows where `glac_id == 'G077186E32649N'` OR `glac_name` contains
      `'Samudra Tapu'` (case-insensitive).
    - Raise `ValueError` with glacier name and ID if the result is empty.
    - Return the filtered GeoDataFrame.
    - _Requirements: 1.4, 1.5_

- [x] 3. Implement `src/glims_utils.py` — rasterization
  - [x] 3.1 Implement `rasterize_glims(gdf, reference_raster, out_path) -> Path`
    - Open `reference_raster` with rasterio; extract `crs`, `transform`, `width`,
      `height`.
    - Reproject `gdf` from EPSG:4326 to the raster CRS using `gdf.to_crs(raster_crs)`.
    - Burn the reprojected polygon to a uint8 array with
      `rasterio.features.rasterize`.
    - Raise `ValueError` with a CRS-mismatch hint if the result has no pixels == 1.
    - Write output as uint8 GeoTIFF (DEFLATE, nodata=0) to `out_path`; create
      parent dirs as needed.
    - Return `out_path`.
    - _Requirements: 2.1, 2.2, 2.3, 2.5_
  - [ ]* 3.2 Write property test for rasterization grid alignment (Property 3)
    - **Property 3: Rasterization grid alignment**
    - **Validates: Requirements 2.1, 2.2**
    - Generate a random valid polygon in EPSG:4326 and a random reference raster
      profile; assert output width, height, transform, and CRS equal the reference.
    - Tag: `# Feature: glacier-segmentation-pipeline, Property 3`
    - `@settings(max_examples=100)`

- [x] 4. Implement core mask operations in `src/glacier_pipeline.py`
  - [x] 4.1 Implement `intersect_masks(ndsi_mask, glims_mask) -> np.ndarray`
    - Return 255 where `ndsi_mask == 255`, 1 where both inputs == 1, 0 elsewhere.
    - Output dtype uint8.
    - _Requirements: 3.1, 3.2, 3.4_
  - [ ]* 4.2 Write property test for intersection value correctness (Property 4)
    - **Property 4: Intersection output value correctness**
    - **Validates: Requirements 3.1, 3.2, 3.4**
    - Generate random uint8 arrays of the same shape; assert output ∈ {0, 1, 255}
      and all three semantic cases hold.
    - Tag: `# Feature: glacier-segmentation-pipeline, Property 4`
    - `@settings(max_examples=100)`
  - [x] 4.3 Implement `refine_mask(candidate, glims_mask, closing_radius=3, min_area_px=10000) -> np.ndarray`
    - Convert candidate to boolean (True where == 1).
    - If `closing_radius > 0`, apply `scipy.ndimage.binary_closing` with a disk
      kernel, masking result to `glims_mask == 1`.
    - Apply `scipy.ndimage.label`; remove components with pixel count < `min_area_px`.
    - Return uint8 array with only values {0, 1}; log component counts.
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_
  - [ ]* 4.4 Write property test for component size filtering (Property 5)
    - **Property 5: Component size filtering**
    - **Validates: Requirements 4.2**
    - Generate random binary masks and thresholds; assert every retained component
      has pixel count ≥ threshold.
    - Tag: `# Feature: glacier-segmentation-pipeline, Property 5`
    - `@settings(max_examples=100)`
  - [ ]* 4.5 Write property test for GLIMS boundary containment (Property 6)
    - **Property 6: GLIMS boundary containment**
    - **Validates: Requirements 4.4, 4.5**
    - Generate random candidate + GLIMS mask pairs; assert output[glims_mask==0] == 0.
    - Tag: `# Feature: glacier-segmentation-pipeline, Property 6`
    - `@settings(max_examples=100)`
  - [ ]* 4.6 Write property test for final mask binary value set (Property 7)
    - **Property 7: Final mask binary value set**
    - **Validates: Requirements 4.6**
    - Generate random inputs; assert output contains only {0, 1}.
    - Tag: `# Feature: glacier-segmentation-pipeline, Property 7`
    - `@settings(max_examples=100)`

- [x] 5. Implement area calculation and raster I/O helpers in `src/glacier_pipeline.py`
  - [x] 5.1 Implement `compute_area_km2(mask, pixel_res_m=10.0) -> float`
    - Return `np.sum(mask == 1) * pixel_res_m**2 / 1e6`.
    - _Requirements: 7.1_
  - [ ]* 5.2 Write property test for area formula correctness (Property 8)
    - **Property 8: Area formula correctness**
    - **Validates: Requirements 7.1**
    - Generate random uint8 arrays; assert result equals
      `np.sum(arr == 1) * 100.0 / 1_000_000`.
    - Tag: `# Feature: glacier-segmentation-pipeline, Property 8`
    - `@settings(max_examples=100)`
  - [x] 5.3 Implement `save_raster(array, reference_profile, out_path, band_desc="", nodata=255) -> None`
    - Write uint8 GeoTIFF with DEFLATE, copying CRS/transform from
      `reference_profile`, setting nodata and optional band description.
    - Create parent directories as needed.
    - _Requirements: 2.3, 3.3, 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 6. Checkpoint — core logic complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement validation plot and process_year in `src/glacier_pipeline.py`
  - [x] 7.1 Implement `make_validation_plot(year, rgb_path, ndsi_mask, glims_mask, candidate, final, area_km2, out_path) -> None`
    - Use `matplotlib.use("Agg")`.
    - 4-panel figure: (a) true-colour RGB B4/B3/B2 with 2–98% stretch,
      (b) NDSI_Mask with GLIMS boundary contour in red,
      (c) Candidate_Mask, (d) Final_Mask.
    - Title: `f"Samudra Tapu Glacier {year} — {area_km2:.2f} km²"`.
    - Save at 150 dpi; close figure.
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
  - [x] 7.2 Implement `process_year(year, ndsi_dir, raw_dir, glims_dir, out_dirs, glims_gdf, closing_radius, min_area_px) -> dict | None`
    - Return `None` with a logged warning if the NDSI mask file is missing.
    - Orchestrate: rasterize GLIMS → intersect → refine → save outputs →
      validate plot → compute area.
    - Return a stats dict with `year`, `candidate_pixels`, `candidate_area_km2`,
      `final_pixels`, `final_area_km2`.
    - _Requirements: 3.5, 5.1–5.5, 6.1–6.6, 7.1–7.3, 8.5, 8.6_

- [x] 8. Implement CLI and main orchestration in `src/glacier_pipeline.py`
  - Implement `main()` using `argparse` with flags `--year`, `--ndsi-dir`,
    `--min-area-km2`.
  - Default: process years 2018–2026; `--year` restricts to one year.
  - Call `download_glims` + `load_and_filter_glims` once before the year loop.
  - Collect results, write `glacier_area_summary.csv` sorted ascending by year.
  - Exit non-zero if `processed_count == 0`.
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 7.2, 7.3_

- [x] 9. Write unit tests in `tests/test_glacier_pipeline.py`
  - [x] 9.1 Write unit tests for `glims_utils` download behaviour
    - `test_download_creates_file`: mock HTTP GET → file written, path returned.
    - `test_download_uses_cache`: pre-existing file → no HTTP call made.
    - `test_download_error_raises`: mock HTTP 500 → `RuntimeError` containing URL.
    - `test_filter_empty_raises`: GDF with no matching rows → `ValueError`.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - [x] 9.2 Write unit tests for rasterization and alignment
    - `test_rasterize_zero_coverage_raises`: polygon outside raster extent → `ValueError`.
    - `test_alignment_guard`: mismatched CRS/transform between two rasters → `ValueError`.
    - _Requirements: 2.4, 2.5_
  - [x] 9.3 Write unit tests for output format and I/O
    - `test_save_raster_metadata`: written file has correct dtype, nodata, compression,
      and band description.
    - `test_validation_plot_creates_file`: PNG written, no GUI window opened.
    - _Requirements: 5.3, 5.4, 6.1, 6.6_
  - [x] 9.4 Write unit tests for CLI behaviour
    - `test_cli_default_years`: no `--year` arg → all years 2018–2026 attempted.
    - `test_cli_single_year`: `--year 2021` → only 2021 processed.
    - `test_cli_missing_all_years_exits_nonzero`: all NDSI files absent → exit code 1.
    - `test_cli_missing_one_year_continues`: one NDSI file absent → warning logged,
      other years succeed.
    - _Requirements: 8.1, 8.2, 8.5, 8.6, 8.7_

- [x] 10. Write property-based tests in `tests/test_properties.py`
  - [x]* 10.1 Write property test for download idempotence (Property 1)
    - **Property 1: Download idempotence**
    - **Validates: Requirements 1.2**
    - Pre-existing file → no HTTP request, content unchanged across two calls.
    - Tag: `# Feature: glacier-segmentation-pipeline, Property 1`
    - `@settings(max_examples=100)`
  - [x]* 10.2 Write property test for GLIMS filter correctness (Property 2)
    - **Property 2: GLIMS filter correctness**
    - **Validates: Requirements 1.4**
    - Generate random GeoDataFrames with arbitrary IDs/names including occasional
      matching rows; assert output contains only rows with matching `glac_id` or
      `glac_name`.
    - Tag: `# Feature: glacier-segmentation-pipeline, Property 2`
    - `@settings(max_examples=100)`
  - [x]* 10.3 Write property test for CSV completeness and ordering (Property 9)
    - **Property 9: CSV completeness and ordering**
    - **Validates: Requirements 7.2, 7.3**
    - Generate random sets of processed-year result dicts (in random order);
      assert the written CSV has exactly one row per year, correct pixel/area
      values, and rows in strictly ascending year order.
    - Tag: `# Feature: glacier-segmentation-pipeline, Property 9`
    - `@settings(max_examples=100)`

- [ ] 11. Single-year smoke test (year 2020)
  - Run `python src/glacier_pipeline.py --year 2020` against the real data in
    `data/processed/ndsi/` and `data/raw/`.
  - Verify that `data/processed/glims/glims_mask_2020.tif`,
    `data/processed/candidates/glacier_candidate_2020.tif`,
    `data/processed/final_masks/glacier_mask_2020.tif`, and
    `data/processed/validation/validation_2020.png` are created.
  - Verify that `data/processed/glacier_area_summary.csv` contains a row for 2020
    with a positive `final_area_km2`.
  - Fix any issues before proceeding to multi-year processing.
  - _Requirements: 8.1, 8.2_

- [ ] 12. Checkpoint — single-year verified
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. Generalize and verify all years 2018–2026
  - Run `python src/glacier_pipeline.py` (no `--year` flag) to process all nine years.
  - Verify that outputs exist for all years 2018–2026 in `final_masks/`,
    `candidates/`, `glims/`, and `validation/`.
  - Verify that `glacier_area_summary.csv` contains exactly nine rows in ascending
    year order, each with a positive `final_area_km2`.
  - Confirm exit code is 0.
  - _Requirements: 8.1, 8.5, 8.6, 8.7, 7.2, 7.3_

- [ ] 14. Final checkpoint — all years complete
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP.
- Property tests in tasks 3.2, 4.2–4.6, 5.2 are co-located with their implementation
  tasks to catch errors early; tasks 10.1–10.3 consolidate the remaining properties.
- `src/ndsi_mask.py` and all files under `data/processed/ndsi/` must never be modified.
- All raster outputs are uint8, DEFLATE-compressed, EPSG:32643, 10 m resolution.
- The GLIMS GeoJSON is fetched once per pipeline run and cached in `data/glims/`.
