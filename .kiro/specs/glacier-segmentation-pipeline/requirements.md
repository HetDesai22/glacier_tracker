# Requirements Document

## Introduction

This feature implements the glacier segmentation pipeline for Samudra Tapu Glacier
starting from the existing NDSI candidate mask. The pipeline uses the GLIMS glacier
inventory as a spatial reference to constrain NDSI detections to the known glacier
boundary, applies image-based refinement to remove false positives (noise, seasonal
snow, cloud artefacts), and produces a final binary glacier mask and visual validation
output for each year (2018–2026). Area is calculated as glacier_pixels × 100 m² /
1,000,000 (since each pixel is 10 m × 10 m = 100 m²). No deep learning, climate
forecasting, or quantitative accuracy metrics are introduced.

The Sentinel-2 preprocessing and NDSI candidate-mask generation already exist in
`src/ndsi_mask.py` and must not be modified.

---

## Glossary

- **Pipeline**: The end-to-end glacier segmentation workflow implemented by this feature.
- **NDSI_Mask**: The existing uint8 binary raster (`_mask.tif`) produced by `ndsi_mask.py`,
  where 1 = NDSI-positive (snow/ice candidate), 0 = non-snow/ice, 255 = nodata.
- **NDSI_Raster**: The existing float32 continuous NDSI raster (`_ndsi.tif`) produced by
  `ndsi_mask.py`.
- **GLIMS**: The Global Land Ice Measurements from Space glacier inventory. Provides
  authoritative polygon boundaries for named glaciers in geographic coordinates (EPSG:4326).
- **GLIMS_Mask**: A uint8 binary raster reprojected and rasterized to the NDSI grid where
  1 = inside GLIMS boundary, 0 = outside.
- **Candidate_Mask**: A uint8 binary raster produced by the logical AND of the NDSI_Mask
  and the GLIMS_Mask, representing the initial constrained glacier candidate pixels.
- **Final_Mask**: The uint8 binary raster after image-based refinement, representing the
  final glacier extent (1 = glacier, 0 = non-glacier).
- **Refinement**: Reproducible image-processing operations (connected-component filtering,
  morphological operations) applied to the Candidate_Mask to remove isolated noise,
  disconnected patches, and obvious false detections.
- **Validation_Plot**: A PNG figure showing the original RGB composite, the GLIMS boundary
  overlay, the Candidate_Mask, and the Final_Mask side by side.
- **Downloader**: The utility function or script that fetches the GLIMS shapefile for
  Samudra Tapu Glacier from the GLIMS WFS endpoint.
- **Reprojector**: The component that transforms GLIMS vector geometry from EPSG:4326 to
  the CRS of the NDSI raster and rasterizes it to an identical pixel grid.
- **Glacier_Area_Summary**: A CSV file recording the final glacier area in km² per year,
  derived from the Final_Mask.

---

## Requirements

### Requirement 1: GLIMS Data Acquisition

**User Story:** As a researcher, I want the pipeline to obtain the GLIMS glacier boundary
for Samudra Tapu Glacier automatically, so that I do not need to manually download or
locate the correct shapefile.

#### Acceptance Criteria

1. WHEN the Downloader is invoked and no GLIMS shapefile is present in `data/glims/`,
   THE Downloader SHALL fetch the GLIMS glacier outline for Samudra Tapu Glacier from
   the GLIMS WFS REST endpoint and save it as a GeoJSON or shapefile in `data/glims/`.
2. WHEN the Downloader is invoked and a GLIMS file already exists in `data/glims/`,
   THE Downloader SHALL skip the download and use the cached file.
3. IF the GLIMS WFS endpoint is unreachable, THEN THE Downloader SHALL raise a
   descriptive error stating the URL that failed and instruct the user to place a
   manually downloaded GLIMS file in `data/glims/`.
4. THE Downloader SHALL filter the downloaded dataset to retain only the polygon
   whose GLIMS name or glacier ID matches Samudra Tapu Glacier (GLIMS ID: G077186E32649N
   or the matching record by name "Samudra Tapu").
5. IF the filtered dataset contains zero polygons, THEN THE Downloader SHALL raise a
   descriptive error identifying the glacier name and ID that were searched.

---

### Requirement 2: GLIMS Rasterization and Spatial Alignment

**User Story:** As a researcher, I want the GLIMS boundary converted to a raster that
exactly matches the NDSI grid, so that per-pixel logical operations are valid.

#### Acceptance Criteria

1. WHEN the Reprojector receives a GLIMS polygon and an NDSI raster, THE Reprojector
   SHALL reproject the polygon from its source CRS (EPSG:4326) to the CRS of the
   NDSI raster.
2. THE Reprojector SHALL rasterize the reprojected polygon to a uint8 grid with the
   same width, height, transform, and CRS as the NDSI raster (1 = inside, 0 = outside).
3. THE Reprojector SHALL save the resulting GLIMS_Mask to
   `data/processed/glims/glims_mask_YEAR.tif` with lossless DEFLATE compression.
4. WHEN the GLIMS_Mask and NDSI_Mask for the same year are compared, THE Pipeline SHALL
   verify that both rasters share identical width, height, CRS, and affine transform
   before performing any intersection; IF they differ, THEN THE Pipeline SHALL raise a
   descriptive alignment error.
5. THE Reprojector SHALL produce a GLIMS_Mask whose covered area (number of pixels = 1)
   is greater than zero; IF the covered area is zero, THEN THE Reprojector SHALL raise
   a descriptive error indicating a possible CRS mismatch or empty geometry.

---

### Requirement 3: NDSI–GLIMS Intersection (Candidate Mask)

**User Story:** As a researcher, I want to constrain the NDSI snow/ice candidates to
the known glacier boundary, so that detections outside the glacier footprint are excluded.

#### Acceptance Criteria

1. WHEN the Pipeline intersects the NDSI_Mask and the GLIMS_Mask for a given year, THE
   Pipeline SHALL produce the Candidate_Mask as the logical AND of pixels where
   NDSI_Mask = 1 and GLIMS_Mask = 1.
2. THE Pipeline SHALL preserve the nodata value (255) from the NDSI_Mask in the
   Candidate_Mask for all pixels where NDSI_Mask = 255.
3. THE Pipeline SHALL save the Candidate_Mask to
   `data/processed/candidates/glacier_candidate_YEAR.tif` as a uint8 raster with
   DEFLATE compression, nodata = 255, and the same CRS and transform as the NDSI_Mask.
4. THE Candidate_Mask SHALL contain only the values 0, 1, and 255.
5. THE Pipeline SHALL log the number of Candidate_Mask pixels equal to 1 and their
   area in km² before refinement.

---

### Requirement 4: Image-Based Refinement

**User Story:** As a researcher, I want isolated noise and disconnected patches removed
from the candidate mask, so that the final mask represents a spatially coherent glacier
body rather than scattered false detections.

#### Acceptance Criteria

1. WHEN the Refinement stage receives a Candidate_Mask, THE Refinement SHALL apply
   connected-component labelling to identify contiguous glacier patches.
2. THE Refinement SHALL retain only connected components whose area meets or exceeds
   a configurable minimum-area threshold (default: 1 km², i.e., 10,000 pixels at 10 m
   resolution).
3. WHERE morphological closing is enabled (default: enabled), THE Refinement SHALL
   apply binary morphological closing with a configurable structuring element radius
   (default: 3 pixels) to fill small interior holes before connected-component filtering.
4. WHERE morphological closing is enabled, THE Refinement SHALL apply morphological
   closing only within the GLIMS_Mask boundary, so that closing does not expand the
   glacier candidate outside the GLIMS polygon.
5. THE Refinement SHALL NOT alter pixel values outside the GLIMS_Mask region.
6. THE Refinement SHALL produce a Final_Mask containing only the values 0 and 1
   (nodata pixels from the input are set to 0 in the Final_Mask, as the GLIMS boundary
   constrains the domain).
7. THE Refinement SHALL log the number of components removed and the pixel count of
   the Final_Mask.

---

### Requirement 5: Final Mask Output

**User Story:** As a researcher, I want the final glacier mask saved as a georeferenced
raster, so that it can be loaded in GIS tools and compared across years.

#### Acceptance Criteria

1. THE Pipeline SHALL save the Final_Mask to
   `data/processed/final_masks/glacier_mask_YEAR.tif` as a uint8 GeoTIFF.
2. THE Final_Mask SHALL have the same CRS, width, height, and affine transform as the
   source NDSI raster.
3. THE Final_Mask SHALL be compressed with DEFLATE and have nodata set to 255 (even
   though the Refinement sets former-nodata pixels to 0, the nodata tag is preserved
   for GIS compatibility).
4. THE Final_Mask SHALL carry a band description of "glacier_mask".
5. THE Pipeline SHALL create the `data/processed/final_masks/` directory if it does
   not exist.

---

### Requirement 6: Visual Validation Output

**User Story:** As a researcher, I want a side-by-side validation figure per year, so
that I can visually inspect the effect of each pipeline stage.

#### Acceptance Criteria

1. THE Pipeline SHALL produce a Validation_Plot saved to
   `data/processed/validation/validation_YEAR.png` for each processed year.
2. THE Validation_Plot SHALL contain at least four panels: (a) true-colour RGB
   composite from the raw Sentinel-2 bands B4/B3/B2, (b) NDSI_Mask with the GLIMS
   boundary contour overlaid, (c) Candidate_Mask, and (d) Final_Mask.
3. THE Validation_Plot SHALL render the GLIMS polygon boundary as a visible contour
   or outline on the panels where it aids interpretation.
4. THE Validation_Plot SHALL include a title showing the year and the final glacier
   area in km².
5. THE Pipeline SHALL create the `data/processed/validation/` directory if it does
   not exist.
6. THE Validation_Plot SHALL be rendered without displaying a GUI window (non-interactive
   Matplotlib backend, e.g., "Agg").

---

### Requirement 7: Glacier Area Calculation and Summary

**User Story:** As a researcher, I want glacier area recorded for every year in a
single CSV file, so that I can track glacier change over time.

#### Acceptance Criteria

1. THE Pipeline SHALL compute glacier area for each year as:
   `area_km2 = count(Final_Mask == 1) × 100 / 1,000,000`
   where 100 is the pixel area in m² (10 m × 10 m).
2. THE Pipeline SHALL append or write a row per year to
   `data/processed/glacier_area_summary.csv` containing at minimum: year, final
   glacier pixels, final glacier area km², and candidate pixels before refinement.
3. WHEN the Pipeline processes multiple years, THE Pipeline SHALL write all rows to
   `data/processed/glacier_area_summary.csv` in ascending year order.
4. THE Pipeline SHALL NOT require ground-truth masks and SHALL NOT compute IoU, Dice,
   or any quantitative accuracy metric.

---

### Requirement 8: Pipeline Orchestration and CLI

**User Story:** As a researcher, I want to run the full pipeline from the command line
for one year or all years, so that processing is reproducible and scriptable.

#### Acceptance Criteria

1. THE Pipeline SHALL be executable as a Python script (`src/glacier_pipeline.py`) from
   the project root with no required arguments (defaults to processing all years).
2. WHEN the `--year` argument is provided with a four-digit year value, THE Pipeline
   SHALL process only that year's data.
3. WHEN the `--ndsi-dir` argument is provided, THE Pipeline SHALL read NDSI inputs
   from the specified directory instead of the default `data/processed/ndsi/`.
4. WHEN the `--min-area-km2` argument is provided, THE Pipeline SHALL use that value
   as the minimum connected-component area threshold in Requirement 4.2.
5. IF an NDSI input file for a requested year is not found, THEN THE Pipeline SHALL
   log a warning and skip that year without terminating the entire run.
6. THE Pipeline SHALL print per-year progress to stdout in a human-readable format
   including year, candidate area before refinement, and final glacier area.
7. THE Pipeline SHALL exit with a non-zero return code if no years were successfully
   processed.

---

### Requirement 9: Dependency and Environment Compatibility

**User Story:** As a researcher, I want the new pipeline to use only packages
compatible with the existing environment, so that no environment conflicts arise.

#### Acceptance Criteria

1. THE Pipeline SHALL use only Python standard library modules plus the packages
   already listed in `requirements.txt` and the following additional packages:
   `geopandas`, `shapely`, `scipy`, and `requests`.
2. THE Pipeline SHALL record the four new packages in `requirements.txt` with pinned
   versions compatible with the existing `rasterio==1.5.2`, `numpy==2.5.3`, and
   `matplotlib==3.11.2`.
3. THE Pipeline SHALL NOT import or depend on deep-learning libraries (PyTorch,
   TensorFlow, Keras, etc.).
4. THE GLIMS utility functions SHALL be placed in a separate module
   `src/glims_utils.py` so that they are independently importable and testable.
5. THE Pipeline SHALL NOT modify or re-run `src/ndsi_mask.py` or any existing NDSI
   output files.
