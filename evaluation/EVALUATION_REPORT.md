# Samudra Tapu Glacier — Segmentation Evaluation

## Evaluation scope

- Requested years: 9
- Successfully evaluated: 9
- Missing or failed: 0
- Reference masks are preliminary automated labels.
- Metrics indicate agreement with the reference, not verified scientific accuracy.

## Results by year

| Year | Dice | IoU | Precision | Recall | Reference km² | Predicted km² | Difference km² |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018 | 0.7164 | 0.5581 | 0.6764 | 0.7615 | 66.105 | 74.419 | +8.314 |
| 2019 | 0.7530 | 0.6039 | 0.7176 | 0.7920 | 64.885 | 71.615 | +6.730 |
| 2020 | 0.7682 | 0.6236 | 0.6705 | 0.8991 | 58.140 | 77.954 | +19.813 |
| 2021 | 0.7393 | 0.5864 | 0.6841 | 0.8041 | 66.097 | 77.692 | +11.595 |
| 2022 | 0.7072 | 0.5470 | 0.7231 | 0.6920 | 66.094 | 63.253 | -2.841 |
| 2023 | 0.7531 | 0.6040 | 0.6911 | 0.8273 | 64.797 | 77.562 | +12.765 |
| 2024 | 0.7849 | 0.6459 | 0.7480 | 0.8256 | 55.445 | 61.199 | +5.753 |
| 2025 | 0.7341 | 0.5799 | 0.7209 | 0.7478 | 65.698 | 68.156 | +2.458 |
| 2026 | 0.7663 | 0.6211 | 0.7439 | 0.7901 | 53.935 | 57.285 | +3.349 |

## Summary statistics

- Mean Dice: 0.7469
- Mean Iou: 0.5967
- Mean Precision: 0.7084
- Mean Recall: 0.7933
- Mean absolute area difference: 8.180 km²

## Interpretation

Review the per-year metrics, area comparison plot, and error maps together. Low precision with high recall indicates overprediction relative to the reference. Low recall indicates missed reference pixels. Investigate year-to-year differences before interpreting them as glacier change.

## Outputs

- `metrics_by_year.csv`: detailed per-year metrics
- `plots/area_comparison.png`: mapped area comparison
- `plots/metrics_by_year.png`: metric trends
- `error_maps/`: spatial disagreement maps

## Limitations

The automated reference masks may contain errors. The evaluation does not independently validate glacier boundaries, seasonal snow, image acquisition consistency, or scientific ground truth. Area differences are not automatically evidence of real glacier retreat or expansion.