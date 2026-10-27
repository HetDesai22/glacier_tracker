from pathlib import Path
import csv
import json
import numpy as np
import rasterio
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent

REFERENCE_DIR = ROOT_DIR / "data" / "processed" / "final_masks"
PREDICTION_DIR = ROOT_DIR / "data" / "processed" / "predictions"
RAW_DIR = ROOT_DIR / "data" / "raw"

OUTPUT_DIR = ROOT_DIR / "evaluation"
ERROR_MAP_DIR = OUTPUT_DIR / "error_maps"
PLOT_DIR = OUTPUT_DIR / "plots"

YEARS = range(2018, 2027)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_MAP_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_year(year):
    reference_path = REFERENCE_DIR / f"glacier_mask_{year}.tif"
    prediction_path = PREDICTION_DIR / f"glacier_prediction_{year}.tif"

    if not reference_path.exists() or not prediction_path.exists():
        return {
            "year": year,
            "status": "MISSING_FILE",
            "error": "Reference or prediction mask is missing",
        }

    with rasterio.open(reference_path) as ref_src, \
         rasterio.open(prediction_path) as pred_src:

        reference = ref_src.read(1)
        prediction = pred_src.read(1)

        if reference.shape != prediction.shape:
            raise ValueError(f"{year}: Mask dimensions differ")

        if ref_src.crs != pred_src.crs:
            raise ValueError(f"{year}: CRS differs")

        if ref_src.transform != pred_src.transform:
            raise ValueError(f"{year}: Raster transforms differ")

        if not np.isin(reference, [0, 1]).all():
            raise ValueError(f"{year}: Reference mask is not binary")

        if not np.isin(prediction, [0, 1]).all():
            raise ValueError(f"{year}: Prediction mask is not binary")

        pixel_area_km2 = abs(
            ref_src.transform.a * ref_src.transform.e
            - ref_src.transform.b * ref_src.transform.d
        ) / 1_000_000

        reference_area = np.count_nonzero(reference == 1) * pixel_area_km2
        prediction_area = np.count_nonzero(prediction == 1) * pixel_area_km2

    tp_mask = (reference == 1) & (prediction == 1)
    fp_mask = (reference == 0) & (prediction == 1)
    fn_mask = (reference == 1) & (prediction == 0)
    tn_mask = (reference == 0) & (prediction == 0)

    tp = int(np.count_nonzero(tp_mask))
    fp = int(np.count_nonzero(fp_mask))
    fn = int(np.count_nonzero(fn_mask))
    tn = int(np.count_nonzero(tn_mask))

    dice_denominator = 2 * tp + fp + fn
    union = tp + fp + fn
    precision_denominator = tp + fp
    recall_denominator = tp + fn

    dice = 2 * tp / dice_denominator if dice_denominator else 1.0
    iou = tp / union if union else 1.0
    precision = (
        tp / precision_denominator if precision_denominator else 1.0
    )
    recall = tp / recall_denominator if recall_denominator else 1.0

    # Create spatial error map: green=TP, red=FP, blue=FN, black=TN
    error_map = np.zeros((*reference.shape, 3), dtype=np.uint8)
    error_map[tp_mask] = [0, 255, 0]
    error_map[fp_mask] = [255, 0, 0]
    error_map[fn_mask] = [0, 0, 255]

    error_map_path = ERROR_MAP_DIR / f"error_map_{year}.png"
    plt.imsave(error_map_path, error_map)

    return {
        "year": year,
        "status": "OK",
        "error": "",
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "reference_area_km2": reference_area,
        "prediction_area_km2": prediction_area,
        "area_difference_km2": prediction_area - reference_area,
        "error_map": str(error_map_path.relative_to(ROOT_DIR)),
    }


def save_csv(results):
    csv_path = OUTPUT_DIR / "metrics_by_year.csv"

    fields = [
        "year", "status", "error", "tp", "fp", "fn", "tn",
        "dice", "iou", "precision", "recall",
        "reference_area_km2", "prediction_area_km2",
        "area_difference_km2", "error_map",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for result in results:
            writer.writerow(result)

    return csv_path


def save_plots(results):
    valid = [r for r in results if r["status"] == "OK"]

    if not valid:
        return

    years = [r["year"] for r in valid]

    # Reference and predicted areas
    plt.figure(figsize=(10, 5))
    plt.plot(
        years, [r["reference_area_km2"] for r in valid],
        marker="o", label="Reference"
    )
    plt.plot(
        years, [r["prediction_area_km2"] for r in valid],
        marker="o", label="Prediction"
    )
    plt.xlabel("Year")
    plt.ylabel("Mapped area (km²)")
    plt.title("Reference vs Predicted Glacier Area")
    plt.xticks(years)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "area_comparison.png", dpi=200)
    plt.close()

    # Metrics by year
    plt.figure(figsize=(10, 5))

    for metric in ["dice", "iou", "precision", "recall"]:
        plt.plot(
            years, [r[metric] for r in valid],
            marker="o", label=metric.capitalize()
        )

    plt.xlabel("Year")
    plt.ylabel("Score")
    plt.title("Segmentation Metrics by Year")
    plt.xticks(years)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "metrics_by_year.png", dpi=200)
    plt.close()


def save_report(results):
    valid = [r for r in results if r["status"] == "OK"]
    missing = [r for r in results if r["status"] != "OK"]

    report_path = OUTPUT_DIR / "EVALUATION_REPORT.md"

    lines = [
        "# Samudra Tapu Glacier — Segmentation Evaluation",
        "",
        "## Evaluation scope",
        "",
        f"- Requested years: {len(list(YEARS))}",
        f"- Successfully evaluated: {len(valid)}",
        f"- Missing or failed: {len(missing)}",
        "- Reference masks are preliminary automated labels.",
        "- Metrics indicate agreement with the reference, not verified "
        "scientific accuracy.",
        "",
    ]

    if valid:
        lines.extend([
            "## Results by year",
            "",
            "| Year | Dice | IoU | Precision | Recall | "
            "Reference km² | Predicted km² | Difference km² |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])

        for r in valid:
            lines.append(
                f"| {r['year']} | {r['dice']:.4f} | {r['iou']:.4f} | "
                f"{r['precision']:.4f} | {r['recall']:.4f} | "
                f"{r['reference_area_km2']:.3f} | "
                f"{r['prediction_area_km2']:.3f} | "
                f"{r['area_difference_km2']:+.3f} |"
            )

        lines.extend([
            "",
            "## Summary statistics",
            "",
        ])

        for metric in ["dice", "iou", "precision", "recall"]:
            values = [r[metric] for r in valid]
            lines.append(
                f"- Mean {metric.capitalize()}: {np.mean(values):.4f}"
            )

        mean_area_error = np.mean([
            abs(r["area_difference_km2"]) for r in valid
        ])
        lines.append(
            f"- Mean absolute area difference: {mean_area_error:.3f} km²"
        )

        lines.extend([
            "",
            "## Interpretation",
            "",
            "Review the per-year metrics, area comparison plot, and error "
            "maps together. Low precision with high recall indicates "
            "overprediction relative to the reference. Low recall indicates "
            "missed reference pixels. Investigate year-to-year differences "
            "before interpreting them as glacier change.",
            "",
            "## Outputs",
            "",
            "- `metrics_by_year.csv`: detailed per-year metrics",
            "- `plots/area_comparison.png`: mapped area comparison",
            "- `plots/metrics_by_year.png`: metric trends",
            "- `error_maps/`: spatial disagreement maps",
            "",
            "## Limitations",
            "",
            "The automated reference masks may contain errors. The "
            "evaluation does not independently validate glacier boundaries, "
            "seasonal snow, image acquisition consistency, or scientific "
            "ground truth. Area differences are not automatically evidence "
            "of real glacier retreat or expansion.",
        ])

    if missing:
        lines.extend(["", "## Missing or failed years", ""])
        for r in missing:
            lines.append(f"- {r['year']}: {r['error']}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    results = []

    for year in YEARS:
        try:
            result = evaluate_year(year)
            results.append(result)

            if result["status"] == "OK":
                print(
                    f"{year}: Dice={result['dice']:.4f}, "
                    f"IoU={result['iou']:.4f}, "
                    f"Precision={result['precision']:.4f}, "
                    f"Recall={result['recall']:.4f}, "
                    f"Area difference="
                    f"{result['area_difference_km2']:+.3f} km²"
                )
            else:
                print(f"{year}: {result['status']} - {result['error']}")

        except Exception as exc:
            result = {
                "year": year,
                "status": "ERROR",
                "error": str(exc),
            }
            results.append(result)
            print(f"{year}: ERROR - {exc}")

    csv_path = save_csv(results)
    save_plots(results)
    report_path = save_report(results)

    print("\nEvaluation completed.")
    print(f"CSV: {csv_path}")
    print(f"Report: {report_path}")
    print(f"Plots: {PLOT_DIR}")
    print(f"Error maps: {ERROR_MAP_DIR}")


if __name__ == "__main__":
    main()