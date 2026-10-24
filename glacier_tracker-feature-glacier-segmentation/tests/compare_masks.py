
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).parent
output_dir = root / "data" / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

for year in [2024, 2026]:
    pred_path = root / "data" / "processed" / "final_masks" / f"glacier_mask_{year}.tif"
    ref_path = root / "data" / "processed" / "glims" / f"glims_mask_{year}.tif"

    with rasterio.open(pred_path) as pred_src, rasterio.open(ref_path) as ref_src:
        pred = pred_src.read(1)
        ref = ref_src.read(1)

        if pred.shape != ref.shape or pred_src.transform != ref_src.transform or pred_src.crs != ref_src.crs:
            print(f"{year}: Masks are not aligned. Skipping.")
            continue

        valid = np.ones(pred.shape, dtype=bool)

        if pred_src.nodata is not None:
            valid &= pred != pred_src.nodata
        if ref_src.nodata is not None:
            valid &= ref != ref_src.nodata

    p = (pred == 1) & valid
    r = (ref == 1) & valid

    # Green = agreement; red = predicted glacier only;
    # blue = GLIMS glacier only; gray = NoData.
    overlay = np.zeros((*pred.shape, 3), dtype=np.float32)
    overlay[p & r] = [0, 0.8, 0]
    overlay[p & ~r & valid] = [1, 0, 0]
    overlay[~p & r & valid] = [0, 0.3, 1]
    overlay[~valid] = [0.5, 0.5, 0.5]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].imshow(np.ma.masked_where(~valid, pred), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"{year}: Predicted mask")

    axes[1].imshow(np.ma.masked_where(~valid, ref), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"{year}: GLIMS reference")

    axes[2].imshow(overlay)
    axes[2].set_title("Agreement and disagreement")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"Samudra Tapu Glacier - {year}")
    fig.tight_layout()

    output_path = output_dir / f"mask_comparison_{year}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")

print("Done. Open the two PNG files in data/outputs.")
