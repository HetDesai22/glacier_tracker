
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn


ROOT_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT_DIR / "data" / "raw"

MODEL_PATH = ROOT_DIR / "models" / "unet" / "best_unet.pth"

OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "predictions"


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
YEARS = list(range(2018, 2027))

INPUT_CHANNELS = 6
PATCH_SIZE = 256
BASE_CHANNELS = 16

PREDICTION_THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# PART 3: U-NET MODEL ARCHITECTURE
# ==================================================

class DoubleConv(nn.Module):
    """Apply two convolution operations with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    """U-Net for six-band satellite glacier segmentation."""

    def __init__(self, in_channels=6, base_channels=16):
        super().__init__()

        b = base_channels

        # Encoder: extract increasingly complex features
        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.enc4 = DoubleConv(b * 4, b * 8)

        self.pool = nn.MaxPool2d(kernel_size=2)

        # Bottleneck: deepest feature representation
        self.bottleneck = DoubleConv(b * 8, b * 16)

        # Decoder: recover spatial resolution
        self.up4 = nn.ConvTranspose2d(
            b * 16, b * 8, kernel_size=2, stride=2
        )
        self.dec4 = DoubleConv(b * 16, b * 8)

        self.up3 = nn.ConvTranspose2d(
            b * 8, b * 4, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(b * 8, b * 4)

        self.up2 = nn.ConvTranspose2d(
            b * 4, b * 2, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(b * 4, b * 2)

        self.up1 = nn.ConvTranspose2d(
            b * 2, b, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(b * 2, b)

        # Output one score per pixel
        self.output_layer = nn.Conv2d(
            b, 1, kernel_size=1
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        bottleneck = self.bottleneck(self.pool(e4))

        d4 = self.up4(bottleneck)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.output_layer(d1)


# ==================================================
# PART 4: LOAD THE TRAINED CHECKPOINT
# ==================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model checkpoint not found: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model = UNet(
    in_channels=checkpoint["input_channels"],
    base_channels=checkpoint["base_channels"]
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("\nTrained model loaded successfully.")
print("Device:", DEVICE)
print("Best training epoch:", checkpoint.get("epoch", "Not recorded"))
print(
    "Best validation loss:",
    checkpoint.get("best_val_loss", "Not recorded")
)


# ==================================================
# PART 5: SATELLITE IMAGE NORMALIZATION
# ==================================================

def normalize_image(image):
    """
    Normalize a six-band satellite image using the
    same scaling procedure used during training.

    Input shape:  (6, height, width)
    Output shape: (6, height, width)
    """

    image = image.astype(np.float32)

    image = image / 10000.0

    image = np.clip(image, 0.0, 2.0)

    image = image / 2.0

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return image


# ==================================================
# PART 6: PATCH-BASED PREDICTION
# ==================================================

def predict_image(model, image):
    """
    Predict glacier probabilities for a complete image.

    Parameters
    ----------
    model : trained UNet
    image : normalized array with shape (6, height, width)

    Returns
    -------
    probability_map : float32 array with shape (height, width)
    """

    bands, height, width = image.shape

    if bands != INPUT_CHANNELS:
        raise ValueError(
            f"Expected {INPUT_CHANNELS} bands, received {bands}"
        )

    probability_map = np.zeros(
        (height, width),
        dtype=np.float32
    )

    model.eval()

    with torch.no_grad():
        for y in range(0, height, PATCH_SIZE):
            for x in range(0, width, PATCH_SIZE):

                patch = image[
                    :,
                    y:min(y + PATCH_SIZE, height),
                    x:min(x + PATCH_SIZE, width)
                ]

                patch_height = patch.shape[1]
                patch_width = patch.shape[2]

                padded_patch = np.zeros(
                    (INPUT_CHANNELS, PATCH_SIZE, PATCH_SIZE),
                    dtype=np.float32
                )

                padded_patch[:, :patch_height, :patch_width] = patch

                input_tensor = torch.from_numpy(
                    padded_patch
                ).unsqueeze(0).to(DEVICE)

                logits = model(input_tensor)

                probabilities = torch.sigmoid(logits)

                probabilities = probabilities.squeeze().cpu().numpy()

                probability_map[
                    y:y + patch_height,
                    x:x + patch_width
                ] = probabilities[:patch_height, :patch_width]

    return probability_map


# ==================================================
# PART 7: SAVE A GEOTIFF PREDICTION
# ==================================================

def save_prediction(probability_map, source_path, output_path):
    """
    Convert probabilities to a binary mask and save it
    using the source satellite image's geographic metadata.
    """

    binary_mask = (
        probability_map >= PREDICTION_THRESHOLD
    ).astype(np.uint8)

    with rasterio.open(source_path) as src:
        profile = src.profile.copy()

        if probability_map.shape != (src.height, src.width):
            raise ValueError(
                "Prediction dimensions do not match source image."
            )

        profile.update(
            count=1,
            dtype="uint8",
            compress="lzw"
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(binary_mask, 1)

    glacier_pixels = int(binary_mask.sum())
    total_pixels = int(binary_mask.size)
    glacier_percentage = 100.0 * glacier_pixels / total_pixels

    print("Saved prediction:", output_path)
    print("Image dimensions:", binary_mask.shape)
    print("Predicted glacier pixels:", glacier_pixels)
    print(f"Predicted glacier percentage: {glacier_percentage:.2f}%")

    return binary_mask


# ==================================================
# PART 8: PROCESS ALL YEARS
# ==================================================

def main():
    print("\nStarting glacier prediction pipeline")
    print("Running on:", DEVICE)

    for year in YEARS:
        image_path = (
            IMAGE_DIR
            / f"Samudra_Tapu_{year}_Scientific.tif"
        )

        output_path = (
            OUTPUT_DIR
            / f"glacier_prediction_{year}.tif"
        )

        print(f"\n{'=' * 50}")
        print(f"Processing year: {year}")
        print(f"Source: {image_path}")

        if not image_path.exists():
            print(f"WARNING: Image not found for {year}; skipping.")
            continue

        try:
            with rasterio.open(image_path) as src:
                if src.count != INPUT_CHANNELS:
                    raise ValueError(
                        f"Expected {INPUT_CHANNELS} bands, "
                        f"but found {src.count}."
                    )

                image = src.read()

            print("Original image shape:", image.shape)

            normalized_image = normalize_image(image)

            probability_map = predict_image(
                model,
                normalized_image
            )

            save_prediction(
                probability_map,
                image_path,
                output_path
            )

            print(f"Year {year} completed successfully.")

        except Exception as exc:
            print(f"ERROR processing {year}: {exc}")
            raise

    print("\nGlacier prediction pipeline finished.")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

