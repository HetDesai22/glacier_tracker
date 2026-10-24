
"""
MELTING ICE TRACKERS
U-Net glacier segmentation training pipeline

Input:
    Six-band satellite GeoTIFFs (B2, B3, B4, B8, B11, B12)

Labels:
    Binary glacier masks (0 = background, 1 = glacier)

Training:
    2018-2024

Validation:
    2025-2026

Outputs:
    models/unet/best_unet.pth
    models/unet/training_history.csv
"""

from pathlib import Path
import csv
import random

import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ==================================================
# PART 1: CONFIGURATION
# ==================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT_DIR / "data" / "raw"
MASK_DIR = ROOT_DIR / "data" / "processed" / "final_masks"
MODEL_DIR = ROOT_DIR / "models" / "unet"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2018, 2027))
TRAIN_YEARS = list(range(2018, 2025))
VAL_YEARS = [2025, 2026]

INPUT_CHANNELS = 6
PATCH_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
BASE_CHANNELS = 16
RANDOM_SEED = 42

CHECKPOINT_PATH = MODEL_DIR / "best_unet.pth"
HISTORY_PATH = MODEL_DIR / "training_history.csv"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 55)
print("MELTING ICE TRACKERS - U-NET TRAINING")
print("=" * 55)
print("Project directory:", ROOT_DIR)
print("Image directory: ", IMAGE_DIR)
print("Mask directory:  ", MASK_DIR)
print("Model directory: ", MODEL_DIR)
print("Input channels:  ", INPUT_CHANNELS)
print("Patch size:      ", f"{PATCH_SIZE} x {PATCH_SIZE}")
print("Batch size:      ", BATCH_SIZE)
print("Maximum epochs:  ", EPOCHS)
print("Learning rate:   ", LEARNING_RATE)
print("Device:          ", DEVICE)

if torch.cuda.is_available():
    print("GPU:             ", torch.cuda.get_device_name(0))
else:
    print("GPU unavailable; training will use the CPU.")

print("=" * 55)


# ==================================================
# PART 2: LOAD AND VALIDATE IMAGE-MASK PAIRS
# ==================================================

def load_image_mask_pairs(years=YEARS):
    pairs = []

    print("\nLoading satellite images and masks...")
    print("-" * 55)

    for year in years:
        image_path = (
            IMAGE_DIR / f"Samudra_Tapu_{year}_Scientific.tif"
        )
        mask_path = MASK_DIR / f"glacier_mask_{year}.tif"

        if not image_path.exists():
            raise FileNotFoundError(
                f"Satellite image not found: {image_path}"
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Glacier mask not found: {mask_path}"
            )

        with rasterio.open(image_path) as src:
            image = src.read()
            image_crs = src.crs
            image_transform = src.transform
            image_height = src.height
            image_width = src.width
            band_count = src.count

        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            mask_crs = src.crs
            mask_transform = src.transform

        if band_count != INPUT_CHANNELS:
            raise ValueError(
                f"{year}: Expected {INPUT_CHANNELS} bands, "
                f"found {band_count}."
            )

        if image.shape[1:] != mask.shape:
            raise ValueError(
                f"{year}: Image and mask dimensions differ: "
                f"{image.shape[1:]} versus {mask.shape}."
            )

        if image_crs != mask_crs:
            raise ValueError(
                f"{year}: Image and mask CRS do not match."
            )

        if image_transform != mask_transform:
            raise ValueError(
                f"{year}: Image and mask transforms do not match."
            )

        if not np.isfinite(image).all():
            raise ValueError(
                f"{year}: Image contains NaN or infinite values."
            )

        if not np.isfinite(mask).all():
            raise ValueError(
                f"{year}: Mask contains NaN or infinite values."
            )

        unique_mask_values = set(np.unique(mask).tolist())

        if not unique_mask_values.issubset({0, 1}):
            raise ValueError(
                f"{year}: Mask must contain only 0 and 1; "
                f"found {unique_mask_values}."
            )

        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        glacier_percentage = 100.0 * float(mask.mean())

        pairs.append({
            "year": year,
            "image": image,
            "mask": mask,
            "crs": image_crs,
            "transform": image_transform,
            "height": image_height,
            "width": image_width,
        })

        print(
            f"{year}: Loaded | "
            f"Image={image.shape} | "
            f"Mask={mask.shape} | "
            f"Glacier pixels={glacier_percentage:.2f}%"
        )

    print("-" * 55)
    print(f"Total annual pairs loaded: {len(pairs)}")

    return pairs


# ==================================================
# PART 3: NORMALIZATION AND PATCH EXTRACTION
# ==================================================

def normalize_image(image):
    """
    Starting reflectance normalization assumption:
    divide digital values by 10000, clip to [0, 2],
    then scale to [0, 1].

    Verify that this scaling is appropriate for the
    source products before interpreting results scientifically.
    """

    image = image.astype(np.float32)
    image = image / 10000.0
    image = np.clip(image, 0.0, 2.0)
    image = image / 2.0

    return np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )


def extract_patches(image, mask, year):
    """
    Extract non-overlapping 256 x 256 patches.

    valid_mask:
        1 = pixel belongs to the original raster
        0 = artificial padding
    """

    patches = []

    height, width = mask.shape
    patch_id = 0

    for row in range(0, height, PATCH_SIZE):
        for col in range(0, width, PATCH_SIZE):

            image_patch = image[
                :,
                row:min(row + PATCH_SIZE, height),
                col:min(col + PATCH_SIZE, width),
            ]

            mask_patch = mask[
                row:min(row + PATCH_SIZE, height),
                col:min(col + PATCH_SIZE, width),
            ]

            original_h, original_w = mask_patch.shape

            valid_patch = np.ones(
                (original_h, original_w),
                dtype=np.float32,
            )

            pad_h = PATCH_SIZE - original_h
            pad_w = PATCH_SIZE - original_w

            if pad_h > 0 or pad_w > 0:
                image_patch = np.pad(
                    image_patch,
                    ((0, 0), (0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )

                mask_patch = np.pad(
                    mask_patch,
                    ((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )

                valid_patch = np.pad(
                    valid_patch,
                    ((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )

            patches.append({
                "year": year,
                "patch_id": patch_id,
                "row": row,
                "col": col,
                "image": image_patch.astype(np.float32),
                "mask": mask_patch.astype(np.float32),
                "valid_mask": valid_patch.astype(np.float32),
            })

            patch_id += 1

    return patches


# ==================================================
# PART 4: PYTORCH DATASET
# ==================================================

class GlacierPatchDataset(Dataset):

    def __init__(self, years, image_mask_pairs=None):
        self.patches = []

        if image_mask_pairs is None:
            image_mask_pairs = load_image_mask_pairs(years)

        selected_years = set(years)

        for pair in image_mask_pairs:
            year = pair["year"]

            if year not in selected_years:
                continue

            image = normalize_image(pair["image"])
            mask = pair["mask"]

            year_patches = extract_patches(
                image,
                mask,
                year,
            )

            self.patches.extend(year_patches)

        if not self.patches:
            raise ValueError(
                f"No patches generated for years: {years}"
            )

        print(
            f"Dataset {min(years)}-{max(years)}: "
            f"{len(self.patches)} patches"
        )

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        patch = self.patches[index]

        image = torch.from_numpy(
            patch["image"].copy()
        ).float()

        mask = torch.from_numpy(
            patch["mask"].copy()
        ).float().unsqueeze(0)

        valid_mask = torch.from_numpy(
            patch["valid_mask"].copy()
        ).float().unsqueeze(0)

        return image, mask, valid_mask


# ==================================================
# PART 5: U-NET ARCHITECTURE
# ==================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):

    def __init__(self, in_channels=6, base_channels=16):
        super().__init__()

        b = base_channels

        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.enc4 = DoubleConv(b * 4, b * 8)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = DoubleConv(b * 8, b * 16)

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

        # Raw logits: do not apply sigmoid inside the model.
        return self.output_layer(d1)


# ==================================================
# PART 6: LOSS FUNCTION AND METRICS
# ==================================================

def combined_loss(logits, targets, valid_mask):
    """
    BCE + Dice loss.
    Artificially padded pixels are excluded.
    """

    valid_mask = valid_mask.float()
    targets = targets.float()

    bce_map = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    valid_count = valid_mask.sum().clamp_min(1.0)

    bce = (bce_map * valid_mask).sum() / valid_count

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities * valid_mask
    targets_valid = targets * valid_mask

    probabilities = probabilities.flatten(start_dim=1)
    targets_valid = targets_valid.flatten(start_dim=1)

    intersection = (
        probabilities * targets_valid
    ).sum(dim=1)

    denominator = (
        probabilities.sum(dim=1)
        + targets_valid.sum(dim=1)
    )

    dice = (
        2.0 * intersection + 1.0
    ) / (
        denominator + 1.0
    )

    return bce + (1.0 - dice).mean()


@torch.no_grad()
def calculate_metrics(
    logits,
    targets,
    valid_mask,
    threshold=0.5,
):
    """
    Compute micro-aggregated pixel metrics for a batch.
    Ignore artificial padding.
    """

    predictions = torch.sigmoid(logits) >= threshold
    truth = targets >= 0.5
    valid = valid_mask >= 0.5

    predictions = predictions & valid
    truth = truth & valid

    tp = (predictions & truth).sum().item()
    fp = (predictions & ~truth & valid).sum().item()
    fn = (~predictions & truth & valid).sum().item()

    dice = (2.0 * tp) / max(2.0 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


# ==================================================
# PART 7: TRAINING AND VALIDATION
# ==================================================

def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_valid_pixels = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for images, masks, valid_masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        valid_masks = valid_masks.to(DEVICE)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)

            loss = combined_loss(
                logits,
                masks,
                valid_masks,
            )

            if training:
                loss.backward()
                optimizer.step()

        batch_valid_pixels = valid_masks.sum().item()

        total_loss += loss.item() * batch_valid_pixels
        total_valid_pixels += batch_valid_pixels

        metrics = calculate_metrics(
            logits,
            masks,
            valid_masks,
        )

        # Recover integer counts from binary pixel metrics
        predictions = torch.sigmoid(logits) >= 0.5
        truth = masks >= 0.5
        valid = valid_masks >= 0.5

        total_tp += (predictions & truth & valid).sum().item()
        total_fp += (predictions & ~truth & valid).sum().item()
        total_fn += (~predictions & truth & valid).sum().item()

    if total_valid_pixels == 0:
        raise ValueError("No valid pixels found in data loader.")

    # Loss is weighted by the number of valid pixels per batch.
    average_loss = total_loss / total_valid_pixels

    dice = (
        2.0 * total_tp
    ) / max(
        2.0 * total_tp + total_fp + total_fn, 1
    )

    iou = total_tp / max(
        total_tp + total_fp + total_fn, 1
    )

    precision = total_tp / max(
        total_tp + total_fp, 1
    )

    recall = total_tp / max(
        total_tp + total_fn, 1
    )

    return {
        "loss": average_loss,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def train_model(model, optimizer, train_loader, val_loader):
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_results = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
        )

        val_results = run_epoch(
            model,
            val_loader,
            optimizer=None,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_results["loss"],
            "val_loss": val_results["loss"],
            "train_dice": train_results["dice"],
            "val_dice": val_results["dice"],
            "train_iou": train_results["iou"],
            "val_iou": val_results["iou"],
            "val_precision": val_results["precision"],
            "val_recall": val_results["recall"],
        }

        history.append(record)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {record['train_loss']:.4f} | "
            f"Val Loss: {record['val_loss']:.4f} | "
            f"Train Dice: {record['train_dice']:.4f} | "
            f"Val Dice: {record['val_dice']:.4f} | "
            f"Val IoU: {record['val_iou']:.4f}"
        )

        if val_results["loss"] < best_val_loss:
            best_val_loss = val_results["loss"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "input_channels": INPUT_CHANNELS,
                    "base_channels": BASE_CHANNELS,
                    "patch_size": PATCH_SIZE,
                    "train_years": TRAIN_YEARS,
                    "validation_years": VAL_YEARS,
                    "normalization": "clip(image / 10000, 0, 2) / 2",
                },
                CHECKPOINT_PATH,
            )

            print("  Best model checkpoint saved.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered after epoch {epoch}."
            )
            break

    with open(
        HISTORY_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)

    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Training history: {HISTORY_PATH}")

    return history


# ==================================================
# PART 8: MAIN ENTRY POINT - START TRAINING
# ==================================================

def main():
    # Load all pairs once and reuse them for both datasets.
    all_pairs = load_image_mask_pairs(YEARS)

    train_dataset = GlacierPatchDataset(
        TRAIN_YEARS,
        image_mask_pairs=all_pairs,
    )

    val_dataset = GlacierPatchDataset(
        VAL_YEARS,
        image_mask_pairs=all_pairs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print("\nDataset summary")
    print("-" * 55)
    print("Training years:", TRAIN_YEARS)
    print("Validation years:", VAL_YEARS)
    print("Training patches:", len(train_dataset))
    print("Validation patches:", len(val_dataset))
    print("Training batches per epoch:", len(train_loader))
    print("Validation batches per epoch:", len(val_loader))

    # Validate one batch before using GPU time for training.
    sample_images, sample_masks, sample_valid = next(
        iter(train_loader)
    )

    print("\nBatch validation")
    print("Images:", tuple(sample_images.shape))
    print("Masks:", tuple(sample_masks.shape))
    print("Valid pixels:", tuple(sample_valid.shape))
    print("Mask values:", torch.unique(sample_masks).tolist())
    print(
        "Valid-mask values:",
        torch.unique(sample_valid).tolist(),
    )

    expected_image_shape = (
        BATCH_SIZE,
        INPUT_CHANNELS,
        PATCH_SIZE,
        PATCH_SIZE,
    )
    expected_mask_shape = (
        BATCH_SIZE,
        1,
        PATCH_SIZE,
        PATCH_SIZE,
    )

    assert tuple(sample_images.shape) == expected_image_shape
    assert tuple(sample_masks.shape) == expected_mask_shape
    assert tuple(sample_valid.shape) == expected_mask_shape

    assert set(torch.unique(sample_masks).tolist()).issubset(
        {0.0, 1.0}
    )
    assert set(torch.unique(sample_valid).tolist()).issubset(
        {0.0, 1.0}
    )

    print("\nDataset checks passed.")
    print("\nCreating U-Net model...")

    model = UNet(
        in_channels=INPUT_CHANNELS,
        base_channels=BASE_CHANNELS,
    ).to(DEVICE)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Trainable parameters: {trainable_parameters:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print("\nStarting actual U-Net training...")
    print("=" * 55)

    train_model(
        model,
        optimizer,
        train_loader,
        val_loader,
    )


if __name__ == "__main__":
    main()
