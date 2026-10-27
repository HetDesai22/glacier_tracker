
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent

HISTORY_PATH = ROOT_DIR / "models" / "unet" / "training_history.csv"
OUTPUT_DIR = ROOT_DIR / "models" / "unet"

history = pd.read_csv(HISTORY_PATH)

print(history.head())
print("\nTraining history loaded successfully.")
print("Number of epochs:", len(history))

# Plot training and validation loss
plt.figure(figsize=(9, 5))

plt.plot(
    history["epoch"],
    history["train_loss"],
    label="Training Loss",
    marker="o",
)

plt.plot(
    history["epoch"],
    history["val_loss"],
    label="Validation Loss",
    marker="o",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("U-Net Training vs Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

loss_path = OUTPUT_DIR / "loss_curve.png"
plt.savefig(loss_path, dpi=150)
plt.show()

print("\nLoss graph saved to:", loss_path)
