import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from cattlelogue.unet import UNet
from cattlelogue.datasets import build_dataset, load_rf_results

from rich import print
import numpy as np
import cv2
import click


def build_unet_data(year=2015, stride=8, ignore_ocean=True):
    """
    We need to break up our data into patches for training. Taking our
    large global map, we break it into smaller overlapping patches of size 32x32
    such that data wraps around the globe.
    """

    STRIDE = stride
    PATCH_SIZE = 32

    dataset = build_dataset(process_ee=True, flatten=False, year=year)
    # Load RF inference results
    crop_results = load_rf_results("crops_")[year]
    pasture_results = load_rf_results("pasture_")[year]
    # HACK upscale results to match the dataset resolution
    crop_results = cv2.resize(
        crop_results, (dataset["glw4_shape"][1], dataset["glw4_shape"][0]), interpolation=cv2.INTER_CUBIC
    )[:, :, np.newaxis]
    pasture_results = cv2.resize(
        pasture_results, (dataset["glw4_shape"][1], dataset["glw4_shape"][0]), interpolation=cv2.INTER_CUBIC
    )[:, :, np.newaxis]
    features = dataset["features"]
    features = np.concatenate((features, crop_results, pasture_results), axis=-1)
    features = np.pad(
        features,
        (
            (PATCH_SIZE // 2, PATCH_SIZE // 2),
            (PATCH_SIZE // 2, PATCH_SIZE // 2),
            (0, 0),
        ),
        mode="wrap",
    )

    ground_truth = dataset["livestock_density"]
    ground_truth = np.pad(
        ground_truth,
        ((PATCH_SIZE // 2, PATCH_SIZE // 2), (PATCH_SIZE // 2, PATCH_SIZE // 2)),
        mode="wrap",
    )

    patches = []
    y = []

    for i in range(0, features.shape[0] - PATCH_SIZE + 1, STRIDE):
        for j in range(0, features.shape[1] - PATCH_SIZE + 1, STRIDE):
            patch_features = features[i : i + PATCH_SIZE, j : j + PATCH_SIZE, :]
            # instead of choosing the center pixel as in CNN, we take the whole patch
            # as ground truth mask
            gt_patch = ground_truth[i : i + PATCH_SIZE, j : j + PATCH_SIZE] > 0.5
            if np.any(gt_patch) or not ignore_ocean:  # ignore ocean patches
                patches.append(patch_features)
                y.append(gt_patch)

    print("Number of patches:", len(patches))
    return patches, y


def calc_livestock_stats(ground_truth):
    """
    Calculate statistics for livestock density.

    Parameters:
    ground_truth (numpy.ndarray): 2D array of livestock density values.

    Returns:
    dict: A dictionary containing the mean, median, and standard deviation.
    """
    mean_density = np.mean(ground_truth)
    median_density = np.median(ground_truth)
    std_density = np.std(ground_truth)

    return {"mean": mean_density, "median": median_density, "std": std_density}


@click.command()
@click.option("--epochs", type=int, default=20, help="Number of training epochs")
@click.option("--batch_size", type=int, default=256, help="Size of each training batch")
@click.option(
    "--learning_rate", type=float, default=0.001, help="Initial learning rate"
)
@click.option(
    "--step_size", type=int, default=10, help="Step size for learning rate scheduler"
)
@click.option(
    "--gamma",
    type=float,
    default=0.1,
    help="Multiplicative factor for learning rate decay",
)
def train_unet_model(epochs, batch_size, learning_rate, step_size, gamma):
    """
    Train a UNet model for livestock density prediction.
    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Size of each training batch.
        learning_rate (float): Initial learning rate for the optimizer.
        step_size (int): Step size for the learning rate scheduler.
        gamma (float): Multiplicative factor for learning rate decay.
    """
    patches, ground_truth = build_unet_data()
    print("Livestock data statistics:", calc_livestock_stats(ground_truth))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=patches[0][0].shape[-1], out_channels=1).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    criterion = nn.BCEWithLogitsLoss()

    train_patches, val_patches, train_y, val_y = train_test_split(
        patches, ground_truth, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(
        list(zip(train_patches, train_y)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        list(zip(val_patches, val_y)),
        batch_size=batch_size,
        shuffle=False,
    )

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_features, batch_ground_truth in train_loader:
            batch_features = np.stack(batch_features, axis=0)
            # reconfigure feature shape for (batch_size, channels, height, width)
            batch_features = batch_features.transpose(0, 3, 1, 2)
            
            if np.isnan(batch_features).any():
                print("NaN values found in batch features!")
                continue
            batch_ground_truth = np.array(batch_ground_truth, dtype=np.float32)

            batch_features = torch.tensor(batch_features, dtype=torch.float32)
            batch_ground_truth = torch.tensor(batch_ground_truth, dtype=torch.float32)

            batch_features = batch_features.to(device)
            batch_ground_truth = batch_ground_truth.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs.squeeze(), batch_ground_truth.squeeze())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}"
        )

        model.eval()
        val_loss = 0.0
        y_true = np.array([], dtype=np.float32)
        y_pred = np.array([], dtype=np.float32)
        with torch.no_grad():
            for batch_features, batch_ground_truth in val_loader:
                batch_features = np.stack(batch_features, axis=0)
                # reconfigure feature shape for (batch_size, channels, height, width)
                batch_features = batch_features.transpose(0, 3, 1, 2)
                batch_ground_truth = np.array(batch_ground_truth, dtype=np.float32)

                batch_features = torch.tensor(batch_features, dtype=torch.float32)
                batch_ground_truth = torch.tensor(
                    batch_ground_truth, dtype=torch.float32
                )

                batch_features = batch_features.to(device)
                batch_ground_truth = batch_ground_truth.to(device)

                outputs = model(batch_features)
                loss = criterion(outputs.squeeze(), batch_ground_truth)
                val_loss += loss.item()
                y_true = np.concatenate(
                    (y_true, batch_ground_truth.flatten().cpu().numpy())
                )
                y_pred = np.concatenate((y_pred, outputs.flatten().cpu().numpy()))
            val_loss /= len(val_loader)

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation AUC: {roc_auc_score(y_true, y_pred):.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_unet_model.pth")
            print("Saved best model.")


if __name__ == "__main__":
    train_unet_model()
