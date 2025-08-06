# calc auc score between aglw data and unet outputs

import numpy as np
import cv2
from sklearn.metrics import roc_auc_score, roc_curve, r2_score
from cattlelogue.datasets import load_aglw_data, load_rf_results, load_glw4_data
import matplotlib.pyplot as plt

def calc_auc_aglw_unet(year):
    """
    Calculate AUC score between AGLW data and UNet outputs for a given year.
    """
    aglw_data = load_aglw_data(year)[0]
    glw4_data, glw4_shape = load_glw4_data(resolution=1)
    aglw_data = cv2.resize(aglw_data, (glw4_shape[1], glw4_shape[0]), interpolation=cv2.INTER_AREA)
    unet_results = load_rf_results("livestock2_")[year]
    unet_results = cv2.resize(unet_results, (glw4_shape[1], glw4_shape[0]), interpolation=cv2.INTER_AREA)

    # Flatten the arrays to compute AUC
    aglw_flat = aglw_data.flatten()
    unet_flat = unet_results.flatten()
    glw4_flat = glw4_data.flatten()

    # Filter out invalid values
    valid_mask = np.where(glw4_flat >= 0)
    print(np.mean(aglw_flat))
    print(aglw_flat[valid_mask], valid_mask)
    # avg of aglw_flat

    print(f"AGLW mean: {np.mean(aglw_flat[valid_mask])}, UNet mean: {np.mean(unet_flat[valid_mask])}")
    # aglw_flat = aglw_flat[valid_mask] >= 1
    aglw_flat = np.where(aglw_flat > 0, aglw_flat, 0)  # consider only values >= 1
    aglw_flat = aglw_flat[valid_mask]
    print("AGLW mean:", np.mean(aglw_flat))
    unet_flat = unet_flat[valid_mask]
    print("num -1", np.sum(unet_flat == -1))
    glw4_flat = glw4_flat[valid_mask]

    # auc_score = roc_auc_score(aglw_flat, unet_flat)
    # # draw roc curve
    fpr, tpr, _ = roc_curve(aglw_flat >= 1, unet_flat)
    plt.figure(figsize=(6, 4))
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 8})
    plt.plot(fpr, tpr, label="Cattlelogue", color="#b81a0f")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    #plt.title(f"ROC Curve for AGLW vs UNet outputs in {year}".upper(), weight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid(color="lightgray", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()
    # aglw_2015 = load_aglw_data(2015)[0]
    # aglw_2015 = cv2.resize(aglw_2015, (glw4_shape[1], glw4_shape[0]), interpolation=cv2.INTER_AREA)
    # aglw_2015_flat = aglw_2015.flatten()
    # aglw_2015_flat = np.where(aglw_2015_flat > 0, aglw_2015_flat, 0)  # consider only values >= 1
    # aglw_2015_flat = aglw_2015_flat[valid_mask]

    # mse = np.mean((aglw_flat - unet_flat) ** 2)
    # print("mse:", mse)
    # print("r2 score:", r2_score(aglw_flat >= 1, unet_flat))
    auc_score = roc_auc_score(aglw_flat >= 1, unet_flat)
    return auc_score

if __name__ == "__main__":
    year = 1961
    auc_score = calc_auc_aglw_unet(year)
    print(f"AUC score for AGLW vs UNet outputs in {year}: {auc_score:.4f}")