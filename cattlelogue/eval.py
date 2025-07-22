# calc auc score between aglw data and unet outputs

import numpy as np
import cv2
from sklearn.metrics import roc_auc_score
from cattlelogue.datasets import load_aglw_data, load_rf_results, load_glw4_data

def calc_auc_aglw_unet(year):
    """
    Calculate AUC score between AGLW data and UNet outputs for a given year.
    """
    aglw_data = load_aglw_data(year)[0]
    print(np.where(np.isnan(aglw_data)))
    glw4_data, glw4_shape = load_glw4_data(resolution=1)
    aglw_data = cv2.resize(aglw_data, (glw4_shape[1], glw4_shape[0]), interpolation=cv2.INTER_AREA)
    unet_results = load_rf_results("livestock_")[year]
    unet_results = cv2.resize(unet_results, (glw4_shape[1], glw4_shape[0]), interpolation=cv2.INTER_AREA)

    # Flatten the arrays to compute AUC
    aglw_flat = aglw_data.flatten()
    unet_flat = unet_results.flatten()
    glw4_flat = glw4_data.flatten()

    # Filter out invalid values
    valid_mask = np.where(glw4_flat >= -1000)
    print(np.mean(aglw_flat))
    print(aglw_flat[valid_mask], valid_mask)
    # avg of aglw_flat
    print(f"AGLW mean: {np.mean(aglw_flat[valid_mask])}, UNet mean: {np.mean(unet_flat[valid_mask])}")
    aglw_flat = aglw_flat[valid_mask] >= 1
    print("AGLW mean:", np.mean(aglw_flat))
    unet_flat = unet_flat[valid_mask]

    auc_score = roc_auc_score(aglw_flat, unet_flat)
    return auc_score

if __name__ == "__main__":
    year = 1961
    auc_score = calc_auc_aglw_unet(year)
    print(f"AUC score for AGLW vs UNet outputs in {year}: {auc_score:.4f}")