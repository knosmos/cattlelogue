# Generate maps of changes in crop/pasture/livestock over time

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import click
from cattlelogue.datasets import load_glw4_data
import colormaps as cmaps

@click.command()
@click.option(
    "--prefix",
    type=str,
    default="crops",
    help="Prefix for the files to load (e.g., 'crops', 'pasture')",
)
@click.option(
    "--year_start",
    type=int,
    default=2015,
    help="Starting year for the analysis",
)
@click.option(
    "--year_end",
    type=int,
    default=2100,
    help="Ending year for the analysis",
)
@click.option(
    "--window",
    type=int,
    default=5,
    help="Size of window to average data over (looks into future for year_start, into past for year_end)",
)
def plot_delta(prefix, year_start, year_end, window):
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    year_start_fnames = [
        os.path.join(OUTPUT_DIR, f"{prefix}_{year}.npy") for year in range(year_start, year_start + window)
    ]
    year_end_fnames = [
        os.path.join(OUTPUT_DIR, f"{prefix}_{year}.npy") for year in range(year_end - window + 1, year_end + 1)
    ]

    year_start_datas = [
        np.load(fname) for fname in year_start_fnames
    ]
    year_end_datas = [
        np.load(fname) for fname in year_end_fnames
    ]
    delta = np.mean(year_end_datas, axis=0) - np.mean(year_start_datas, axis=0)
    #smooth/blur delta
    #delta = cv2.GaussianBlur(delta, (7, 7), 0)

    # hack for masking land areas
    glw4_data = load_glw4_data(resolution=1)
    land_mask = glw4_data[0] >= 0
    delta_masked = np.ma.masked_where(~land_mask, delta)

    plt.figure(figsize=(12, 6))
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 8})
    plt.title(f"Change in livestock distribution | {year_start}-{year_end}".upper(), weight="bold")
    plt.imshow(
        delta_masked,
        #cmap="coolwarm_r",
        cmap=cmaps.pride[::-1],
        vmin=-np.max(np.abs(delta_masked)) * 0.15,
        vmax=np.max(np.abs(delta_masked)) * 0.15,
        interpolation="nearest",
        zorder=10,
    )
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    plt.gca().set_xticks(np.linspace(0, delta_masked.shape[1], 36))
    plt.gca().set_yticks(np.linspace(0, delta_masked.shape[0], 18))
    plt.grid(axis="both", color="#ddd", linestyle="--", linewidth=0.5, zorder=0)
    for tick in frame1.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    for tick in frame1.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    im_ratio = delta.shape[0]/delta.shape[1]
    plt.colorbar(label="Delta".upper(), fraction=0.047*im_ratio)
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{prefix}_delta_{year_start}_{year_end}.png"),
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=500,
    )
    plt.show()

if __name__ == "__main__":
    plot_delta()