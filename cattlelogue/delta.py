# Generate maps of changes in crop/pasture/livestock over time

import numpy as np
import os
import matplotlib.pyplot as plt
import colormaps as cmaps
import click
from cattlelogue.datasets import load_glw4_data

@click.command()
@click.option(
    "--prefix",
    type=str,
    default="crops_",
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

    # hack for masking land areas
    glw4_data = load_glw4_data(resolution=2)
    land_mask = glw4_data[0] >= 0
    delta_masked = np.ma.masked_where(~land_mask, delta)

    plt.figure(figsize=(12, 6))
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 8})
    plt.title(f"Change in {prefix} | {year_start}-{year_end}".upper(), weight="bold")
    plt.imshow(
        delta_masked,
        cmap="coolwarm",
        vmin=-np.max(np.abs(delta_masked)),
        vmax=np.max(np.abs(delta_masked)),
        interpolation="nearest",
    )
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.colorbar(label="Delta".upper(), fraction=0.04)
    plt.show()

if __name__ == "__main__":
    plot_delta()