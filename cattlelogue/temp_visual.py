import matplotlib.pyplot as plt
import numpy as np
import click
import os
from cattlelogue.datasets import process_timeseries_data, load_worldcereal_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_PATH = os.path.join(
    BASE_DIR, "inputs/ts_Amon_GISS-E2-1-G_ssp460_r1i1p1f2_gn_20150116-21001216.nc"
)


@click.command()
@click.option(
    "--year", type=int, default=2030, help="Year for which to generate predictions"
)
@click.option(
    "--use_cached",
    is_flag=True,
    default=False,
    help="Use cached predictions if available",
)
@click.option(
    "--save_predictions",
    is_flag=True,
    default=True,
    help="Save predictions to a file",
)
@click.option(
    "--output",
    type=str,
    default="",
    help="Output file for the prediction visualization",
)
def visualize_predictions(year, use_cached, save_predictions, output) -> None:
    """
    Visualize temperature data predictions for a specific year using CMIP data.

    Args:
        year (int): Year for which to generate predictions.
        use_cached (bool): Use cached predictions if available.
        output (str): Output file for the prediction visualization.
    """

    data = process_timeseries_data(
        TEMP_PATH,
        shape=(270, 540),
        year=year,
    )
    # reflect across x-axis
    data = np.flipud(data)
    # np roll
    data = np.roll(data, shift=540 // 2, axis=1)
    data = data[:, :, 0]

    plt.figure(figsize=(12, 6))
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10})
    plt.title(f"CMIP6 Surface Temperature Map - {year}")
    plt.gca().set_facecolor("lightgray")
    plt.imshow(
        data,
        cmap="viridis",
        interpolation="nearest",
    )
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.colorbar(label="Surface Temperature (K)", fraction=0.04)
    if output == "":
        output = os.path.join(BASE_DIR, f"outputs/temp_{year}.png")
    plt.savefig(output, bbox_inches="tight", pad_inches=0.1, dpi=500)
    print(f"Projection visualization saved to {output}")
    plt.show()


if __name__ == "__main__":
    visualize_predictions()
