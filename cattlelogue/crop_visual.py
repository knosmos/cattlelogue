from joblib import load
import matplotlib.pyplot as plt
import numpy as np
import click
import os

from cattlelogue.datasets import build_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@click.command()
@click.option(
    "--model_path",
    type=str,
    default="livestock_model.joblib",
    help="Path to the trained model",
)
@click.option(
    "--start_year",
    type=int,
    default=2030,
    help="Starting year for which to generate predictions",
)
@click.option(
    "--end_year", type=int, default=2030, help="End year for generating predictions"
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
def visualize_predictions(
    model_path, start_year, end_year, use_cached, save_predictions, output
) -> None:
    """
    Visualize crop density predictions for a range of years using a trained random
    forest model.

    Args:
        model_path (str): Path to the trained model file.
        start_year (int): Starting year for which to generate predictions.
        end_year (int): End year for generating predictions.
        use_cached (bool): Use cached predictions if available.
        save_predictions (bool): Save predictions to NumPy file.
        output (str): Output file for the prediction visualization.
    """

    for year in range(start_year, end_year + 1, 1):
        visualize_predictions_year(
            model_path, year, use_cached, save_predictions, output
        )


def visualize_predictions_year(
    model_path, year, use_cached, save_predictions, output
) -> None:
    """
    Visualize crop density predictions for a given year using a trained random
    forest model.

    Args:
        model_path (str): Path to the trained model file.
        start_year (int): Year for which to generate predictions.
        use_cached (bool): Use cached predictions if available.
        save_predictions (bool): Save predictions to NumPy file.
        output (str): Output file for the prediction visualization.
    """

    predictions_path = os.path.join(BASE_DIR, f"outputs/crops_{year}.npy")
    if use_cached:
        if os.path.exists(predictions_path):
            print(f"Loading cached predictions from {predictions_path}")
            predictions_reshaped = np.load(predictions_path)
        else:
            print(
                f"No cached predictions found for {year}. Generating new predictions."
            )
            use_cached = False

    if not use_cached:
        model = load(model_path)

        dataset = build_dataset(year=year, process_ee=False)
        feature_vectors, livestock_data, glw4_shape = (
            dataset["features"],
            dataset["livestock_density"],
            dataset["glw4_shape"],
        )
        valid_indices = np.where(livestock_data >= 0)[0]

        predictions = model.predict(feature_vectors)

        predictions_masked = np.zeros(glw4_shape[0] * glw4_shape[1])
        predictions_masked[:] = -1
        predictions_masked[valid_indices] = predictions[valid_indices].flatten()
        predictions_reshaped = predictions_masked.reshape(glw4_shape)
        predictions_reshaped = np.roll(
            predictions_reshaped, predictions_reshaped.shape[1] // 2, axis=1
        )
        if save_predictions:
            np.save(predictions_path, predictions_reshaped)
            print(f"Projections saved to {predictions_path}")

    predictions_reshaped = np.ma.masked_where(
        predictions_reshaped == -1, predictions_reshaped
    )

    plt.figure(figsize=(12, 6))
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 8})
    plt.title(f"Projected Cereal Crop Density Map - {year}".upper(), weight="bold")
    plt.imshow(
        predictions_reshaped,
        cmap="plasma",
        interpolation="nearest",
        vmin=0,
        vmax=50,
    )
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.colorbar(label="Probability/Percentage Covered".upper(), fraction=0.04)
    if output == "":
        output = predictions_path.replace(".npy", ".png")
    plt.savefig(output, bbox_inches="tight", pad_inches=0.1, dpi=500)
    plt.savefig(
        os.path.join(BASE_DIR, f"outputs/last_output.png"),
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=500,
    )
    print(f"Projection visualization saved to {output}")
    # plt.show()


if __name__ == "__main__":
    visualize_predictions()
