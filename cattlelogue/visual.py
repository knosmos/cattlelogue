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
def visualize_predictions(
    model_path, year, use_cached, save_predictions, output
) -> None:
    """
    Visualize livestock density predictions for a given year using a trained random
    forest model.

    Args:
        model_path (str): Path to the trained model file.
        year (int): Year for which to generate predictions.
        use_cached (bool): Use cached predictions if available.
        save_predictions (bool): Save predictions to NumPy file.
        output (str): Output file for the prediction visualization.
    """

    if use_cached:
        predictions_path = os.path.join(BASE_DIR, f"outputs/predictions_{year}.npy")
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
            predictions_path = os.path.join(BASE_DIR, f"outputs/predictions_{year}.npy")
            np.save(predictions_path, predictions_reshaped)
            print(f"Predictions saved to {predictions_path}")

    predictions_reshaped = np.ma.masked_where(
        predictions_reshaped == -1, predictions_reshaped
    )

    plt.figure(figsize=(12, 6))
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10})
    plt.title(f"Projected Livestock Density Map - {year}")
    plt.gca().set_facecolor("lightgray")
    plt.imshow(
        predictions_reshaped,
        cmap="viridis",
        interpolation="nearest",
    )
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.colorbar(label="Livestock Density (animals/km^2 * pixel area)", fraction=0.04)
    if output == "":
        output = os.path.join(BASE_DIR, f"outputs/projections_{year}.png")
    plt.savefig(output, bbox_inches="tight", pad_inches=0.1, dpi=500)
    print(f"Projection visualization saved to {output}")
    plt.show()


if __name__ == "__main__":
    visualize_predictions()
