from joblib import load
import matplotlib.pyplot as plt
import numpy as np
import click
import os

from cattlelogue.datasets import build_dataset, load_rf_results

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

    predictions_path = os.path.join(BASE_DIR, f"outputs/livestock2_{year}.npy")
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

        dataset = build_dataset(year=year, process_ee=True)
        feature_vectors, livestock_data, glw4_shape = (
            dataset["features"],
            dataset["livestock_density"],
            dataset["glw4_shape"],
        )
        feature_vectors = feature_vectors.reshape(-1, feature_vectors.shape[-1])
        crop_results = load_rf_results("crops_")[year].reshape(-1, 1)
        pasture_results = load_rf_results("pasture_")[year].reshape(-1, 1)
        livestock_unet_results = load_rf_results("livestock_")[year].reshape(-1, 1)
        feature_vectors = np.concatenate(
            [feature_vectors, crop_results, pasture_results, livestock_unet_results], axis=1
        )
        valid_indices = np.where(livestock_data >= 0)[0]

        predictions = model.predict_proba(feature_vectors)[:, 1]
        # Normalize livestock UNet results to [0, 1] range
        predictions = predictions.reshape(-1, 1)
        predictions = predictions - np.mean(predictions) / (np.std(predictions) * 2) + 0.5
        
        # fit such that most of livestock scores are between 0 and 1 (stddev)
        livestock_unet_normalized = (
            livestock_unet_results - np.mean(livestock_unet_results)
        ) / (np.std(livestock_unet_results) * 2) + 0.5

        # predictions = model.predict(feature_vectors)
        print(model.get_booster().get_score(importance_type='weight'))

        print(predictions.shape)
        predictions = predictions.flatten()
        WEIGHT_FACTOR = 0.4
        predictions = (
            predictions * (1 - WEIGHT_FACTOR)
            + WEIGHT_FACTOR * livestock_unet_normalized.flatten()
        )

        predictions_masked = np.zeros(glw4_shape[0] * glw4_shape[1])
        predictions_masked[:] = -1
        predictions_masked[valid_indices] = predictions[valid_indices].flatten()
        predictions_reshaped = predictions_masked.reshape(glw4_shape)
        if save_predictions:
            np.save(predictions_path, predictions_reshaped)
            print(f"Projections saved to {predictions_path}")

    predictions_reshaped = np.ma.masked_where(
        predictions_reshaped == -1, predictions_reshaped
    )

    plt.figure(figsize=(12, 6))
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 8})
    plt.title(f"Projected Pasture Density Map - {year}".upper(), weight="bold")
    plt.imshow(
        predictions_reshaped,
        cmap="viridis",
        interpolation="nearest",
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
    plt.close()
    # plt.show()


if __name__ == "__main__":
    visualize_predictions()
