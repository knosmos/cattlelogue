import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os


def mean_dist_from_equator(heatmap):
    """
    Calculate the mean distance from the equator for a given heatmap.

    Parameters:
    heatmap (numpy.ndarray): A 2D array representing the heatmap, where each element corresponds to a latitude.

    Returns:
    float: The mean distance from the equator.
    """
    latitudes = np.arange(-90, 90, 180 / heatmap.shape[0])
    # Calculate distances from the equator (0 degrees latitude)
    distances = np.abs(latitudes)
    heatmap = np.mean(heatmap, axis=1)  # Avg across longitude
    weighted_mean_distance = np.average(distances, weights=heatmap)

    return weighted_mean_distance


def load_inference_results(prefix):
    """
    Load inference results from a specified prefix.

    Parameters:
    prefix (str): The prefix for the file path where inference results are stored.

    Returns:
    dict: A dictionary containing the loaded inference results.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FILE_DIR = os.path.join(BASE_DIR, "outputs")
    outputs = {}
    for file in os.listdir(FILE_DIR):
        if file.startswith(prefix) and file.endswith(".npy"):
            file_path = os.path.join(FILE_DIR, file)
            year = int(file.split("_")[1].split(".")[0])
            outputs[year] = np.load(file_path)
    return outputs


def eval_over_time(prefix, function):
    """
    Evaluate the mean distance from the equator for heatmaps loaded from inference results.

    Parameters:
    prefix (str): The prefix for the file path where inference results are stored.

    Returns:
    dict: A dictionary with filenames as keys and mean distances as values.
    """
    outputs = load_inference_results(prefix)
    results = {}

    for filename, heatmap in outputs.items():
        mean_distance = function(heatmap)
        results[filename] = mean_distance

    return results


def plot_mean_dist(results):
    """
    Plot the results of mean distances from the equator.

    Parameters:
    results (dict): A dictionary with filenames as keys and mean distances as values.
    """
    filenames = list(results.keys())
    distances = list(results.values())

    # styling
    plt.style.use("bmh")
    font_path = "WWF.otf"
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = "Cascadia Code"

    # Raw values
    plt.figure(figsize=(10, 5))
    plt.plot(filenames, distances)
    plt.xlabel("Year".upper())
    plt.ylabel("Distance from Equator (degrees)".upper())
    plt.title(
        "Mean Livestock Distance from Equator".upper(),
        fontsize=20,
        fontfamily=prop.get_name(),
    )

    # Trend line
    z = np.polyfit(range(len(distances)), distances, 1)
    p = np.poly1d(z)
    plt.plot(filenames, p(range(len(distances))), linestyle="--")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    prefix = "livestock_"
    results = eval_over_time(prefix, mean_dist_from_equator)
    plot_mean_dist(results)
