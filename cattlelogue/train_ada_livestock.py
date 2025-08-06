import numpy as np

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import joblib
import click
from rich import print
import os

from cattlelogue.datasets import build_dataset, load_rf_results

# CROP: n_est = 20, max_depth = 4
# PASTURE: n_est = 100, max_depth = 4


@click.command()
@click.option(
    "--test_size", type=float, default=0.1, help="Proportion of data to use for testing"
)
@click.option(
    "--n_estimators", type=int, default=20, help="Number of estimators for AdaBoost"
)
@click.option(
    "--ground_truth",
    type=str,
    default="livestock_density",
    help="Ground truth dataset to use",
)
@click.option(
    "--output", type=str, default="livestock_model_2.joblib", help="Output model file"
)
def train_model(test_size, n_estimators, ground_truth, output) -> None:
    """
    Train a density prediction model using AdaBoost on decision trees regressors.
    """
    # Load datasets
    dataset = build_dataset(process_ee=True)
    print("Datasets loaded successfully")
    feature_vectors = dataset["features"]
    crop_results = load_rf_results("crops_")[2015].reshape(-1, 1)
    pasture_results = load_rf_results("pasture_")[2015].reshape(-1, 1)
    livestock_unet_results = load_rf_results("livestock_")[2015].reshape(-1, 1)
    feature_vectors = np.concatenate(
        [feature_vectors, crop_results, pasture_results, livestock_unet_results], axis=1
    )
    ground_truth_set = dataset[ground_truth]
    # labels = livestock_data.reshape(-1, 1)
    labels = ground_truth_set.reshape(-1, 1)

    livestock_data = dataset["livestock_density"]

    # Filter out invalid data points
    valid_indices = np.where(livestock_data >= 0)[0]
    feature_vectors = feature_vectors[valid_indices]
    livestock_data = livestock_data.reshape(-1, 1)
    labels = labels[valid_indices]

    X_train, X_test, y_train, y_test = train_test_split(
        feature_vectors, labels, test_size=test_size, random_state=42
    )
    print(f"Starting training run with {len(X_train)} samples")
    print(f"Training set shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Sample of training data: {X_train[0]}")

    # Initialize and fit the model
    model = AdaBoostRegressor(
        # estimator=RandomForestRegressor(n_estimators=50, random_state=42, verbose=1),
        estimator=DecisionTreeRegressor(
            max_depth=4, min_samples_split=10, random_state=42
        ),
        n_estimators=n_estimators,
        random_state=42,
    )
    print("Model parameters:", model.get_params())

    model.fit(X_train, y_train.ravel())

    joblib.dump(model, os.path.join("cattlelogue/outputs/", output))
    print(f"Model saved to {output}")

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)


if __name__ == "__main__":
    train_model()
