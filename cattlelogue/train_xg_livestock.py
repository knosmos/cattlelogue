import numpy as np

import xgboost as xgb
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
    "--test_size", type=float, default=0.2, help="Proportion of data to use for testing"
)
@click.option(
    "--n_estimators", type=int, default=400, help="Number of estimators for AdaBoost"
)
@click.option(
    "--ground_truth",
    type=str,
    default="aglw_data",
    help="Ground truth dataset to use",
)
@click.option(
    "--output", type=str, default="livestock_model_3.joblib", help="Output model file"
)
def train_model(test_size, n_estimators, ground_truth, output) -> None:
    """
    Train a density prediction model using AdaBoost on decision trees regressors.
    """
    # Load datasets
    all_feature_vectors = None
    all_ground_truth = None
    for year in range(1980, 2016, 5):
        print(f"Loading data for year {year}")
        dataset = build_dataset(process_ee=True, year=year)
        print("Datasets loaded successfully")
        feature_vectors = dataset["features"]
        crop_results = load_rf_results("crops_")[year].reshape(-1, 1)
        pasture_results = load_rf_results("pasture_")[year].reshape(-1, 1)
        livestock_unet_results = load_rf_results("livestock_")[year].reshape(-1, 1)
        feature_vectors = np.concatenate(
            [feature_vectors, crop_results, pasture_results, livestock_unet_results], axis=1
        )
        ground_truth_set = dataset[ground_truth]
        if all_feature_vectors is None:
            all_feature_vectors = feature_vectors
            all_ground_truth = ground_truth_set
        else:
            all_feature_vectors = np.concatenate(
                [all_feature_vectors, feature_vectors], axis=0
            )
            all_ground_truth = np.concatenate(
                [all_ground_truth, ground_truth_set], axis=0
            )
    feature_vectors = all_feature_vectors
    ground_truth_set = all_ground_truth
    # labels = livestock_data.reshape(-1, 1)
    labels = ground_truth_set.reshape(-1, 1)
    labels = np.where(labels >= 0, labels, 0)

    livestock_data = dataset["livestock_density"]

    # Filter out invalid data points
    valid_indices = np.where(livestock_data >= 0)[0]
    feature_vectors = feature_vectors[valid_indices]
    livestock_data = livestock_data.reshape(-1, 1)
    labels = labels[valid_indices]

    # binary
    labels = np.where(labels >= 1, 1, 0)

    print(f"Feature vectors shape: {feature_vectors.shape}")
    X_train, X_test, y_train, y_test = train_test_split(
        feature_vectors, labels, test_size=test_size, random_state=42
    )
    print(f"Starting training run with {len(X_train)} samples")
    print(f"Training set shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Sample of training data: {X_train[0]}")

    # Initialize and fit the model
    # model = xgb.XGBRegressor(
    #     n_estimators=n_estimators,
    #     max_depth=4,
    #     learning_rate=0.1,
    #     objective="reg:squarederror",
    #     tree_method="hist",
    #     random_state=42,
    # )
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.1,
        objective="binary:logistic",
        tree_method="hist",
        random_state=42,
    )
    print("Model parameters:", model.get_params())

    model.fit(X_train, y_train.ravel())
    weights = model.get_booster().get_score(importance_type="weight")
    for x in range(78):
        print(weights.get(f"f{x}", 0))

    joblib.dump(model, os.path.join("cattlelogue/outputs/", output))
    print(f"Model saved to {output}")

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)


if __name__ == "__main__":
    train_model()
