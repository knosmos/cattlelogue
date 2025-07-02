import numpy as np

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import joblib
import click
from rich import print

from cattlelogue.datasets import build_dataset


@click.command()
@click.option(
    "--test_size", type=float, default=0.8, help="Proportion of data to use for testing"
)
@click.option(
    "--n_estimators", type=int, default=50, help="Number of estimators for AdaBoost"
)
@click.option(
    "--output", type=str, default="livestock_model.joblib", help="Output model file"
)
def train_model(test_size, n_estimators, output):
    """
    Train a livestock density prediction model using AdaBoost with Random Forest as the base estimator.
    """
    # Load datasets
    dataset = build_dataset()
    feature_vectors = dataset["features"]
    livestock_data = dataset["livestock_density"]
    labels = livestock_data.reshape(-1, 1)

    # Filter out invalid data points
    valid_indices = np.where(labels >= 0)[0]
    feature_vectors = feature_vectors[valid_indices]
    labels = labels[valid_indices]

    X_train, X_test, y_train, y_test = train_test_split(
        feature_vectors, labels, test_size=test_size, random_state=42
    )
    print(f"Starting training run with {len(X_train)} samples")

    # Initialize and fit the model
    model = AdaBoostRegressor(
        estimator=RandomForestRegressor(n_estimators=50, random_state=42, verbose=1),
        n_estimators=n_estimators,
        random_state=42,
    )
    print("Model parameters:", model.get_params())

    model.fit(X_train, y_train.ravel())

    joblib.dump(model, output)
    print(f"Model saved to {output}")

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)


if __name__ == "__main__":
    train_model()
