import numpy as np
import cv2

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score

import joblib
import click
from rich import print
import os

from cattlelogue.datasets import build_dataset, load_rf_results, load_aglw_data, load_glw4_data

from hyperopt import fmin, tpe, hp, Trials, space_eval
space = {
    "n_estimators": hp.choice("n_estimators", [100, 200, 300, 400, 500, 600, 800]),
    "max_depth": hp.choice("max_depth", [3, 4, 5, 6, 8, 10]),
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),
    "min_child_weight": hp.choice("min_child_weight", [1, 5, 10]),
}

# CROP: n_est = 20, max_depth = 4
# PASTURE: n_est = 100, max_depth = 4


@click.command()
@click.option(
    "--test_size", type=float, default=0.1, help="Proportion of data to use for testing"
)
@click.option(
    "--n_estimators", type=int, default=500, help="Number of estimators for AdaBoost"
)
@click.option(
    "--ground_truth",
    type=str,
    default="aglw_data",
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
    labels = np.where(labels >= 0, labels, 0)

    livestock_data = dataset["livestock_density"]

    # Filter out invalid data points
    valid_indices = np.where(livestock_data >= 0)[0]
    feature_vectors = feature_vectors[valid_indices]
    livestock_data = livestock_data.reshape(-1, 1)
    labels = labels[valid_indices]

    # binary
    labels = np.where(labels >= 2, 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(
        feature_vectors, labels, test_size=test_size, random_state=42
    )
    print(f"Starting training run with {len(X_train)} samples")
    print(f"Training set shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Sample of training data: {X_train[0]}")

    # Initialize and fit the model
    # model = xgb.XGBRegressor(
    #     n_estimators=n_estimators,
    #     max_depth=6,
    #     learning_rate=0.1,
    #     objective="reg:squarederror",
    #     tree_method="hist",
    #     random_state=42,
    # )

    dataset = build_dataset(year=1961, process_ee=True)
    feature_vectors_2, livestock_data, aglw_data = (
        dataset["features"],
        dataset["livestock_density"],
        dataset["aglw_data"],
    )

    crop_results_2 = load_rf_results("crops_")[1961]
    crop_results_2 = crop_results_2.reshape(-1, 1)
    pasture_results_2 = load_rf_results("pasture_")[1961].reshape(-1, 1)
    livestock_unet_results_2 = load_rf_results("livestock_")[1961].reshape(-1, 1)
    feature_vectors_2 = np.concatenate(
        [feature_vectors_2, crop_results_2, pasture_results_2, livestock_unet_results_2], axis=1
    )

    feature_vectors_2 = feature_vectors_2[valid_indices]
    aglw_flat = aglw_data.flatten()
    aglw_flat = aglw_flat[valid_indices]

    def objective(params):
        print(params)
        model = xgb.XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            min_child_weight=params['min_child_weight'],
            objective="binary:logistic",
            tree_method="hist",
            random_state=42,
        )
        model.fit(X_train, y_train.ravel())
        y_pred = model.predict_proba(feature_vectors_2)[:, 1]
        y_pred = y_pred.flatten()


        auc = roc_auc_score(aglw_flat >= 1, y_pred)
        return {'loss': -auc, 'status': 'ok'}

    # model = xgb.XGBClassifier(
    #     n_estimators=n_estimators,
    #     max_depth=4,
    #     learning_rate=0.1,
    #     objective="binary:logistic",
    #     tree_method="hist",
    #     random_state=42,
    # )

    # run hyperparameter optimization
    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=10, trials=trials)
    best = space_eval(space, best)
    print("Best hyperparameters:", best)
    model = xgb.XGBClassifier(
        n_estimators=best['n_estimators'],
        max_depth=best['max_depth'],
        learning_rate=best['learning_rate'],
        min_child_weight=best['min_child_weight'],
        objective="binary:logistic",
        tree_method="hist",
        random_state=42,
    )


    print("Model parameters:", model.get_params())

    model.fit(X_train, y_train.ravel())

    joblib.dump(model, os.path.join("cattlelogue/outputs/", output))
    print(f"Model saved to {output}")

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    y_pred = model.predict_proba(feature_vectors_2)[:, 1]
    auc = roc_auc_score(aglw_flat >= 1, y_pred)
    print("AUC Score:", auc)


if __name__ == "__main__":
    train_model()
