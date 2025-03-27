import mlflow
import argparse

import numpy as np
import pandas as pd
import pickle as pkl

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

# Import utilities scripts
from funcs.train_func import run_training, prepare_train_data
from utils.params import scoring, clfs_df
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
)
from utils.constance import GLOBAL_EXPT, PREDICT_LABELS


def main(
    train_data_path: str,
    ml_model: str,
    output_dir: str,
    n_cores: int,
    search_method: str = "gridsearchcv",
) -> None:
    """Main function to run stratified, repeated, k-fold cross-validation on specified features type for a given model

    Args:
        - train_data_path (str):     Path to training data (.h5ad)
        - ml_model (str):            Name of model to train. Must exist in src.utils.params.clfs
        - output_dir (str):          Path to output directory
        - n_cores (int):             Number of available cores used for training
        - search_method (str):       Type of search method to use
    """
    # Prepare train and test data
    (X_train, y_train, feat_meta_df) = prepare_train_data(
        train_data_path=train_data_path,
    )

    # Get classifier to train
    clf_df = clfs_df[clfs_df["model"] == ml_model]
    # Quick sanity check to make sure that we only have 1 entry (i.e. 1 row) for each model
    assert clf_df.shape[0] == 1
    # Then get classifer object and parameters dict
    clf = clf_df["clf"].values[0]
    params = clf_df["params"].values[0]

    # Create MLflow experiment if specified
    if GLOBAL_EXPT:
        mlflow_expt = mlflow.set_experiment(GLOBAL_EXPT)

    # Run training within a mlflow session
    with mlflow.start_run(
        run_name=f"{ml_model}_{search_method}", experiment_id=mlflow_expt.experiment_id
    ):
        # Autologging
        mlflow.sklearn.autolog(log_input_examples=True, log_post_training_metrics=False)

        # Run training
        search = run_training(
            X_train=X_train,
            y_train=y_train,
            feat_meta_df=feat_meta_df,
            clf=clf,
            params=params,
            n_avail_cores=n_cores,
            scoring=scoring,
            search_method=search_method,
        )

        # Log weighted f1 and roc-auc scores on entire training data
        y_pred = search.predict(X_train)
        y_proba = search.predict_proba(X_train)
        mlflow.log_metrics(
            {
                "weighted_f1": f1_score(
                    y_train, y_pred, average="weighted", labels=PREDICT_LABELS
                ),
                "weighted_roc_auc": roc_auc_score(
                    y_train, y_proba[:, 1], average="weighted", labels=PREDICT_LABELS
                ),
            }
        )

    # If output_dir is available, save search object
    # Otherwise mlflow already saves the trained search object
    if output_dir:
        pkl.dump(
            search,
            open(Path(output_dir).joinpath(f"{ml_model}").with_suffix(".pkl"), "wb"),
        )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description=__doc__)

    # Required arguments
    parser.add_argument(
        "--train-data-path",
        type=str,
        required=True,
        help="Path to features metadata",
    )
    parser.add_argument(
        "--ml-model",
        type=str,
        required=True,
        choices=[clfs_df["model"].tolist()],
        help="Name of ML model. Must exist in utils.params.clfs",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        help="Number of available cores used for training.",
    )

    # Optional arguments
    parser.add_argument(
        "--search-method",
        type=str,
        default="gridsearchcv",
        choices=["gridsearchcv", "randomsearchcv"],
        help="Type of search method to use",
    )

    # Parse arguments
    args, unknowns = parser.parse_known_args()

    # Run main function
    main(
        train_data_path=args.train_data_path,
        ml_model=args.ml_model,
        output_dir=args.output_dir,
        search_method=args.search_method,
        n_cores=args.n_cores,
    )
