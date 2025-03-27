import numpy as np
import pandas as pd
import scanpy as sc

from pathlib import Path
from typing import Tuple, List, Dict

# Scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
)
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

# Utilities
from ..utils.constance import UNIVERSAL_RANDOM_STATE, TARGET_COL, DROP_FEAT_TYPE


def prepare_train_data(
    train_data_path: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare training data for training.
    Args:
        - train_data_path (str):        Path to training data

    Returns:
        - X_train (pd.DataFrame):       Training data
        - y_train (pd.DataFrame):       Training labels
        - feat_meta_df (pd.DataFrame):  Features metadata
    """
    # Load training data (need to be a .h5ad file)
    try:
        train_adata = sc.read(train_data_path)
        train_df = train_adata.to_df()
    except Exception as e:
        raise ValueError(f"Invalid data type: {e}")

    # Get features metadata
    feat_meta_df = train_adata.var

    # X_train are columns != TARGET / UNUSED
    X_train = train_df.drop(
        feat_meta_df[feat_meta_df["feat_type"].isin(DROP_FEAT_TYPE)].index,
        axis=1,
    )
    y_train = train_df[TARGET_COL]

    # Shuffle up training data to increase randomness
    X_train = X_train.sample(frac=1, axis=0, random_state=UNIVERSAL_RANDOM_STATE)
    y_train = y_train.loc[X_train.index,]

    return X_train, y_train, feat_meta_df


def run_training(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feat_meta_df: pd.DataFrame,
    clf: object,
    params: Dict,
    n_avail_cores: int,
    scoring: Dict,
    search_method: str = "gridsearchcv",
    randomsearch_n_iter: int = 10000,
    cv_n_splits: int = 5,
    cv_n_repeats: int = 10,
    refit_metrics: str = "weighted_roc_auc",
) -> object:
    """Run gridsearch on training data and train on entire training data with best estimator.

    Args:
        - X_train (pd.DataFrame):       Training data
        - y_train (pd.Series):          Training labels
        - feat_meta_df (pd.DataFrame):  Features metadata
        - clf (object):                 lassifier object
        - params (Dict):                Parameters to search
        - n_avail_cores (int):          Number of available cores
        - scoring (object):             Scoring metrics
        - search_method (str):          Search method to use
        - randomsearch_n_iter (int):    Number of iterations for random search
        - cv_n_splits (int):            Number of cross-validation splits
        - cv_n_repeats (int):           Number of cross-validation repeats
        - refit_metrics (str):          Metrics to refit

    Returns:
        - search (object):              Trained search object. Either GridSearchCV or RandomizedSearchCV
    """
    # If there's a "var_type" column in the feature metadata, use it to determine numerical and categorical features
    if "var_type" in feat_meta_df.columns:
        categorical_features = feat_meta_df[
            (feat_meta_df["var_type"] == "CATEGORICAL")
            & ~(feat_meta_df["feat_type"].isin(DROP_FEAT_TYPE))
        ].index
        numerical_features = feat_meta_df[
            (feat_meta_df["var_type"] == "NUMERIC")
            & ~(feat_meta_df["feat_type"].isin(DROP_FEAT_TYPE))
        ].index

        # Quick sanity check to make sure X_train contains both categorical and numerical features
        assert categorical_features.isin(X_train.columns).all()
        assert numerical_features.isin(X_train.columns).all()

    # Otherwise, implicitly infer numerical and categorical features
    else:
        categorical_features = X_train.select_dtypes(exclude="number").columns
        numerical_features = [
            i for i in X_train.columns if i not in categorical_features
        ]

    # Apply one-hot encode categorical variables
    categorical_pipeline = Pipeline(
        steps=[
            ("one-hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Apply standard (z-score) scaling on numerical variables
    numerical_pipeline = Pipeline(
        steps=[
            ("scaling", StandardScaler()),
        ]
    )

    # Create a column transformer object from all pipelines
    data_processor = ColumnTransformer(
        transformers=[
            (
                "category",
                categorical_pipeline,
                # Only apply one-hot encoder on categorical columns with more than 2 classes
                # One-hot encoding binary categorical variables is not necessary and only add more noises
                [i for i in categorical_features if X_train[i].unique().shape[0] > 2],
            ),
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
        ],
        remainder="passthrough",
    )

    # Create classifer pipeline
    clf_pipeline = Pipeline(
        steps=[
            ("preprocess", data_processor),
            ("clf", clf),
        ]
    )

    # Run either GridSearch or RandomizedSearchCV
    rskf = RepeatedStratifiedKFold(
        n_splits=cv_n_splits,
        n_repeats=cv_n_repeats,
        random_state=UNIVERSAL_RANDOM_STATE,
    )
    if search_method == "gridsearchcv":
        search = GridSearchCV(
            clf_pipeline,
            params,
            n_jobs=n_avail_cores,
            pre_dispatch="n_jobs",
            cv=rskf,
            scoring=scoring,
            refit="weighted_roc_auc",
        )
        print(f"Running hyperparameter tuning with GridSearchCV:\n{search}")
        search.fit(X_train, y_train)

    elif search_method == "randomsearchcv":
        search = RandomizedSearchCV(
            clf_pipeline,
            params,
            n_iter=randomsearch_n_iter,
            n_jobs=n_avail_cores,
            pre_dispatch="n_jobs",
            cv=rskf,
            scoring=scoring,
            refit=refit_metrics,
            random_state=UNIVERSAL_RANDOM_STATE,
        )
        print(f"Running hyperparameter tuning with RandomizedSearchCV:\n{search}")
        search.fit(X_train, y_train)

    else:
        raise ValueError(
            f"Neither gridsearch or randomsearch was speicified: {search_method}"
        )
        exit()

    return search
