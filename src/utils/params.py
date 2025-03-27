import numpy as np
import pandas as pd

# Scikit-learn
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    make_scorer,
)

# Utilities
from .constance import PREDICT_LABELS, UNIVERSAL_RANDOM_STATE

# Define params to tune for each uni-model in the ensemble
ensemble_lr_params = {
    "clf__lr__max_iter": [100, 1000],
    "clf__lr__C": [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1],
    "clf__lr__l1_ratio": [0.2, 0.4, 0.6, 0.8],
    "clf__lr__penalty": ["l1", "l2", "elasticnet"],
    "clf__lr__solver": ["lbfgs", "newton-cg", "saga"],
}
ensemble_rf_params = {
    "clf__rf__n_estimators": [1000, 2000, 5000],
    "clf__rf__max_depth": [6, 8],
    "clf__rf__criterion": ["gini", "entropy"],
}
ensemble_svm_params = {
    "clf__svm__kernel": ["linear", "rbf"],
    "clf__svm__C": [1e-3, 1e-2, 1e-1, 1],
    "clf__svm__gamma": [1e-4, 1e-3, 1e-2, 1e-1],
    "clf__svm__degree": [2, 3, 4],
}


# Model and paramaters
clfs = [
    {
        "model": "logistic_regression",
        "clf": LogisticRegression(random_state=UNIVERSAL_RANDOM_STATE),
        "params": {
            "clf__max_iter": [10, 100, 1000],
            "clf__C": [1e-4, 5e-3, 1e-3, 1e-2, 5e-1, 1.0, 1e1],
            "clf__l1_ratio": np.arange(0.1, 1.0, 0.1),
            "clf__penalty": ["l1", "l2", "elasticnet"],
            "clf__solver": ["liblinear", "lbfgs", "newton-cg", "saga"],
        },
    },
    {
        "model": "random_forest",
        "clf": RandomForestClassifier(random_state=UNIVERSAL_RANDOM_STATE),
        "params": {
            "clf__n_estimators": [1000, 2000, 5000],
            "clf__max_depth": [6, 8, 10],
            "clf__criterion": ["gini", "entropy"],
            "clf__max_features": ["sqrt", 0.5, 0.75],
        },
    },
    {
        "model": "svm",
        "clf": SVC(probability=True, random_state=UNIVERSAL_RANDOM_STATE),
        "params": {
            "clf__kernel": ["linear", "rbf", "sigmoid"],
            "clf__C": [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2],
            "clf__degree": [2, 3, 4, 5],
            "clf__gamma": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
        },
    },
    {
        "model": "ensemble_lr_rf",
        "clf": VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(random_state=UNIVERSAL_RANDOM_STATE)),
                ("rf", RandomForestClassifier(random_state=UNIVERSAL_RANDOM_STATE)),
            ],
            voting="soft",
        ),
        "params": ensemble_lr_params | ensemble_rf_params,
    },
    {
        "model": "ensemble_lr_rf_svm",
        "clf": VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(random_state=UNIVERSAL_RANDOM_STATE)),
                ("rf", RandomForestClassifier(random_state=UNIVERSAL_RANDOM_STATE)),
                (
                    "svm",
                    SVC(probability=True, random_state=UNIVERSAL_RANDOM_STATE),
                ),
            ],
            voting="soft",
        ),
        "params": ensemble_lr_params | ensemble_rf_params | ensemble_svm_params,
    },
]

# Convert clfs to dataframe format
clfs_df = pd.DataFrame(clfs)

# Creating list of scores to keep track of during 5-fold cross validation
scoring = {
    "acc": make_scorer(accuracy_score),
    # NOTE: For binary classification, micro-roc_auc and weighted-roc_auc are the same
    "micro_roc_auc": make_scorer(
        roc_auc_score, average="micro", labels=PREDICT_LABELS, needs_proba=True
    ),
    "weighted_roc_auc": make_scorer(
        roc_auc_score, average="weighted", labels=PREDICT_LABELS, needs_proba=True
    ),
    # F1
    "micro_f1": make_scorer(f1_score, average="micro", labels=PREDICT_LABELS),
    "weighted_f1": make_scorer(f1_score, average="weighted", labels=PREDICT_LABELS),
    # Precision
    "micro_prec": make_scorer(precision_score, average="micro", labels=PREDICT_LABELS),
    "weighted_prec": make_scorer(
        precision_score, average="weighted", labels=PREDICT_LABELS
    ),
    # Recall
    "micro_rec": make_scorer(recall_score, average="micro", labels=PREDICT_LABELS),
    "micro_rec": make_scorer(recall_score, average="weighted", labels=PREDICT_LABELS),
}
