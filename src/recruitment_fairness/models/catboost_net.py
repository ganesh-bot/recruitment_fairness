# src/recruitment_fairness/models/catboost_net.py

from catboost import CatBoostClassifier, Pool
import numpy as np
from collections import Counter


def _compute_balanced_weights(y):
    """
    Inverse frequency weighting: weight = total / (n_classes * count)
    """
    y_array = np.asarray(y)
    counts = Counter(y_array.tolist())
    if len(counts) <= 1:
        return np.ones_like(y_array, dtype=float)
    total = len(y_array)
    n_classes = len(counts)
    class_weight = {cls: total / (n_classes * cnt) for cls, cnt in counts.items()}
    return np.vectorize(lambda v: class_weight[v])(y_array)


class CatBoostNet:
    def __init__(
        self,
        cat_features=None,
        auto_class_weight: bool = False,
        **model_params,
    ):
        """
        Wrapper around CatBoostClassifier with optional balancing.

        :param cat_features: list of column names or indices to treat as categorical
        :param auto_class_weight: if True, apply custom inverse-frequency sample weights
        :param model_params: CatBoostClassifier args; may include CatBoost's own
                             `auto_class_weights` (e.g., 'Balanced' or 'SqrtBalanced').
        """
        defaults = dict(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            eval_metric="AUC",
            verbose=100,
            random_state=42,
        )
        defaults.update(model_params)

        self.cat_features = cat_features or []
        self.auto_class_weight = auto_class_weight

        # Extract CatBoost's internal auto_class_weights if provided (e.g., 'Balanced')
        self._cb_auto_class_weights = None
        if "auto_class_weights" in defaults:
            self._cb_auto_class_weights = defaults.pop("auto_class_weights")
            # pass it explicitly when constructing
        self.model = CatBoostClassifier(
            **defaults,
            cat_features=self.cat_features,
            **(
                {"auto_class_weights": self._cb_auto_class_weights}
                if self._cb_auto_class_weights is not None
                else {}
            ),
        )

    def _prepare_pool(self, X, y=None, sample_weight=None):
        if isinstance(X, Pool):
            return X
        # if explicit sample_weight given, use it; else if custom auto_class_weight requested, compute
        if sample_weight is None and self.auto_class_weight and y is not None:
            sample_weight = _compute_balanced_weights(y)
        return Pool(data=X, label=y, weight=sample_weight, cat_features=self.cat_features)

    def fit(
        self,
        train_data,
        train_labels=None,
        eval_set=None,
        early_stopping_rounds: int = None,
        sample_weight=None,
        eval_sample_weight=None,
    ):
        """
        train_data / eval_set can be Pools or (X, y) tuples.
        """
        fit_kwargs = {}
        if early_stopping_rounds is not None:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

        # Training pool
        if isinstance(train_data, Pool):
            train_pool = train_data
        else:
            X_train, y_train = train_data, train_labels
            train_pool = self._prepare_pool(X_train, y_train, sample_weight)

        # Eval pool
        prepared_eval = None
        if eval_set is not None:
            if isinstance(eval_set, Pool):
                prepared_eval = eval_set
            else:
                X_val, y_val = eval_set
                prepared_eval = self._prepare_pool(X_val, y_val, eval_sample_weight)

        self.model.fit(
            train_pool,
            eval_set=prepared_eval,
            use_best_model=True,
            **fit_kwargs,
        )
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: str):
        self.model.save_model(path)

    def load(self, path: str):
        self.model.load_model(path)
