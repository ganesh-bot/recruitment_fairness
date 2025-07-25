# src/recruitment_fairness/models/catboost_net.py

from catboost import CatBoostClassifier, Pool


class CatBoostNet:
    def __init__(self, cat_features=None, **model_params):
        """
        Wrapper around CatBoostClassifier.
        :param cat_features: list of column names or indices to treat as categorical
        :param model_params: any CatBoostClassifier constructor args
        """
        # set defaults, then override with anything passed in
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
        self.model = CatBoostClassifier(**defaults, cat_features=self.cat_features)

    def fit(
        self,
        train_data,
        train_labels=None,
        eval_set=None,
        early_stopping_rounds: int = None,
    ):
        """
        train_data / eval_set can be Pools or (X, y) tuples.
        """
        fit_kwargs = {}
        if early_stopping_rounds is not None:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

        # If user passed a Pool for train_data
        if isinstance(train_data, Pool):
            self.model.fit(
                train_data, eval_set=eval_set, use_best_model=True, **fit_kwargs
            )
        else:
            # assume numpy arrays or DataFrames
            X_train, y_train = train_data, train_labels
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
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
