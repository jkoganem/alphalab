"""Machine learning based alpha models.

This module implements alpha models using scikit-learn for classification
and regression on financial features.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class MLAlpha:
    """Machine learning alpha model.

    Uses supervised learning to predict future returns or return direction
    from features.

    Parameters
    ----------
    model_type : Literal["classification", "regression"], default "classification"
        Whether to predict return direction (classification) or magnitude (regression)
    estimator : Literal["logistic", "ridge", "gbm"], default "gbm"
        Estimator type to use
    forward_horizon : int, default 5
        Number of days ahead to predict
    train_window : int, default 504
        Rolling training window size (2 years ~= 504 days)
    feature_cols : list[str] | None, optional
        Specific feature columns to use (if None, uses all numeric columns)
    scale_features : bool, default True
        Whether to standardize features

    Examples
    --------
    >>> ml_alpha = MLAlpha(model_type="classification", estimator="gbm")
    >>> alpha = ml_alpha.score(features)
    """

    def __init__(
        self,
        model_type: Literal["classification", "regression"] = "classification",
        estimator: Literal["logistic", "ridge", "gbm"] = "gbm",
        forward_horizon: int = 5,
        train_window: int = 504,
        feature_cols: list[str] | None = None,
        scale_features: bool = True,
    ) -> None:
        """Initialize ML alpha model."""
        self.model_type = model_type
        self.estimator_type = estimator
        self.forward_horizon = forward_horizon
        self.train_window = train_window
        self.feature_cols = feature_cols
        self.scale_features = scale_features

        self.scaler = StandardScaler() if scale_features else None
        self.model: BaseEstimator | None = None
        self.feature_importance_: pd.Series | None = None

        logger.info(
            f"Initialized MLAlpha with model_type={model_type}, "
            f"estimator={estimator}, horizon={forward_horizon}"
        )

    def _create_estimator(self) -> BaseEstimator:
        """Create the ML estimator.

        Returns
        -------
        BaseEstimator
            Scikit-learn estimator
        """
        if self.model_type == "classification":
            if self.estimator_type == "logistic":
                return LogisticRegression(max_iter=1000, random_state=42)
            elif self.estimator_type == "gbm":
                return GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                )
            else:
                msg = f"Unknown classifier: {self.estimator_type}"
                raise ValueError(msg)
        else:  # regression
            if self.estimator_type == "ridge":
                return Ridge(alpha=1.0, random_state=42)
            elif self.estimator_type == "gbm":
                return GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                )
            else:
                msg = f"Unknown regressor: {self.estimator_type}"
                raise ValueError(msg)

    def _prepare_labels(self, features: pd.DataFrame) -> pd.Series:
        """Create labels from forward returns.

        Parameters
        ----------
        features : pd.DataFrame
            Features including return columns

        Returns
        -------
        pd.Series
            Labels (binary for classification, continuous for regression)
        """
        # Look for forward return column or create it
        fwd_ret_col = f"ret_{self.forward_horizon}d"

        if fwd_ret_col in features.columns:
            fwd_returns = features[fwd_ret_col]
        else:
            # Look for 1-day returns and shift
            if "ret_1d" in features.columns:
                fwd_returns = features["ret_1d"].groupby(level="symbol").shift(
                    -self.forward_horizon
                )
            else:
                msg = f"Need '{fwd_ret_col}' or 'ret_1d' in features for label creation"
                raise ValueError(msg)

        if self.model_type == "classification":
            # Binary: 1 if positive return, 0 if negative
            labels = (fwd_returns > 0).astype(int)
        else:
            # Regression: use raw forward returns
            labels = fwd_returns

        return labels

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate ML-based alpha scores.

        Uses rolling window training to avoid look-ahead bias.

        Parameters
        ----------
        features : pd.DataFrame
            Feature DataFrame with MultiIndex (date, symbol)

        Returns
        -------
        pd.DataFrame
            Alpha scores (predictions from ML model)
        """
        logger.info("Generating ML alpha scores with rolling window training")

        # Select feature columns
        if self.feature_cols is None:
            # Use all numeric columns except returns (to avoid leakage)
            exclude_patterns = ["ret_", "alpha"]
            feature_cols = [
                col
                for col in features.columns
                if not any(pattern in col for pattern in exclude_patterns)
            ]
        else:
            feature_cols = self.feature_cols

        logger.info(f"Using {len(feature_cols)} features for ML model")

        # Prepare labels
        labels = self._prepare_labels(features)

        # Get unique dates
        dates = features.index.get_level_values("date").unique().sort_values()

        # Initialize predictions
        predictions = pd.Series(np.nan, index=features.index)

        # Rolling window training and prediction
        for i, date in enumerate(dates):
            if i < self.train_window:
                # Not enough history yet
                continue

            # Training window
            train_start_idx = i - self.train_window
            train_dates = dates[train_start_idx:i]

            # Get training data
            train_mask = features.index.get_level_values("date").isin(train_dates)
            X_train = features.loc[train_mask, feature_cols]
            y_train = labels.loc[train_mask]

            # Remove NaN
            valid_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]

            if len(X_train) < 50:  # Need minimum samples
                continue

            # Scale features
            if self.scaler is not None:
                X_train_scaled = self.scaler.fit_transform(X_train)
            else:
                X_train_scaled = X_train.values

            # Train model
            self.model = self._create_estimator()

            try:
                self.model.fit(X_train_scaled, y_train)

                # Predict for current date
                pred_mask = features.index.get_level_values("date") == date
                X_pred = features.loc[pred_mask, feature_cols]

                # Remove NaN
                valid_pred = ~X_pred.isna().any(axis=1)
                X_pred_valid = X_pred[valid_pred]

                if len(X_pred_valid) > 0:
                    if self.scaler is not None:
                        X_pred_scaled = self.scaler.transform(X_pred_valid)
                    else:
                        X_pred_scaled = X_pred_valid.values

                    # Get predictions
                    if self.model_type == "classification":
                        # Use probability of positive class as alpha score
                        preds = self.model.predict_proba(X_pred_scaled)[:, 1]
                        # Center around 0 (subtract 0.5)
                        preds = preds - 0.5
                    else:
                        # Use predicted returns directly
                        preds = self.model.predict(X_pred_scaled)

                    predictions.loc[X_pred_valid.index] = preds

            except Exception as e:
                logger.warning(f"Error training/predicting for {date}: {e}")
                continue

        # Extract feature importance from last model
        if self.model is not None and hasattr(self.model, "feature_importances_"):
            self.feature_importance_ = pd.Series(
                self.model.feature_importances_, index=feature_cols
            ).sort_values(ascending=False)
            logger.info(
                f"Top 5 features: {self.feature_importance_.head().to_dict()}"
            )

        result = pd.DataFrame({"alpha": predictions}, index=features.index)

        n_valid = result["alpha"].notna().sum()
        logger.info(f"Generated {n_valid} ML alpha predictions")

        return result

    def get_feature_importance(self) -> pd.Series | None:
        """Get feature importance from last trained model.

        Returns
        -------
        pd.Series | None
            Feature importances (if available)
        """
        return self.feature_importance_


class SimpleMLAlpha:
    """Simplified ML alpha using single train-test split.

    Faster but less realistic than rolling window approach.
    Useful for quick prototyping.

    Parameters
    ----------
    model_type : Literal["classification", "regression"], default "classification"
        Prediction type
    forward_horizon : int, default 5
        Prediction horizon
    train_fraction : float, default 0.7
        Fraction of data to use for training

    Examples
    --------
    >>> simple_ml = SimpleMLAlpha(model_type="classification")
    >>> alpha = simple_ml.score(features)
    """

    def __init__(
        self,
        model_type: Literal["classification", "regression"] = "classification",
        forward_horizon: int = 5,
        train_fraction: float = 0.7,
    ) -> None:
        """Initialize simple ML alpha."""
        self.model_type = model_type
        self.forward_horizon = forward_horizon
        self.train_fraction = train_fraction

        logger.info(f"Initialized SimpleMLAlpha with train_fraction={train_fraction}")

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate alpha scores with single train-test split.

        Parameters
        ----------
        features : pd.DataFrame
            Features

        Returns
        -------
        pd.DataFrame
            Alpha scores
        """
        logger.info("Generating simple ML alpha (single split)")

        # Use MLAlpha but with single large training window
        dates = features.index.get_level_values("date").unique()
        train_size = int(len(dates) * self.train_fraction)

        ml_alpha = MLAlpha(
            model_type=self.model_type,
            estimator="gbm",
            forward_horizon=self.forward_horizon,
            train_window=train_size,
        )

        return ml_alpha.score(features)


def create_ml_alpha(
    alpha_type: str = "rolling", **kwargs: object
) -> MLAlpha | SimpleMLAlpha:
    """Factory function to create ML alpha models.

    Parameters
    ----------
    alpha_type : str, default "rolling"
        Type: "rolling" or "simple"
    **kwargs : object
        Alpha-specific parameters

    Returns
    -------
    MLAlpha | SimpleMLAlpha
        ML alpha instance
    """
    if alpha_type == "rolling":
        return MLAlpha(**kwargs)  # type: ignore[arg-type]
    elif alpha_type == "simple":
        return SimpleMLAlpha(**kwargs)  # type: ignore[arg-type]
    else:
        msg = f"Unknown ML alpha type: {alpha_type}"
        raise ValueError(msg)
