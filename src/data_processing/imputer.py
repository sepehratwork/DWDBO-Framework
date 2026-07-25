"""
KNN Data Imputation Module.
Handles missing value filling for time-series features (Load, Wind, Solar, Price).
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


class TimeSeriesKNNImputer:
    """
    K-Nearest Neighbors Imputer tailored for power system multivariate time series data.
    Preserves local dynamics and spatial/temporal correlations.
    """

    def __init__(self, n_neighbors: int = 5, weights: str = "distance"):
        """
        Initialize the KNN Imputer.

        :param n_neighbors: Number of neighboring samples to use for imputation.
        :param weights: Weight function used in prediction ('uniform' or 'distance').
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights)

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in the provided DataFrame using KNN Imputation.

        :param df: Input DataFrame with potential missing values (NaNs).
        :return: Fully imputed DataFrame.
        """
        feature_cols = df.select_dtypes(include=[np.number]).columns
        imputed_array = self.imputer.fit_transform(df[feature_cols])
        
        df_imputed = df.copy()
        df_imputed[feature_cols] = imputed_array
        return df_imputed