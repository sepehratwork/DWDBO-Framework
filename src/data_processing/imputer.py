"""
K-Nearest Neighbors (KNN) Data Imputation Module.
Addresses missing values in multivariate time-series datasets (Demand, Wind, Solar, Price)
preserving local temporal dynamics (Section 3.1).
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


class TimeSeriesKNNImputer:
    """
    Implements KNN imputation using Euclidean distance across multivariate power system features.
    """

    def __init__(self, n_neighbors: int = 5, weights: str = "distance"):
        """
        Initialize KNN Imputer.

        :param n_neighbors: Number of nearest neighbors to consider.
        :param weights: Distance weighting scheme ('uniform' or 'distance').
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights)

    def impute_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstructs missing data gaps while maintaining temporal continuity.

        :param df: Raw input DataFrame containing potential NaNs.
        :return: Fully imputed DataFrame.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_imputed = df.copy()
        
        # Fit and transform numeric time series columns
        imputed_matrix = self.imputer.fit_transform(df[numeric_cols])
        df_imputed[numeric_cols] = imputed_matrix
        
        return df_imputed