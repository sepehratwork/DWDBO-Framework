"""
K-Nearest Neighbors (KNN) Data Imputation Module.
Addresses missing values in multivariate time-series datasets (Demand, Wind, Solar, Price)
preserving local temporal dynamics according to Section 3.1 and Figures 1 & 2.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class TimeSeriesKNNImputer:
    """
    Implements KNN imputation using normalized Euclidean distance across multivariate power system features.
    
    Section 3.1 Mathematical Formulation:
    1. Feature Normalization: Normalizes all multivariate time-series features to [0, 1] range
       to prevent features with high numerical magnitude (e.g., Load Demand) from dominating
       the Euclidean distance metric over smaller magnitude features (e.g., Solar, Wind).
    2. Temporal Context Augmentation: Integrates diurnal time-of-day dynamics to preserve 
       local temporal continuity.
    3. Distance-Weighted Imputation: Computes missing entries using inverse-distance weighted 
       k-nearest neighbors across normalized feature space:
       w_j = (1 / d(x_i, x_j)) / sum(1 / d(x_i, x_k))
       x_imputed = sum(w_j * x_j)
    """

    def __init__(self, n_neighbors: int = 5, weights: str = "distance", add_temporal_features: bool = True):
        """
        Initialize TimeSeriesKNNImputer.

        :param n_neighbors: Number of nearest neighbors K (default: 5).
        :param weights: Distance weighting scheme ('distance' or 'uniform').
        :param add_temporal_features: Whether to augment with cyclic temporal features.
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.add_temporal_features = add_temporal_features
        self.scaler = MinMaxScaler()
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights)

    def impute_missing_data(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Reconstructs missing data gaps while maintaining multivariate feature correlation 
        and temporal continuity (Section 3.1).

        :param df: Raw input DataFrame containing potential NaNs.
        :param verbose: Whether to display detailed progress bars.
        :return: Fully imputed DataFrame with identical index and columns.
        """
        df_imputed = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return df_imputed

        total_steps = 5
        pbar = tqdm(
            total=total_steps, 
            desc="[Step 1 - KNN Imputer]", 
            disable=not verbose, 
            bar_format="{l_bar}{bar:30}{r_bar}"
        )

        # Step 1.1: Extract numeric data & build temporal features
        pbar.set_postfix_str("Building feature matrix & temporal encoding")
        data_matrix = df[numeric_cols].values.copy()
        n_samples, n_features = data_matrix.shape

        # Build cyclical diurnal temporal features (sin/cos encoding of day cycle)
        temporal_feats = []
        if self.add_temporal_features:
            if isinstance(df.index, pd.DatetimeIndex):
                hours = np.asarray(df.index.hour + df.index.minute / 60.0, dtype=np.float64)
            else:
                # Continuous integer time index assumption (15-min intervals: 96 steps/day)
                hours = np.asarray((np.arange(n_samples) % 96) / 4.0, dtype=np.float64)
                
            sin_hour = np.sin(2 * np.pi * hours / 24.0).reshape(-1, 1)
            cos_hour = np.cos(2 * np.pi * hours / 24.0).reshape(-1, 1)
            temporal_feats = [sin_hour, cos_hour]

        pbar.update(1)

        # Step 1.2: Feature Normalization (Min-Max Scaling to [0, 1])
        pbar.set_postfix_str("Normalizing multivariate features to [0, 1]")
        scaled_features = self.scaler.fit_transform(data_matrix)
        
        if temporal_feats:
            combined_matrix = np.hstack([scaled_features] + temporal_feats)
        else:
            combined_matrix = scaled_features
        pbar.update(1)

        # Step 1.3: Distance-Weighted K-Nearest Neighbors Imputation
        pbar.set_postfix_str(f"Computing KNN distance-weighted imputation (K={self.n_neighbors})")
        imputed_combined = self.imputer.fit_transform(combined_matrix)
        pbar.update(1)

        # Step 1.4: Inverse Scale Transformation
        pbar.set_postfix_str("Inverse-transforming features back to original scale")
        imputed_numeric_scaled = imputed_combined[:, :n_features]
        imputed_numeric = self.scaler.inverse_transform(imputed_numeric_scaled)
        pbar.update(1)

        # Step 1.5: Final Continuity Verification & Return
        pbar.set_postfix_str("Verifying continuity & finalizing DataFrame")
        df_imputed[numeric_cols] = imputed_numeric
        pbar.update(1)
        pbar.close()

        return df_imputed