from typing import List, Optional

import numpy as np
import pandas as pd


class PreprocessingArtifacts:
    """
    Bundle of fitted preprocessing artifacts from training, with a unified
    transform(df) method that reproduces PCA components expected at inference,
    while preserving non-PCA passthrough features present in the input.

    Steps reproduced:
    - Apply QuantileTransformer to numeric feature columns (as trained)
    - Apply StandardScaler to numeric feature columns (as trained)
    - Order/select columns matching PCA input features and fill missing with 0
        - Apply PCA transform and scale PCA outputs with the trained scaler
        - Concatenate PCA components with passthrough columns (all columns not used
            as PCA inputs), returning a DataFrame that contains both.
    """

    def __init__(
        self,
        numeric_features: List[str],
        quantile_transformer,
        standard_scaler_input,
        pca_model,
        pca_output_scaler,
        pca_input_features: List[str],
    ) -> None:
        self.numeric_features = numeric_features or []
        self.quantile_transformer = quantile_transformer
        self.standard_scaler_input = standard_scaler_input
        self.pca_model = pca_model
        self.pca_output_scaler = pca_output_scaler
        self.pca_input_features = pca_input_features or []

        # Derive PCA component names from fitted PCA
        n_components = getattr(self.pca_model, "n_components_", None) or getattr(self.pca_model, "n_components", None)
        if not n_components:
            raise ValueError("Invalid PCA model: n_components is not set")
        self.pca_columns = [f"pca_component_{i+1}" for i in range(int(n_components))]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform incoming features into PCA component space expected by the model.

        Accepts a pandas DataFrame. If a numpy array is passed, it will be coerced
        to a DataFrame using the known PCA input feature names when the shape matches.
        """
        # Coerce numpy arrays to DataFrame when possible
        if not isinstance(df, pd.DataFrame):
            if isinstance(df, (np.ndarray, list)):
                arr = np.asarray(df)
                # If shape matches PCA input features, use them as columns
                if arr.ndim == 2 and len(self.pca_input_features) == arr.shape[1]:
                    df = pd.DataFrame(arr, columns=self.pca_input_features)
                else:
                    raise ValueError("PreprocessingArtifacts.transform expects a pandas DataFrame")

        work = df.copy()

        # One-hot encode categorical/object columns to mirror training
        try:
            cat_cols = work.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
            if cat_cols:
                work = pd.get_dummies(work, columns=cat_cols, drop_first=True, dtype=int)
        except Exception:
            # If dummies fail for any reason, continue; alignment will zero-fill later
            pass

        # Coerce remaining to numeric; non-numeric become NaN then fill 0
        work = work.apply(pd.to_numeric, errors="coerce")
        if work.isnull().values.any():
            work = work.fillna(0)

        # Align to the exact training column set (post-one-hot) before scaling
        # This ensures transformers see the same column order/shape as during fit
        work = work.reindex(columns=self.pca_input_features, fill_value=0.0)

        # Apply training QuantileTransformer and StandardScaler on numeric features
        if self.numeric_features:
            # Use the feature names the transformers were actually fit on when available
            qt_names = (
                getattr(self.quantile_transformer, "feature_names_in_", None)
                if self.quantile_transformer is not None
                else None
            )
            ss_names = (
                getattr(self.standard_scaler_input, "feature_names_in_", None)
                if self.standard_scaler_input is not None
                else None
            )
            numeric_cols = list(qt_names or ss_names or self.numeric_features)

            # Align to full numeric feature set used during training before transforms
            numeric_aligned = work.reindex(columns=numeric_cols, fill_value=0.0)
            try:
                if self.quantile_transformer is not None:
                    qt_out = self.quantile_transformer.transform(numeric_aligned)
                    numeric_aligned = (
                        qt_out
                        if isinstance(qt_out, pd.DataFrame)
                        else pd.DataFrame(qt_out, columns=numeric_cols, index=work.index)
                    )
                if self.standard_scaler_input is not None:
                    ss_out = self.standard_scaler_input.transform(numeric_aligned)
                    numeric_aligned = (
                        ss_out
                        if isinstance(ss_out, pd.DataFrame)
                        else pd.DataFrame(ss_out, columns=numeric_cols, index=work.index)
                    )
                # Replace numeric columns in one shot to avoid fragmentation
                work = work.drop(columns=numeric_cols, errors="ignore")
                numeric_aligned = numeric_aligned.astype(np.float32)
                work = pd.concat([work, numeric_aligned], axis=1)
            except Exception as e:
                # Surface errors so inference does not silently degrade
                raise RuntimeError(f"Numeric transformer failed: {e}")

        # Align to PCA input feature set, fill missing columns with 0 in one shot
        pca_input_df = work.reindex(columns=self.pca_input_features, fill_value=0.0)

        # PCA transform and scale outputs (use DataFrame input to preserve feature names validity)
        pca_out = self.pca_model.transform(pca_input_df)
        if self.pca_output_scaler is not None:
            try:
                pca_out = self.pca_output_scaler.transform(pca_out)
            except Exception:
                pass

        # Build PCA component DataFrame
        pca_df = (
            pca_out
            if isinstance(pca_out, pd.DataFrame)
            else pd.DataFrame(pca_out, columns=self.pca_columns, index=df.index)
        ).astype(np.float32)

        # Concatenate PCA components with original engineered features
        # This mirrors training where PCA components were appended to original data
        out = pd.concat([work, pca_df], axis=1)

        return out
