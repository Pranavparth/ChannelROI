import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

class BudgetSimulator:
    def __init__(self):
        self.xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        self.lr_model = LogisticRegression(max_iter=1000, random_state=42)
        self.scaler = StandardScaler()
        self.channels = ['Google', 'Meta', 'Email', 'SEO']
        self.baseline_df = None
        self.feature_cols = None
        
    def _extract_features(self, paths_df):
        """
        Convert user paths into purely numerical features for ML.
        Features will be standard counts of touchpoints per channel.
        For simplicity, we use counts instead of spend because SEO has 0 spend,
        and spend correlates heavily with counts anyway for paid channels.
        """
        features = []
        
        for _, row in paths_df.iterrows():
            path = row['path']
            f = {f"count_{c}": 0 for c in self.channels}
            for c in path:
                f[f"count_{c}"] += 1
            f['conversion'] = row['conversion']
            features.append(f)
            
        return pd.DataFrame(features)

    def train(self, paths_df):
        # 1. Feature extraction
        df_feat = self._extract_features(paths_df)
        self.feature_cols = [f"count_{c}" for c in self.channels]
        
        X = df_feat[self.feature_cols]
        y = df_feat['conversion']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 2. Train XGBoost (for feature importance / non-linear lift)
        self.xgb_model.fit(X_train, y_train)
        xgb_auc = roc_auc_score(y_test, self.xgb_model.predict_proba(X_test)[:, 1])
        
        # 3. Train Logistic Regression (for smooth budget simulation extrapolation)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.lr_model.fit(X_train_scaled, y_train)
        lr_auc = roc_auc_score(y_test, self.lr_model.predict_proba(X_test_scaled)[:, 1])
        
        # Save baseline to simulate from
        self.baseline_df = df_feat.copy()
        
        # Calculate baseline metrics
        baseline_preds = self.lr_model.predict_proba(self.scaler.transform(self.baseline_df[self.feature_cols]))[:, 1]
        self.baseline_conversions = np.sum(baseline_preds)
        
        # Extract total baseline counts to represent "budget" proxy
        self.baseline_counts = {c: self.baseline_df[f"count_{c}"].sum() for c in self.channels}
        
        return {
            "xgb_auc": xgb_auc,
            "lr_auc": lr_auc,
            "feature_importance": dict(zip(self.channels, self.xgb_model.feature_importances_)),
            "lr_coefficients": dict(zip(self.channels, self.lr_model.coef_[0]))
        }
        
    def simulate_budget(self, budget_multipliers):
        """
        budget_multipliers: dict { 'Google': 1.2, 'Meta': 0.8 ... }
        This translates to "increase Google interaction by 20%, decrease Meta by 20%".
        We scale the counts, then predict with Logistic Regression (since it handles scaled continuous values smoothly).
        """
        if self.baseline_df is None:
            raise ValueError("Model not trained yet.")
            
        modified_X = self.baseline_df[self.feature_cols].copy().astype(float)
        
        for c in self.channels:
            if c in budget_multipliers:
                # Scale the feature proxying the budget
                modified_X[f"count_{c}"] *= budget_multipliers[c]
                
        # Predict with LR
        scaled_X = self.scaler.transform(modified_X)
        new_preds = self.lr_model.predict_proba(scaled_X)[:, 1]
        new_conversions = np.sum(new_preds)
        
        return {
            "projected_conversions": new_conversions,
            "delta_conversions": new_conversions - self.baseline_conversions,
            "percent_change": ((new_conversions - self.baseline_conversions) / self.baseline_conversions) * 100
        }
