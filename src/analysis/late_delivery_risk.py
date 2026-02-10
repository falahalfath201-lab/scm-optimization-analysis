"""
Late Delivery Risk Assessment Module
ML-based classification to predict delivery delays

Models:
1. Logistic Regression (baseline)
2. Random Forest
3. Gradient Boosting
4. XGBoost (if available)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import time
import logging
import warnings

warnings.filterwarnings('ignore')

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from ..config import RANDOM_SEED, TEST_SIZE

logger = logging.getLogger(__name__)


class LateDeliveryRiskModel:
    """
    Late Delivery Risk Prediction using ML Classification
    
    Target: Late_delivery_risk (0 = On-time, 1 = Late)
    
    Features used:
    - Shipping Mode
    - Order characteristics (quantity, price, etc.)
    - Temporal features (day of week, month, etc.)
    - Geographical features
    """
    
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns: List[str] = []
        self.target_column = 'Late_delivery_risk'
        self.is_trained = False
        self.results: Dict[str, Any] = {}
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for training/prediction
        
        Returns:
            X (features), y (target)
        """
        data = df.copy()
        
        # Define feature groups
        numeric_features = [
            'Days for shipment (scheduled)',
            'Order Item Quantity',
            'Product Price',
            'Order Item Total',
            'Sales per customer',
            'order_dayofweek',
            'order_month',
            'order_hour',
            'is_weekend'
        ]
        
        categorical_features = [
            'Shipping Mode',
            'Customer Segment',
            'Market',
            'Order Region'
        ]
        
        # Filter existing columns
        numeric_features = [f for f in numeric_features if f in data.columns]
        categorical_features = [f for f in categorical_features if f in data.columns]
        
        # Encode categorical features
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                data[f'{col}_encoded'] = self.label_encoders[col].fit_transform(data[col].astype(str))
            else:
                # Handle unseen labels
                le = self.label_encoders[col]
                data[f'{col}_encoded'] = data[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        encoded_features = [f'{col}_encoded' for col in categorical_features]
        
        # Combine all features
        self.feature_columns = numeric_features + encoded_features
        
        # Handle missing values
        X = data[self.feature_columns].fillna(0)
        y = data[self.target_column] if self.target_column in data.columns else None
        
        return X, y
    
    def train(self, df: pd.DataFrame, 
              test_size: float = TEST_SIZE,
              models_to_train: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train classification models
        
        Args:
            df: Training data
            test_size: Test split ratio
            models_to_train: List of models ('logistic', 'rf', 'gb', 'xgb')
            
        Returns:
            Training results with metrics
        """
        print("Preparing features...")
        X, y = self.prepare_features(df)
        
        if y is None:
            raise ValueError(f"Target column '{self.target_column}' not found")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        print(f"Features: {len(self.feature_columns)}")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Define models
        available_models = {
            'logistic': ('Logistic Regression', LogisticRegression(
                random_state=self.random_state, max_iter=1000
            )),
            'rf': ('Random Forest', RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )),
            'gb': ('Gradient Boosting', GradientBoostingClassifier(
                n_estimators=100, random_state=self.random_state
            ))
        }
        
        if XGBOOST_AVAILABLE:
            available_models['xgb'] = ('XGBoost', XGBClassifier(
                n_estimators=100, random_state=self.random_state,
                use_label_encoder=False, eval_metric='logloss'
            ))
        
        models_to_train = models_to_train or list(available_models.keys())
        
        results = {
            'models': {},
            'best_model': None,
            'best_score': 0,
            'feature_columns': self.feature_columns
        }
        
        print("\n" + "-" * 50)
        print("Training models...")
        print("-" * 50)
        
        for model_key in models_to_train:
            if model_key not in available_models:
                continue
            
            name, model = available_models[model_key]
            start_time = time.time()
            
            print(f"\n[DATA] {name}...")
            
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'training_time_sec': time.time() - start_time
            }
            
            # Store model and results
            self.models[model_key] = model
            results['models'][model_key] = {
                'name': name,
                'metrics': metrics,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            print(f"   Accuracy:  {metrics['accuracy']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall:    {metrics['recall']:.4f}")
            print(f"   F1 Score:  {metrics['f1']:.4f}")
            print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"   Time:      {metrics['training_time_sec']:.2f}s")
            
            # Track best model by F1 score
            if metrics['f1'] > results['best_score']:
                results['best_score'] = metrics['f1']
                results['best_model'] = model_key
        
        # Feature importance (from best tree-based model)
        if 'rf' in self.models:
            importance = self.models['rf'].feature_importances_
            results['feature_importance'] = dict(zip(self.feature_columns, importance))
        
        self.is_trained = True
        self.results = results
        
        return results
    
    def predict(self, df: pd.DataFrame, 
                model_key: Optional[str] = None) -> np.ndarray:
        """
        Predict late delivery risk
        
        Args:
            df: Data to predict
            model_key: Which model to use (default: best model)
            
        Returns:
            Predictions (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        model_key = model_key or self.results.get('best_model', 'rf')
        model = self.models.get(model_key)
        
        if model is None:
            raise ValueError(f"Model '{model_key}' not found")
        
        X, _ = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        return model.predict(X_scaled)
    
    def predict_proba(self, df: pd.DataFrame,
                      model_key: Optional[str] = None) -> np.ndarray:
        """
        Predict probability of late delivery
        
        Returns:
            Probability scores [0, 1]
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        model_key = model_key or self.results.get('best_model', 'rf')
        model = self.models.get(model_key)
        
        X, _ = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        return model.predict_proba(X_scaled)[:, 1]
    
    def get_risk_segments(self, df: pd.DataFrame,
                          model_key: Optional[str] = None) -> pd.DataFrame:
        """
        Segment orders by risk level
        
        Returns:
            DataFrame with risk scores and segments
        """
        proba = self.predict_proba(df, model_key)
        
        result = df.copy()
        result['risk_probability'] = proba
        result['risk_segment'] = pd.cut(
            proba,
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return result
    
    def analyze_risk_factors(self) -> Dict[str, Any]:
        """
        Analyze key risk factors based on feature importance
        """
        if 'feature_importance' not in self.results:
            return {"error": "No feature importance available"}
        
        importance = self.results['feature_importance']
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'top_risk_factors': sorted_features[:10],
            'all_factors': sorted_features
        }


def print_risk_report(results: Dict[str, Any]) -> None:
    """Pretty print risk assessment results"""
    print("\n" + "=" * 60)
    print("LATE DELIVERY RISK - MODEL COMPARISON")
    print("=" * 60)
    
    # Model comparison table
    print(f"\n{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10}")
    print("-" * 75)
    
    for key, model_data in results.get('models', {}).items():
        m = model_data['metrics']
        name = model_data['name']
        best_marker = " [BEST]" if key == results.get('best_model') else ""
        print(f"{name + best_marker:<25} {m['accuracy']:<10.4f} {m['precision']:<10.4f} "
              f"{m['recall']:<10.4f} {m['f1']:<10.4f} {m['roc_auc']:<10.4f}")
    
    # Best model
    best = results.get('best_model')
    if best:
        print(f"\n[BEST] Best Model: {results['models'][best]['name']} (F1: {results['best_score']:.4f})")
    
    # Feature importance
    if 'feature_importance' in results:
        print(f"\n[DATA] Top Risk Factors:")
        sorted_imp = sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True)
        for i, (feat, imp) in enumerate(sorted_imp[:5], 1):
            bar = "#" * int(imp * 50)
            print(f"   {i}. {feat:<30} {imp:.4f} {bar}")
    
    print("\n" + "=" * 60)
