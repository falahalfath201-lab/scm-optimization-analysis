"""
Performance Metrics Utilities
Provides standardized metrics calculations for all analysis modules
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Calculate classification accuracy"""
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    predictions = (y_pred >= threshold).astype(int)
    return np.mean(predictions == y_true)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared score"""
    return r2_score(y_true, y_pred)


def calculate_cost_savings(baseline_cost: float, optimized_cost: float) -> Dict[str, float]:
    """
    Calculate cost savings metrics
    
    Returns:
        Dict with 'absolute_savings', 'percentage_savings', 'baseline', 'optimized'
    """
    absolute_savings = baseline_cost - optimized_cost
    percentage_savings = (absolute_savings / baseline_cost * 100) if baseline_cost > 0 else 0
    
    return {
        'baseline_cost': baseline_cost,
        'optimized_cost': optimized_cost,
        'absolute_savings': absolute_savings,
        'percentage_savings': percentage_savings
    }


def calculate_improvement_pct(baseline: float, improved: float, 
                              higher_is_better: bool = True) -> float:
    """
    Calculate percentage improvement
    
    Args:
        baseline: Baseline metric value
        improved: Improved metric value
        higher_is_better: If True, higher values are better (e.g., accuracy)
                         If False, lower values are better (e.g., cost)
    
    Returns:
        Percentage improvement (positive means improvement)
    """
    if baseline == 0:
        return 0.0
    
    if higher_is_better:
        improvement = ((improved - baseline) / abs(baseline)) * 100
    else:
        improvement = ((baseline - improved) / abs(baseline)) * 100
    
    return improvement


def calculate_distance_metrics(distance_matrix: np.ndarray) -> Dict[str, float]:
    """Calculate statistics from distance matrix"""
    return {
        'total_distance': np.sum(distance_matrix),
        'avg_distance': np.mean(distance_matrix[distance_matrix > 0]),
        'max_distance': np.max(distance_matrix),
        'min_distance': np.min(distance_matrix[distance_matrix > 0])
    }


def calculate_delivery_metrics(df: pd.DataFrame, 
                               actual_col: str = 'Days for shipping (real)',
                               predicted_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate delivery performance metrics
    
    Args:
        df: DataFrame with delivery data
        actual_col: Column name for actual delivery days
        predicted_col: Optional column for predicted days
    
    Returns:
        Dictionary with delivery metrics
    """
    metrics = {
        'mean_delivery_days': df[actual_col].mean(),
        'median_delivery_days': df[actual_col].median(),
        'std_delivery_days': df[actual_col].std(),
        'min_delivery_days': df[actual_col].min(),
        'max_delivery_days': df[actual_col].max()
    }
    
    # Late delivery analysis
    if 'Late_delivery_risk' in df.columns:
        metrics['late_delivery_count'] = df['Late_delivery_risk'].sum()
        metrics['late_delivery_rate'] = (df['Late_delivery_risk'].sum() / len(df)) * 100
        metrics['on_time_rate'] = 100 - metrics['late_delivery_rate']
    
    # Prediction accuracy if available
    if predicted_col and predicted_col in df.columns:
        metrics['prediction_mae'] = calculate_mae(df[actual_col], df[predicted_col])
        metrics['prediction_rmse'] = calculate_rmse(df[actual_col], df[predicted_col])
        metrics['prediction_r2'] = calculate_r2(df[actual_col], df[predicted_col])
    
    return metrics


def calculate_cost_metrics(df: pd.DataFrame,
                           sales_col: str = 'Sales',
                           profit_col: str = 'Order Profit Per Order') -> Dict[str, Any]:
    """Calculate cost and profitability metrics"""
    metrics = {
        'total_revenue': df[sales_col].sum(),
        'total_profit': df[profit_col].sum(),
        'avg_order_value': df[sales_col].mean(),
        'avg_profit_per_order': df[profit_col].mean(),
        'profit_margin': (df[profit_col].sum() / df[sales_col].sum() * 100) if df[sales_col].sum() > 0 else 0,
        'total_orders': len(df)
    }
    
    # Shipping cost estimation (if available)
    if 'estimated_shipping_cost' in df.columns:
        metrics['total_shipping_cost'] = df['estimated_shipping_cost'].sum()
        metrics['avg_shipping_cost'] = df['estimated_shipping_cost'].mean()
        metrics['shipping_cost_ratio'] = (df['estimated_shipping_cost'].sum() / df[sales_col].sum() * 100)
    
    return metrics


def calculate_route_metrics(routes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate metrics for route optimization results"""
    total_distance = sum(r.get('distance', 0) for r in routes)
    total_time = sum(r.get('time', 0) for r in routes)
    total_stops = sum(len(r.get('stops', [])) for r in routes)
    
    return {
        'num_routes': len(routes),
        'total_distance': total_distance,
        'avg_distance_per_route': total_distance / len(routes) if routes else 0,
        'total_time': total_time,
        'avg_time_per_route': total_time / len(routes) if routes else 0,
        'total_stops': total_stops,
        'avg_stops_per_route': total_stops / len(routes) if routes else 0,
        'max_route_distance': max([r.get('distance', 0) for r in routes]) if routes else 0,
        'min_route_distance': min([r.get('distance', 0) for r in routes]) if routes else 0
    }


def format_metric(value: float, metric_type: str = 'number', precision: int = 2) -> str:
    """
    Format metric for display
    
    Args:
        value: Numeric value to format
        metric_type: 'number', 'currency', 'percentage', 'time'
        precision: Decimal places
    
    Returns:
        Formatted string
    """
    if metric_type == 'currency':
        return f"${value:,.{precision}f}"
    elif metric_type == 'percentage':
        return f"{value:.{precision}f}%"
    elif metric_type == 'time':
        hours = int(value // 60)
        minutes = int(value % 60)
        return f"{hours}h {minutes}m"
    else:
        return f"{value:,.{precision}f}"


def compare_scenarios(scenarios: Dict[str, Dict[str, float]], 
                     metric_name: str = 'total_cost',
                     lower_is_better: bool = True) -> Dict[str, Any]:
    """
    Compare multiple scenarios and identify best
    
    Args:
        scenarios: Dict of scenario_name -> metrics dict
        metric_name: Key to compare on
        lower_is_better: If True, lower value is better
    
    Returns:
        Comparison results with rankings
    """
    # Extract metric values
    values = {name: data[metric_name] for name, data in scenarios.items()}
    
    # Find best
    if lower_is_better:
        best_name = min(values, key=values.get)
        best_value = min(values.values())
    else:
        best_name = max(values, key=values.get)
        best_value = max(values.values())
    
    # Calculate vs best
    comparisons = {}
    for name, value in values.items():
        diff = value - best_value
        diff_pct = (diff / best_value * 100) if best_value != 0 else 0
        comparisons[name] = {
            'value': value,
            'vs_best': diff,
            'vs_best_pct': diff_pct,
            'is_best': name == best_name
        }
    
    # Sort by value
    sorted_names = sorted(values.keys(), 
                         key=lambda x: values[x], 
                         reverse=not lower_is_better)
    
    return {
        'best_scenario': best_name,
        'best_value': best_value,
        'comparisons': comparisons,
        'ranking': sorted_names
    }


def calculate_balance_score(values: List[float]) -> float:
    """
    Calculate balance score (1.0 = perfect balance)
    Uses coefficient of variation
    """
    if not values or len(values) < 2:
        return 1.0
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    if mean_val == 0:
        return 1.0
    
    cv = std_val / mean_val
    balance_score = 1 / (1 + cv)
    
    return balance_score
