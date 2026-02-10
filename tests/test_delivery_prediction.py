"""
Test Delivery Time Prediction Module
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.analysis.delivery_prediction import DeliveryPredictor


def test_delivery_prediction():
    """Test delivery prediction with actual data"""
    
    print("=" * 70)
    print("TESTING DELIVERY PREDICTION MODULE")
    print("=" * 70)
    
    # 1. Load Data
    print("\n[1] Loading data...")
    loader = DataLoader()
    df = loader.load_data()
    print(f"    Loaded {len(df):,} rows")
    
    # 2. Preprocess
    print("\n[2] Preprocessing...")
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess_data(df)
    print(f"    Processed {len(df_processed):,} rows")
    
    # Check required columns
    print("\n[3] Checking required columns...")
    required_cols = ['Days for shipping (real)', 'Order Item Quantity', 'Product Price', 
                     'order_month', 'order_hour', 'shipping_month']
    
    missing = [col for col in required_cols if col not in df_processed.columns]
    if missing:
        print(f"    [WARNING] Missing columns: {missing}")
    else:
        print(f"    [OK] All required columns present")
    
    # 3. Initialize Predictor
    print("\n[4] Initializing Delivery Predictor...")
    predictor = DeliveryPredictor()
    print("    [OK] Predictor initialized")
    
    # 4. Analyze Delivery Patterns
    print("\n[5] Analyzing Delivery Patterns...")
    analysis = predictor.analyze_delivery_patterns(df_processed)
    
    if 'delivery_stats' in analysis:
        stats = analysis['delivery_stats']
        print(f"\n    [DATA] Delivery Time Statistics:")
        print(f"       Mean:   {stats['mean_delivery_time']:.2f} days")
        print(f"       Median: {stats['median_delivery_time']:.2f} days")
        print(f"       Std:    {stats['std_delivery_time']:.2f} days")
        print(f"       Min:    {stats['min_delivery_time']:.2f} days")
        print(f"       Max:    {stats['max_delivery_time']:.2f} days")
    
    if 'late_delivery_rate' in analysis:
        print(f"\n    [WARNING] Late Delivery Rate: {analysis['late_delivery_rate']:.2f}%")
    
    # Monthly patterns
    if 'monthly_delivery_stats' in analysis:
        print(f"\n    [CALENDAR] Monthly Delivery Patterns:")
        monthly = analysis['monthly_delivery_stats']
        if 'mean' in monthly:
            for month, avg_time in sorted(monthly['mean'].items())[:6]:
                count = monthly['count'].get(month, 0)
                print(f"       Month {month}: {avg_time:.2f} days (n={count:,})")
    
    # 5. Train Prediction Models
    print("\n[6] Training Prediction Models...")
    print("    (This may take a minute...)")
    
    # Use sample for faster testing
    sample_size = min(50000, len(df_processed))
    df_sample = df_processed.sample(n=sample_size, random_state=42)
    print(f"    Using {sample_size:,} samples for training")
    
    results = predictor.train_prediction_models(df_sample)
    
    if 'error' in results:
        print(f"    [ERROR] Error: {results['error']}")
    else:
        print(f"\n    [ML] Model Performance:")
        print(f"       {'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'CV-MAE':>8}")
        print("       " + "-" * 60)
        
        for model_name, model_data in results.items():
            if 'metrics' in model_data:
                m = model_data['metrics']
                print(f"       {model_name:<20} {m['mae']:>8.2f} {m['rmse']:>8.2f} "
                      f"{m['r2_score']:>8.3f} {m['cv_mae']:>8.2f}")
        
        # Find best model
        best_model = min(results.items(), 
                        key=lambda x: x[1]['metrics']['mae'] if 'metrics' in x[1] else float('inf'))
        print(f"\n    [OK] Best Model: {best_model[0]} (MAE: {best_model[1]['metrics']['mae']:.2f})")
        
        # Feature importance for best model
        if 'feature_importance' in best_model[1] and best_model[1]['feature_importance']:
            print(f"\n    [TARGET] Top 5 Feature Importance ({best_model[0]}):")
            for i, (feature, importance) in enumerate(list(best_model[1]['feature_importance'].items())[:5], 1):
                print(f"       {i}. {feature}: {importance:.4f}")
    
    # 6. Time Series Forecast (if Prophet available)
    print("\n[7] Testing Time Series Forecasting...")
    forecast_result = predictor.forecast_delivery_times(df_sample, periods=30)
    
    if 'error' in forecast_result:
        print(f"    [WARNING] {forecast_result['error']}")
    else:
        print(f"    [OK] Forecast generated for {forecast_result['forecast_periods']} periods")
        if forecast_result['forecast']:
            print(f"\n    [CHART] Sample Forecast (first 5 days):")
            print(f"       {'Date':<12} {'Predicted':>12} {'Lower':>12} {'Upper':>12}")
            print("       " + "-" * 52)
            for i, pred in enumerate(forecast_result['forecast'][:5], 1):
                print(f"       {str(pred['ds'])[:10]:<12} "
                      f"{pred['yhat']:>12.2f} "
                      f"{pred['yhat_lower']:>12.2f} "
                      f"{pred['yhat_upper']:>12.2f}")
    
    # 7. Make Predictions on Test Data
    if predictor.models:
        print("\n[8] Making Predictions on New Data...")
        test_sample = df_processed.sample(n=min(1000, len(df_processed)), random_state=123)
        
        try:
            predictions = predictor.predict_delivery_time(test_sample, model_name='random_forest')
            actual = test_sample['Days for shipping (real)'].values
            
            # Calculate error
            from sklearn.metrics import mean_absolute_error
            mae = mean_absolute_error(actual, predictions)
            
            print(f"    [OK] Predictions made on {len(predictions)} samples")
            print(f"    MAE on test sample: {mae:.2f} days")
            
            print(f"\n    Sample Predictions (first 5):")
            print(f"       {'Actual':>8} {'Predicted':>10} {'Error':>8}")
            print("       " + "-" * 30)
            for i in range(min(5, len(predictions))):
                error = abs(actual[i] - predictions[i])
                print(f"       {actual[i]:>8.2f} {predictions[i]:>10.2f} {error:>8.2f}")
        
        except Exception as e:
            print(f"    [ERROR] Prediction error: {e}")
    
    # 8. Model Comparison
    print("\n[9] Model Comparison Summary:")
    if results and not 'error' in results:
        comparison = predictor.get_model_comparison(results)
        print(f"\n{comparison.to_string(index=False)}")
    
    print("\n" + "=" * 70)
    print("[OK] DELIVERY PREDICTION TEST COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    test_delivery_prediction()
