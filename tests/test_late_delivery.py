"""
Test Late Delivery Risk Prediction
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
from src.config import PROCESSED_DATA_FILE
from src.analysis.late_delivery_risk import LateDeliveryRiskModel, print_risk_report

print("=" * 60)
print("LATE DELIVERY RISK PREDICTION")
print("=" * 60)

# Load data
df = pd.read_csv(PROCESSED_DATA_FILE, low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Check target distribution
print(f"\nTarget Distribution (Late_delivery_risk):")
print(df['Late_delivery_risk'].value_counts())
print(f"Late Delivery Rate: {df['Late_delivery_risk'].mean()*100:.2f}%")

# Initialize model
print("\n" + "-" * 60)
model = LateDeliveryRiskModel()

# Train all models
results = model.train(
    df, 
    test_size=0.2,
    models_to_train=['logistic', 'rf', 'gb']
)

# Print report
print_risk_report(results)

# Risk segmentation example
print("\n" + "=" * 60)
print("RISK SEGMENTATION (Sample)")
print("=" * 60)

sample = df.sample(1000, random_state=42)
risk_df = model.get_risk_segments(sample)

print("\nRisk Segment Distribution:")
print(risk_df['risk_segment'].value_counts())

print("\nSample High-Risk Orders:")
high_risk = risk_df[risk_df['risk_segment'] == 'Critical'][
    ['Shipping Mode', 'Order Region', 'risk_probability']
].head(5)
print(high_risk.to_string())

print("\n" + "=" * 60)
print("Late Delivery Risk Test Complete!")
print("=" * 60)
