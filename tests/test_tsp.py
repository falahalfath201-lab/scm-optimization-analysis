"""
Test TSP Optimization with Benchmark
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
from src.config import PROCESSED_DATA_FILE
from src.analysis.route_optimization import RouteOptimizer

print("=" * 60)
print("TSP OPTIMIZATION BENCHMARK")
print("=" * 60)

# Load data
df = pd.read_csv(PROCESSED_DATA_FILE, low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows")

# Initialize unified optimizer
optimizer = RouteOptimizer()

# Show regions
print("\nTop 5 Regions:")
print(optimizer.get_available_regions(df, top_n=5))

# Test California with TSP
print("\n" + "-" * 60)
results = optimizer.optimize_region(
    df=df,
    region_column='Customer State',
    region_value='CA',
    max_locations=30,
    problem_type='tsp'
)
optimizer.print_tsp_report(results)
