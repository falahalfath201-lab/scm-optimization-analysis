"""
Test VRP Optimization with Time Minimization
Compares: Cluster-TSP, Balanced, Min-Total strategies
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
from src.config import PROCESSED_DATA_FILE
from src.analysis.route_optimization import RouteOptimizer

print("=" * 70)
print("VRP OPTIMIZATION - TIME MINIMIZATION")
print("=" * 70)

# Load data
df = pd.read_csv(PROCESSED_DATA_FILE, low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows")

# Configuration
NUM_VEHICLES = 3
SPEED_KMH = 40
REGION = 'CA'

print(f"\nConfig: {NUM_VEHICLES} vehicles, {SPEED_KMH} km/h")
print(f"Region: {REGION}")

# Initialize unified optimizer
optimizer = RouteOptimizer(
    num_vehicles=NUM_VEHICLES,
    speed_kmh=SPEED_KMH,
    max_time_per_vehicle_min=480
)

# Show available regions
print("\nTop Regions:")
print(optimizer.get_available_regions(df, top_n=5))

# Run VRP optimization with benchmark
print("\n" + "-" * 70)
print("Running VRP Benchmark (all strategies)...")
results = optimizer.optimize_region(
    df=df,
    region_column='Customer State',
    region_value=REGION,
    max_locations=30,
    problem_type='vrp'
)

optimizer.print_vrp_report(results)

# Compare TSP vs VRP
print("\n" + "=" * 70)
print("TSP vs VRP COMPARISON")
print("=" * 70)

# Get locations for comparison
region_df = df[df['Customer State'] == REGION]
unique_locs = region_df[['Latitude', 'Longitude']].drop_duplicates().head(30)
centroid = (unique_locs['Latitude'].mean(), unique_locs['Longitude'].mean())
locations = [centroid] + list(zip(unique_locs['Latitude'], unique_locs['Longitude']))

comparison = optimizer.compare_tsp_vs_vrp(locations, num_vehicles=3)
optimizer.print_comparison(comparison)

print("\n" + "=" * 70)
print("VRP Test Complete!")
print("=" * 70)
