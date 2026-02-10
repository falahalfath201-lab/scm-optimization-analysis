"""
Test Warehouse Location Optimization Module
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.analysis.warehouse_location import WarehouseOptimizer, print_warehouse_report


def test_warehouse_location():
    """Test warehouse location optimization with actual data"""
    
    print("=" * 70)
    print("TESTING WAREHOUSE LOCATION OPTIMIZATION MODULE")
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
    
    # 3. Initialize Optimizer
    print("\n[3] Initializing Warehouse Optimizer...")
    optimizer = WarehouseOptimizer(
        max_facilities=10,
        facility_cost=50000,  # $50K per facility per month
        facility_capacity=5000  # 5000 orders/month capacity
    )
    print("    [OK] Optimizer initialized")
    
    # 4. K-Means Location (5 facilities)
    print("\n[4] Finding Optimal Locations (K-Means, 5 facilities)...")
    
    # Use sample for faster K-Means
    sample_size_kmeans = min(50000, len(df_processed))
    df_kmeans_sample = df_processed.sample(n=sample_size_kmeans, random_state=42)
    print(f"    Using {sample_size_kmeans:,} samples for analysis")
    
    kmeans_result = optimizer.kmeans_location(
        df_kmeans_sample, 
        n_facilities=5,
        weight_by_demand=True
    )
    
    if kmeans_result.get('status') == 'success':
        print(f"\n    [OK] K-Means optimization complete")
        print(f"    Total Distance: {kmeans_result['total_distance_km']:,.2f} km")
        print(f"    Avg Distance per Order: {kmeans_result['avg_distance_km']:.2f} km")
        print(f"    Total Facility Cost: ${kmeans_result['total_facility_cost']:,.2f}")
        
        print(f"\n    Facility Summary:")
        for fac in kmeans_result['facilities'][:3]:  # Show first 3
            print(f"      Facility {fac['facility_id']}: "
                  f"{fac['customers_served']:,} customers, "
                  f"avg dist {fac['avg_distance_km']:.2f} km")
    
    # 5. Center of Gravity (single facility)
    print("\n[5] Finding Optimal Single Location (Center of Gravity)...")
    cog_result = optimizer.center_of_gravity(df_processed)
    
    if cog_result.get('status') == 'success':
        print(f"\n    [OK] Center of Gravity found")
        loc = cog_result['facility_location']
        print(f"    Location: ({loc['latitude']:.4f}, {loc['longitude']:.4f})")
        print(f"    Avg Distance: {cog_result['avg_distance_km']:.2f} km")
        print(f"    Weighted Avg Distance: {cog_result['weighted_avg_distance_km']:.2f} km")
    
    # 6. P-Median Optimization
    print("\n[6] Running P-Median Optimization (3 facilities)...")
    print("    (This uses Linear Programming and may take a moment...)")
    
    # Use sample for faster computation
    sample_size = min(10000, len(df_processed))
    df_sample = df_processed.sample(n=sample_size, random_state=42)
    print(f"    Using {sample_size:,} samples")
    
    pmedian_result = optimizer.p_median_optimization(df_sample, n_facilities=3)
    
    if pmedian_result.get('status') == 'optimal':
        print(f"\n    [OK] P-Median optimization successful")
        print(f"    Objective Value: {pmedian_result['objective_value']:,.2f}")
        print(f"    Total Distance: {pmedian_result['total_distance_km']:,.2f} km")
        print(f"    Avg Distance: {pmedian_result['avg_distance_km']:.2f} km")
    elif 'error' in pmedian_result:
        print(f"    [WARNING] {pmedian_result['error']}")
    
    # 7. Coverage Analysis
    print("\n[7] Analyzing Coverage (100 km radius)...")
    if kmeans_result.get('facilities'):
        # Use sample for coverage analysis
        df_coverage_sample = df_processed.sample(n=min(30000, len(df_processed)), random_state=55)
        
        coverage = optimizer.coverage_analysis(
            df_coverage_sample,
            kmeans_result['facilities'],
            coverage_radius_km=100
        )
        
        if 'total_coverage_pct' in coverage:
            print(f"\n    [DATA] Coverage Statistics:")
            print(f"    Total Coverage: {coverage['total_coverage_pct']:.1f}%")
            print(f"    Customers Covered: {coverage['customers_covered']:,} / {coverage['total_customers']:,}")
            print(f"    Customers Uncovered: {coverage['customers_uncovered']:,}")
            
            print(f"\n    Per-Facility Coverage:")
            for fac_cov in coverage['facility_coverage'][:5]:
                print(f"      Facility {fac_cov['facility_id']}: "
                      f"{fac_cov['customers_in_range']:,} customers ({fac_cov['coverage_pct']:.1f}%)")
    
    # 8. Scenario Comparison
    print("\n[8] Comparing Different Facility Count Scenarios...")
    print("    Testing: 3, 5, 7, 10 facilities")
    
    # Use sample for faster scenario comparison
    sample_size_scenario = min(20000, len(df_processed))
    df_scenario_sample = df_processed.sample(n=sample_size_scenario, random_state=99)
    print(f"    Using {sample_size_scenario:,} samples for faster comparison")
    
    scenarios = optimizer.compare_scenarios(df_scenario_sample, facility_counts=[3, 5, 7, 10])
    
    if scenarios.get('scenarios'):
        print(f"\n    [CHART] Scenario Results:")
        print(f"    {'Facilities':>11} {'Avg Dist (km)':>14} {'Facility Cost':>14} {'Transport Cost':>15} {'TOTAL':>12}")
        print("    " + "-" * 70)
        
        for name, scenario in scenarios['scenarios'].items():
            print(f"    {scenario['n_facilities']:>11} "
                  f"{scenario['avg_distance_km']:>14.2f} "
                  f"${scenario['total_cost']:>13,.0f} "
                  f"${scenario['transport_cost']:>14,.0f} "
                  f"${scenario['total_system_cost']:>11,.0f}")
        
        print(f"\n    [OK] Best Scenario: {scenarios['best_n_facilities']} facilities")
        print(f"       Total System Cost: ${scenarios['best_total_cost']:,.2f}")
    
    # 9. Regional Optimization
    print("\n[9] Optimizing by Region (2 facilities per region)...")
    
    # Use sample for regional optimization
    sample_size_regional = min(30000, len(df_processed))
    df_regional_sample = df_processed.sample(n=sample_size_regional, random_state=77)
    print(f"    Using {sample_size_regional:,} samples")
    
    regional = optimizer.optimize_by_region(
        df_regional_sample,
        region_column='Order Region',
        facilities_per_region=2
    )
    
    if regional.get('status') == 'success':
        summary = regional['summary']
        print(f"\n    [OK] Regional optimization complete")
        print(f"    Regions Analyzed: {summary['total_regions']}")
        print(f"    Total Facilities: {summary['total_facilities']}")
        print(f"    Total Customers: {summary['total_customers']:,}")
        print(f"    Avg Distance: {summary['avg_distance_km']:.2f} km")
        
        print(f"\n    Top 3 Regions:")
        sorted_regions = sorted(regional['regions'].items(), 
                               key=lambda x: x[1]['customers'], 
                               reverse=True)
        for region_name, region_data in sorted_regions[:3]:
            print(f"      {region_name}: {region_data['customers']:,} customers, "
                  f"avg dist {region_data['avg_distance_km']:.2f} km")
    
    # Print full reports for key methods
    print("\n" + "=" * 70)
    print("DETAILED REPORTS")
    print("=" * 70)
    
    # K-Means Report
    if kmeans_result.get('status') == 'success':
        print_warehouse_report(kmeans_result)
    
    # Center of Gravity Report
    if cog_result.get('status') == 'success':
        print_warehouse_report(cog_result)
    
    # Scenario Comparison Report
    if scenarios.get('scenarios'):
        print_warehouse_report(scenarios)
    
    print("\n" + "=" * 70)
    print("[OK] WAREHOUSE LOCATION OPTIMIZATION TEST COMPLETE")
    print("=" * 70)
    
    return {
        'kmeans': kmeans_result,
        'center_of_gravity': cog_result,
        'scenarios': scenarios,
        'regional': regional
    }


if __name__ == "__main__":
    test_warehouse_location()
