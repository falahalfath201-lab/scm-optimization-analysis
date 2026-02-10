"""
Test Cost Optimization Module
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.analysis.cost_optimization import CostOptimizer, print_cost_report


def test_cost_optimization():
    """Test cost optimization with actual data"""
    
    print("=" * 70)
    print("TESTING COST OPTIMIZATION MODULE")
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
    
    # 3. Initialize Cost Optimizer
    print("\n[3] Initializing Cost Optimizer...")
    optimizer = CostOptimizer(
        fuel_cost_per_km=0.15,
        labor_cost_per_hour=25.0,
        vehicle_fixed_cost=150.0
    )
    print("    [OK] Optimizer initialized")
    
    # 4. Analyze Current Costs
    print("\n[4] Analyzing Current Costs...")
    current_costs = optimizer.analyze_current_costs(df_processed)
    
    print(f"\n    [DATA] Cost by Shipping Mode:")
    for mode, stats in current_costs['by_shipping_mode'].items():
        print(f"       {mode}:")
        print(f"         Orders: {stats['order_count']:,}")
        print(f"         Total Revenue: ${stats['total_revenue']:,.2f}")
        print(f"         Est. Shipping Cost: ${stats['est_shipping_cost']:,.2f}")
    
    if 'late_delivery' in current_costs:
        ld = current_costs['late_delivery']
        print(f"\n    [WARNING] Late Delivery Impact:")
        print(f"       Late Orders: {ld['late_orders']:,} ({ld['late_rate']:.1f}%)")
        print(f"       Total Delay Days: {ld['total_delay_days']:,.0f}")
        print(f"       Estimated Penalty: ${ld['estimated_penalty']:,.2f}")
    
    # 5. Optimize Shipping Modes
    print("\n[5] Optimizing Shipping Modes...")
    optimized = optimizer.optimize_shipping_mode(df_processed)
    
    print(f"\n    [TARGET] Optimization Results:")
    print(f"       Current Cost:   ${optimized['current_cost']:,.2f}")
    print(f"       Optimized Cost: ${optimized['optimized_cost']:,.2f}")
    print(f"       [MONEY] Savings:     ${optimized['savings']:,.2f} ({optimized['savings_pct']:.1f}%)")
    print(f"       Expected Service Level: {optimized['expected_service_level']*100:.1f}%")
    
    print(f"\n    [PACKAGE] Recommended Distribution:")
    for mode, count in optimized['mode_distribution'].items():
        pct = count / len(df_processed) * 100
        print(f"       {mode}: {count:,} orders ({pct:.1f}%)")
    
    # 6. Cost-Benefit Analysis
    print("\n[6] Running Cost-Benefit Analysis...")
    cba = optimizer.cost_benefit_analysis(df_processed)
    
    print(f"\n    [CHART] Strategy Comparison:")
    print(f"       {'Strategy':<25} {'Shipping Cost':>12} {'Late Penalty':>12} {'TOTAL':>12}")
    print("       " + "-" * 65)
    
    for name, strategy in cba['strategies'].items():
        marker = "[BEST]" if name == cba['best_strategy'] else "  "
        print(f"       {marker} {strategy['name']:<23} "
              f"${strategy['cost']:>10,.0f} "
              f"${strategy['late_penalty']:>10,.0f} "
              f"${strategy['total_cost']:>10,.0f}")
    
    print(f"\n    [OK] Best Strategy: {cba['best_strategy_name']}")
    print(f"       Total Cost: ${cba['best_total_cost']:,.2f}")
    
    # 7. Get Full Summary
    print("\n[7] Generating Full Optimization Report...")
    summary = optimizer.get_optimization_summary(df_processed)
    
    print(f"\n    [IDEA] Recommendations:")
    for i, rec in enumerate(summary['recommendations'], 1):
        print(f"       {i}. {rec}")
    
    # 8. Test with Sample Orders (PuLP if available)
    print("\n[8] Testing Advanced LP (PuLP)...")
    sample_orders = [
        {'demand': 10, 'region': 'West'},
        {'demand': 5, 'region': 'East'},
        {'demand': 15, 'region': 'Central'},
        {'demand': 8, 'region': 'South'},
        {'demand': 12, 'region': 'North'},
    ]
    
    pulp_result = optimizer.optimize_with_pulp(sample_orders * 20)  # 100 orders
    
    if 'error' in pulp_result:
        print(f"    [WARNING] {pulp_result['error']}")
    else:
        print(f"    [OK] PuLP optimization successful")
        print(f"       Total Cost: ${pulp_result['total_cost']:.2f}")
        print(f"       Mode Distribution: {pulp_result['mode_distribution']}")
    
    # Print Full Report
    print_cost_report(summary)
    
    print("\n" + "=" * 70)
    print("[OK] COST OPTIMIZATION TEST COMPLETE")
    print("=" * 70)
    
    return summary


if __name__ == "__main__":
    test_cost_optimization()
