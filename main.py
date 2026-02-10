"""
SCM Optimization - Main Entry Point

3 Execution Modes:
   Mode 1: Single Analysis (Fast, for development)
   Mode 2: Selected Analyses (Medium, targeted insights)
   Mode 3: All Analyses (Comprehensive, final report)
"""

import sys
from typing import List
from datetime import datetime

from src.pipelines.master_pipeline import MasterPipeline
from src.pipelines.cost_optimization_pipeline import CostOptimizationPipeline
from src.pipelines.route_optimization_pipeline import RouteOptimizationPipeline
from src.pipelines.delivery_prediction_pipeline import DeliveryPredictionPipeline
from src.pipelines.risk_analysis_pipeline import RiskAnalysisPipeline
from src.pipelines.warehouse_location_pipeline import WarehouseLocationPipeline


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*70)
    print(" " * 15 + "SUPPLY CHAIN OPTIMIZATION SYSTEM")
    print("="*70)
    print(f"{'Date:':<20} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


def print_menu():
    """Print mode selection menu"""
    print("\n" + "-"*70)
    print("SELECT EXECUTION MODE:")
    print("-"*70)
    print("  [1] Single Analysis    - Run one specific analysis (Fast)")
    print("  [2] Selected Analyses  - Run multiple selected analyses (Medium)")
    print("  [3] All Analyses       - Run complete optimization suite (Comprehensive)")
    print("  [0] Exit")
    print("-"*70)


def print_analysis_menu():
    """Print available analyses"""
    print("\n" + "-"*70)
    print("AVAILABLE ANALYSES:")
    print("-"*70)
    print("  [1] Cost Optimization      - Optimize shipping costs")
    print("  [2] Route Optimization     - Optimize delivery routes")
    print("  [3] Delivery Prediction    - Predict delivery times with ML")
    print("  [4] Risk Analysis          - Analyze late delivery risks")
    print("  [5] Warehouse Location     - Optimize warehouse locations")
    print("-"*70)


def get_user_choice(prompt: str, valid_choices: List[str]) -> str:
    """Get and validate user input"""
    while True:
        choice = input(prompt).strip()
        if choice in valid_choices:
            return choice
        print(f"Invalid choice. Please select from: {', '.join(valid_choices)}")


def run_single_analysis():
    """Mode 1: Run single analysis independently"""
    print("\n" + "="*70)
    print("MODE 1: SINGLE ANALYSIS")
    print("="*70)
    
    print_analysis_menu()
    
    choice = get_user_choice("\nSelect analysis [1-5]: ", ['1', '2', '3', '4', '5'])
    
    # Map choice to pipeline
    pipelines = {
        '1': ('Cost Optimization', CostOptimizationPipeline),
        '2': ('Route Optimization', RouteOptimizationPipeline),
        '3': ('Delivery Prediction', DeliveryPredictionPipeline),
        '4': ('Risk Analysis', RiskAnalysisPipeline),
        '5': ('Warehouse Location', WarehouseLocationPipeline)
    }
    
    name, PipelineClass = pipelines[choice]
    
    print(f"\nStarting {name} Analysis...")
    print("This will load data independently (standalone mode)\n")
    
    # Run pipeline in standalone mode
    pipeline = PipelineClass()
    result = pipeline.run(shared_data=None)  # No shared data
    
    if result['status'] == 'success':
        print(f"\n[OK] {name} completed successfully!")
    else:
        print(f"\n[ERROR] {name} failed: {result.get('error', 'Unknown error')}")
    
    return result


def run_selected_analyses():
    """Mode 2: Run selected analyses with shared data"""
    print("\n" + "="*70)
    print("MODE 2: SELECTED ANALYSES")
    print("="*70)
    
    print_analysis_menu()
    
    print("\nSelect analyses to run (e.g., '1,3,4' or '1 3 4'):")
    selection = input("Enter numbers separated by commas or spaces: ").strip()
    
    # Parse selection
    selected = []
    for char in selection:
        if char.isdigit():
            selected.append(char)
    
    if not selected:
        print("[ERROR] No valid selections made")
        return None
    
    # Map to pipeline names
    pipeline_map = {
        '1': 'cost_optimization',
        '2': 'route_optimization',
        '3': 'delivery_prediction',
        '4': 'risk_analysis',
        '5': 'warehouse_location'
    }
    
    pipeline_names = [pipeline_map[s] for s in selected if s in pipeline_map]
    
    if not pipeline_names:
        print("[ERROR] No valid pipelines selected")
        return None
    
    print(f"\nSelected {len(pipeline_names)} analyses:")
    for name in pipeline_names:
        print(f"   - {name.replace('_', ' ').title()}")
    
    print("\n[SHARED DATA MODE] Loading data once for all analyses...")
    
    # Run with master pipeline
    master = MasterPipeline()
    results = master.run_selected(pipeline_names)
    
    # Generate master report
    master.save_master_report()
    
    # Summary
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)
    
    success_count = sum(1 for r in results.values() if r.get('status') == 'success')
    total_count = len(results)
    
    for name, result in results.items():
        status_icon = "[OK]" if result.get('status') == 'success' else "[FAIL]"
        print(f"{status_icon} {name.replace('_', ' ').title()}")
    
    print(f"\nTotal: {total_count} | Success: {success_count} | Failed: {total_count - success_count}")
    print("="*70)
    
    return results


def run_all_analyses():
    """Mode 3: Run all analyses comprehensively"""
    print("\n" + "="*70)
    print("MODE 3: ALL ANALYSES (COMPREHENSIVE)")
    print("="*70)
    
    print("\nThis will run ALL optimization analyses:")
    print("   1. Cost Optimization")
    print("   2. Route Optimization")
    print("   3. Delivery Prediction")
    print("   4. Risk Analysis")
    print("   5. Warehouse Location")
    
    confirm = input("\nProceed? This may take several minutes [y/N]: ").strip().lower()
    
    if confirm != 'y':
        print("Cancelled.")
        return None
    
    print("\n[COMPREHENSIVE MODE] Running all analyses with shared data...\n")
    
    # Run all pipelines
    master = MasterPipeline()
    results = master.run_all()
    
    # Generate master report
    master.save_master_report()
    
    # Summary
    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print("="*70)
    
    success_count = sum(1 for r in results.values() if r.get('status') == 'success')
    total_count = len(results)
    
    for name, result in results.items():
        status_icon = "[OK]" if result.get('status') == 'success' else "[FAIL]"
        print(f"{status_icon} {name.replace('_', ' ').title()}")
    
    print(f"\nTotal: {total_count} | Success: {success_count} | Failed: {total_count - success_count}")
    print(f"\nAll reports saved to: outputs/reports/")
    print(f"All figures saved to: outputs/figures/")
    print("="*70)
    
    return results


def main():
    """Main execution function"""
    print_banner()
    
    while True:
        print_menu()
        
        mode = get_user_choice("\nSelect mode [0-3]: ", ['0', '1', '2', '3'])
        
        if mode == '0':
            print("\nExiting. Goodbye!\n")
            sys.exit(0)
        
        elif mode == '1':
            run_single_analysis()
        
        elif mode == '2':
            run_selected_analyses()
        
        elif mode == '3':
            run_all_analyses()
        
        # Ask to continue
        print("\n")
        continue_choice = input("Run another analysis? [y/N]: ").strip().lower()
        if continue_choice != 'y':
            print("\nExiting. Goodbye!\n")
            break


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Execution cancelled by user.\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
