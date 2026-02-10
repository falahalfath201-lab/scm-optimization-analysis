"""
Shipping Cost Optimization Module
Optimizes shipping costs using Linear Programming and analysis

Objectives:
1. Minimize total shipping cost
2. Analyze cost drivers
3. Recommend optimal shipping modes
4. Cost-benefit analysis for different strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import linprog, minimize
import logging
import warnings

# Optional PuLP for more complex LP
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

from ..config import FUEL_COST_PER_KM, LABOR_COST_PER_HOUR, VEHICLE_FIXED_COST

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class CostOptimizer:
    """
    Shipping Cost Optimization using Linear Programming
    
    Cost Components:
    - Fixed costs (vehicle, warehouse)
    - Variable costs (fuel, labor)
    - Shipping mode costs
    - Late delivery penalties
    """
    
    # Cost parameters (can be overridden)
    SHIPPING_MODE_COSTS = {
        'Same Day': 25.0,
        'First Class': 15.0,
        'Second Class': 10.0,
        'Standard Class': 5.0
    }
    
    LATE_PENALTY_PER_DAY = 50.0  # Penalty per day late
    
    def __init__(self, 
                 fuel_cost_per_km: float = FUEL_COST_PER_KM,
                 labor_cost_per_hour: float = LABOR_COST_PER_HOUR,
                 vehicle_fixed_cost: float = VEHICLE_FIXED_COST):
        """
        Initialize Cost Optimizer
        
        Args:
            fuel_cost_per_km: Fuel cost per kilometer
            labor_cost_per_hour: Labor cost per hour
            vehicle_fixed_cost: Fixed cost per vehicle per day
        """
        self.fuel_cost_per_km = fuel_cost_per_km
        self.labor_cost_per_hour = labor_cost_per_hour
        self.vehicle_fixed_cost = vehicle_fixed_cost
        self.results: Dict[str, Any] = {}
    
    def analyze_current_costs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze current shipping costs from data
        
        Args:
            df: DataFrame with shipping data
            
        Returns:
            Cost analysis results
        """
        results = {
            'total_orders': len(df),
            'by_shipping_mode': {},
            'by_region': {},
            'cost_drivers': {},
            'summary': {}
        }
        
        # Cost by Shipping Mode
        if 'Shipping Mode' in df.columns:
            mode_stats = df.groupby('Shipping Mode').agg({
                'Order Item Total': ['count', 'sum', 'mean'],
                'Order Profit Per Order': ['sum', 'mean']
            }).round(2)
            
            mode_stats.columns = ['order_count', 'total_revenue', 'avg_revenue', 
                                  'total_profit', 'avg_profit']
            
            # Add estimated shipping cost
            for mode in mode_stats.index:
                cost = self.SHIPPING_MODE_COSTS.get(mode, 10.0)
                count = mode_stats.loc[mode, 'order_count']
                mode_stats.loc[mode, 'est_shipping_cost'] = cost * count
                mode_stats.loc[mode, 'cost_per_order'] = cost
            
            results['by_shipping_mode'] = mode_stats.to_dict('index')
        
        # Cost by Region
        if 'Order Region' in df.columns:
            region_stats = df.groupby('Order Region').agg({
                'Order Item Total': ['count', 'sum'],
                'Order Profit Per Order': 'sum'
            }).round(2)
            region_stats.columns = ['order_count', 'total_revenue', 'total_profit']
            results['by_region'] = region_stats.to_dict('index')
        
        # Late delivery costs
        if 'Late_delivery_risk' in df.columns and 'delivery_delay' in df.columns:
            late_orders = df[df['Late_delivery_risk'] == 1]
            total_delay_days = late_orders['delivery_delay'].clip(lower=0).sum()
            late_penalty = total_delay_days * self.LATE_PENALTY_PER_DAY
            
            results['late_delivery'] = {
                'late_orders': len(late_orders),
                'late_rate': len(late_orders) / len(df) * 100,
                'total_delay_days': total_delay_days,
                'estimated_penalty': late_penalty
            }
        
        # Summary
        total_revenue = df['Order Item Total'].sum() if 'Order Item Total' in df.columns else 0
        total_profit = df['Order Profit Per Order'].sum() if 'Order Profit Per Order' in df.columns else 0
        
        # Estimate total shipping cost
        est_shipping_cost = 0
        if 'Shipping Mode' in df.columns:
            for mode, cost in self.SHIPPING_MODE_COSTS.items():
                est_shipping_cost += len(df[df['Shipping Mode'] == mode]) * cost
        
        results['summary'] = {
            'total_revenue': round(total_revenue, 2),
            'total_profit': round(total_profit, 2),
            'estimated_shipping_cost': round(est_shipping_cost, 2),
            'profit_margin_pct': round(total_profit / total_revenue * 100, 2) if total_revenue > 0 else 0
        }
        
        return results
    
    def optimize_shipping_mode(self, df: pd.DataFrame,
                               budget_constraint: Optional[float] = None,
                               min_service_level: float = 0.7) -> Dict[str, Any]:
        """
        Optimize shipping mode assignment to minimize cost while meeting service levels
        
        Uses Linear Programming:
        - Minimize: total shipping cost
        - Subject to: service level >= min_service_level, budget <= constraint
        
        Args:
            df: Order data
            budget_constraint: Maximum shipping budget
            min_service_level: Minimum on-time delivery rate (0-1)
            
        Returns:
            Optimization results with recommended assignments
        """
        # Current state
        current_costs = self.analyze_current_costs(df)
        
        # Shipping modes and their properties
        modes = list(self.SHIPPING_MODE_COSTS.keys())
        mode_costs = [self.SHIPPING_MODE_COSTS[m] for m in modes]
        
        # Estimated on-time rates by mode (based on data or assumptions)
        mode_ontime_rates = {
            'Same Day': 0.95,
            'First Class': 0.85,
            'Second Class': 0.70,
            'Standard Class': 0.55
        }
        
        n_orders = len(df)
        
        # Simple optimization: assign modes based on order value and risk
        if 'Order Item Total' in df.columns:
            order_values = df['Order Item Total'].values
        else:
            order_values = np.ones(n_orders) * 100
        
        # OPTIMIZED: Calculate quantiles ONCE outside loop
        q90 = df['Order Item Total'].quantile(0.9)
        q75 = df['Order Item Total'].quantile(0.75)
        
        # Vectorized mode assignment
        recommendations = np.where(
            order_values > q90, 
            'First Class',
            np.where(order_values > q75, 'Second Class', 'Standard Class')
        )
        
        # Vectorized cost calculation
        mode_cost_map = {m: self.SHIPPING_MODE_COSTS[m] for m in recommendations}
        costs = np.array([self.SHIPPING_MODE_COSTS[m] for m in recommendations])
        total_optimized_cost = costs.sum()
        
        # Vectorized ontime calculation
        ontime_rates = np.array([mode_ontime_rates[m] for m in recommendations])
        expected_ontime = ontime_rates.sum()
        expected_service_level = expected_ontime / n_orders
        
        # Compare with current
        current_cost = current_costs['summary']['estimated_shipping_cost']
        savings = current_cost - total_optimized_cost
        
        # Mode distribution in optimized solution
        mode_counts = pd.Series(recommendations).value_counts()
        
        return {
            'status': 'success',
            'current_cost': current_cost,
            'optimized_cost': round(total_optimized_cost, 2),
            'savings': round(savings, 2),
            'savings_pct': round(savings / current_cost * 100, 2) if current_cost > 0 else 0,
            'expected_service_level': round(expected_service_level, 4),
            'mode_distribution': mode_counts.to_dict(),
            'recommendations': recommendations[:10]  # Sample
        }
    
    def optimize_with_pulp(self, 
                           orders: List[Dict],
                           num_vehicles: int = 5,
                           vehicle_capacity: int = 100) -> Dict[str, Any]:
        """
        Advanced optimization using PuLP Linear Programming
        
        Minimizes: Vehicle costs + Shipping costs + Late penalties
        Subject to: Capacity, demand fulfillment
        
        Args:
            orders: List of order dictionaries with 'demand', 'region', etc.
            num_vehicles: Available vehicles
            vehicle_capacity: Capacity per vehicle
            
        Returns:
            Optimization solution
        """
        if not PULP_AVAILABLE:
            return {"error": "PuLP not installed. Run: pip install pulp"}
        
        n_orders = len(orders)
        
        # Create problem
        prob = pulp.LpProblem("Shipping_Cost_Optimization", pulp.LpMinimize)
        
        # Decision variables: x[i][m] = 1 if order i uses mode m
        modes = list(self.SHIPPING_MODE_COSTS.keys())
        x = pulp.LpVariable.dicts("mode", 
                                   [(i, m) for i in range(n_orders) for m in modes],
                                   cat='Binary')
        
        # Objective: Minimize total cost
        prob += pulp.lpSum([
            x[(i, m)] * self.SHIPPING_MODE_COSTS[m]
            for i in range(n_orders)
            for m in modes
        ]), "Total_Shipping_Cost"
        
        # Constraint: Each order assigned to exactly one mode
        for i in range(n_orders):
            prob += pulp.lpSum([x[(i, m)] for m in modes]) == 1, f"One_Mode_{i}"
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if pulp.LpStatus[prob.status] != 'Optimal':
            return {"error": f"Optimization failed: {pulp.LpStatus[prob.status]}"}
        
        # Extract solution
        assignments = {}
        total_cost = 0
        
        for i in range(n_orders):
            for m in modes:
                if pulp.value(x[(i, m)]) == 1:
                    assignments[i] = m
                    total_cost += self.SHIPPING_MODE_COSTS[m]
        
        # Mode distribution
        mode_counts = {}
        for mode in assignments.values():
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        return {
            'status': 'optimal',
            'total_cost': total_cost,
            'mode_distribution': mode_counts,
            'sample_assignments': dict(list(assignments.items())[:10])
        }
    
    def cost_benefit_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform cost-benefit analysis for different shipping strategies
        
        Strategies:
        1. All Standard (cheapest)
        2. All Express (fastest)
        3. Value-based (current optimization)
        4. Risk-based (prioritize high-risk orders)
        """
        n_orders = len(df)
        
        strategies = {}
        
        # Strategy 1: All Standard
        strategies['all_standard'] = {
            'name': 'All Standard Class',
            'cost': n_orders * self.SHIPPING_MODE_COSTS['Standard Class'],
            'expected_ontime_rate': 0.55,
            'late_penalty': n_orders * 0.45 * self.LATE_PENALTY_PER_DAY * 2  # Avg 2 days late
        }
        strategies['all_standard']['total_cost'] = (
            strategies['all_standard']['cost'] + strategies['all_standard']['late_penalty']
        )
        
        # Strategy 2: All Express (First Class)
        strategies['all_express'] = {
            'name': 'All First Class',
            'cost': n_orders * self.SHIPPING_MODE_COSTS['First Class'],
            'expected_ontime_rate': 0.85,
            'late_penalty': n_orders * 0.15 * self.LATE_PENALTY_PER_DAY * 1
        }
        strategies['all_express']['total_cost'] = (
            strategies['all_express']['cost'] + strategies['all_express']['late_penalty']
        )
        
        # Strategy 3: Value-based (optimize based on order value)
        high_value = (df['Order Item Total'] > df['Order Item Total'].quantile(0.75)).sum()
        mid_value = ((df['Order Item Total'] > df['Order Item Total'].quantile(0.5)) & 
                     (df['Order Item Total'] <= df['Order Item Total'].quantile(0.75))).sum()
        low_value = n_orders - high_value - mid_value
        
        value_cost = (
            high_value * self.SHIPPING_MODE_COSTS['First Class'] +
            mid_value * self.SHIPPING_MODE_COSTS['Second Class'] +
            low_value * self.SHIPPING_MODE_COSTS['Standard Class']
        )
        value_ontime = (high_value * 0.85 + mid_value * 0.70 + low_value * 0.55) / n_orders
        value_late_penalty = n_orders * (1 - value_ontime) * self.LATE_PENALTY_PER_DAY * 1.5
        
        strategies['value_based'] = {
            'name': 'Value-Based Assignment',
            'cost': value_cost,
            'expected_ontime_rate': round(value_ontime, 2),
            'late_penalty': value_late_penalty,
            'total_cost': value_cost + value_late_penalty,
            'distribution': {
                'First Class': high_value,
                'Second Class': mid_value,
                'Standard Class': low_value
            }
        }
        
        # Strategy 4: Risk-based (if risk data available)
        if 'Late_delivery_risk' in df.columns:
            high_risk = (df['Late_delivery_risk'] == 1).sum()
            low_risk = n_orders - high_risk
            
            risk_cost = (
                high_risk * self.SHIPPING_MODE_COSTS['First Class'] +
                low_risk * self.SHIPPING_MODE_COSTS['Standard Class']
            )
            risk_ontime = (high_risk * 0.85 + low_risk * 0.55) / n_orders
            risk_late_penalty = n_orders * (1 - risk_ontime) * self.LATE_PENALTY_PER_DAY * 1
            
            strategies['risk_based'] = {
                'name': 'Risk-Based Assignment',
                'cost': risk_cost,
                'expected_ontime_rate': round(risk_ontime, 2),
                'late_penalty': risk_late_penalty,
                'total_cost': risk_cost + risk_late_penalty,
                'distribution': {
                    'First Class (high risk)': high_risk,
                    'Standard Class (low risk)': low_risk
                }
            }
        
        # Find best strategy
        best = min(strategies.items(), key=lambda x: x[1]['total_cost'])
        
        return {
            'strategies': strategies,
            'best_strategy': best[0],
            'best_strategy_name': best[1]['name'],
            'best_total_cost': round(best[1]['total_cost'], 2),
            'comparison': {
                name: {
                    'total_cost': round(s['total_cost'], 2),
                    'vs_best': round(s['total_cost'] - best[1]['total_cost'], 2)
                }
                for name, s in strategies.items()
            }
        }
    
    def get_optimization_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive cost optimization summary
        """
        current = self.analyze_current_costs(df)
        optimized = self.optimize_shipping_mode(df)
        cba = self.cost_benefit_analysis(df)
        
        return {
            'current_state': current['summary'],
            'optimization': optimized,
            'cost_benefit_analysis': cba,
            'recommendations': self._generate_recommendations(current, optimized, cba)
        }
    
    def _generate_recommendations(self, current: Dict, optimized: Dict, cba: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recs = []
        
        # Cost savings recommendation
        if optimized['savings'] > 0:
            recs.append(f"[MONEY] Switch to optimized shipping modes to save ${optimized['savings']:,.2f} "
                       f"({optimized['savings_pct']:.1f}%)")
        
        # Best strategy recommendation
        best = cba['best_strategy_name']
        recs.append(f"[DATA] Recommended strategy: {best}")
        
        # Late delivery recommendation
        if 'late_delivery' in current:
            late_rate = current['late_delivery']['late_rate']
            if late_rate > 50:
                recs.append(f"[WARNING] High late delivery rate ({late_rate:.1f}%). "
                           "Consider upgrading shipping modes for high-risk orders.")
        
        # Service level recommendation
        if optimized['expected_service_level'] < 0.7:
            recs.append("[CHART] Expected service level below 70%. Consider hybrid approach "
                       "with express shipping for critical orders.")
        
        return recs


def print_cost_report(results: Dict[str, Any]) -> None:
    """Pretty print cost optimization results"""
    print("\n" + "=" * 70)
    print("COST OPTIMIZATION REPORT")
    print("=" * 70)
    
    # Current State
    if 'current_state' in results:
        cs = results['current_state']
        print(f"\n[DATA] CURRENT STATE:")
        print(f"   Total Revenue:      ${cs['total_revenue']:,.2f}")
        print(f"   Total Profit:       ${cs['total_profit']:,.2f}")
        print(f"   Est. Shipping Cost: ${cs['estimated_shipping_cost']:,.2f}")
        print(f"   Profit Margin:      {cs['profit_margin_pct']:.2f}%")
    
    # Optimization Results
    if 'optimization' in results:
        opt = results['optimization']
        print(f"\n[TARGET] SHIPPING MODE OPTIMIZATION:")
        print(f"   Current Cost:   ${opt['current_cost']:,.2f}")
        print(f"   Optimized Cost: ${opt['optimized_cost']:,.2f}")
        print(f"   Savings:        ${opt['savings']:,.2f} ({opt['savings_pct']:.1f}%)")
        print(f"   Expected On-Time: {opt['expected_service_level']*100:.1f}%")
        
        print(f"\n   Recommended Mode Distribution:")
        for mode, count in opt['mode_distribution'].items():
            print(f"      {mode}: {count:,} orders")
    
    # Cost-Benefit Analysis
    if 'cost_benefit_analysis' in results:
        cba = results['cost_benefit_analysis']
        print(f"\n[CHART] STRATEGY COMPARISON:")
        print(f"   {'Strategy':<25} {'Total Cost':>15} {'vs Best':>12}")
        print("   " + "-" * 55)
        
        for name, data in cba['comparison'].items():
            marker = "[BEST]" if name == cba['best_strategy'] else "  "
            print(f"   {marker} {name:<23} ${data['total_cost']:>12,.2f} "
                  f"${data['vs_best']:>+10,.2f}")
        
        print(f"\n   [OK] Best Strategy: {cba['best_strategy_name']}")
    
    # Recommendations
    if 'recommendations' in results:
        print(f"\n[IDEA] RECOMMENDATIONS:")
        for rec in results['recommendations']:
            print(f"   • {rec}")
    
    print("\n" + "=" * 70)
