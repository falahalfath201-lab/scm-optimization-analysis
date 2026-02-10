"""
Warehouse Location Optimization Module
Determines optimal warehouse/distribution center locations using:
1. K-Means clustering
2. Center of Gravity method
3. P-Median facility location
4. Coverage and cost analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import logging
import warnings

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

from ..utils.distance_calculator import DistanceCalculator
from ..config import MAX_FACILITIES, FACILITY_FIXED_COST, FACILITY_CAPACITY

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class WarehouseOptimizer:
    """
    Optimize warehouse/distribution center locations
    
    Methods:
    1. K-Means Clustering: Group customers by proximity
    2. Center of Gravity: Find centroid weighted by demand
    3. P-Median: Minimize total distance-weighted cost
    4. Coverage Analysis: Analyze service coverage
    """
    
    def __init__(self, 
                 max_facilities: int = MAX_FACILITIES,
                 facility_cost: float = FACILITY_FIXED_COST,
                 facility_capacity: int = FACILITY_CAPACITY):
        """
        Initialize Warehouse Optimizer
        
        Args:
            max_facilities: Maximum number of facilities to consider
            facility_cost: Fixed cost per facility per period
            facility_capacity: Capacity per facility (orders/day)
        """
        self.max_facilities = max_facilities
        self.facility_cost = facility_cost
        self.facility_capacity = facility_capacity
        self.dist_calc = DistanceCalculator()
        self.results: Dict[str, Any] = {}
    
    def kmeans_location(self, df: pd.DataFrame, 
                       n_facilities: int = 5,
                       weight_by_demand: bool = True) -> Dict[str, Any]:
        """
        Find optimal locations using K-Means clustering
        
        Args:
            df: Customer location data with lat/lon
            n_facilities: Number of facilities to locate
            weight_by_demand: Weight clustering by order volume
            
        Returns:
            Facility locations and assignments
        """
        # Extract coordinates
        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            return {"error": "Missing Latitude/Longitude columns"}
        
        coords = df[['Latitude', 'Longitude']].dropna()
        if len(coords) < n_facilities:
            return {"error": f"Insufficient data points ({len(coords)}) for {n_facilities} facilities"}
        
        # Prepare weights
        if weight_by_demand and 'Order Item Quantity' in df.columns:
            weights = df.loc[coords.index, 'Order Item Quantity'].fillna(1).values
            # Repeat points based on weights for weighted clustering
            weighted_coords = np.repeat(coords.values, 
                                       np.maximum(1, (weights / weights.mean()).astype(int)), 
                                       axis=0)
        else:
            weighted_coords = coords.values
        
        # Apply K-Means
        kmeans = KMeans(n_clusters=n_facilities, random_state=42, n_init=10)
        kmeans.fit(weighted_coords)
        
        # Get facility locations (cluster centers)
        facilities = kmeans.cluster_centers_
        
        # Assign original points to nearest facility
        labels = kmeans.predict(coords.values)
        
        # Calculate statistics per facility
        facility_stats = []
        total_distance = 0
        
        for i in range(n_facilities):
            cluster_mask = labels == i
            cluster_points = coords.values[cluster_mask]
            cluster_indices = coords.index[cluster_mask]
            
            if len(cluster_points) == 0:
                continue
            
            # Calculate distances to facility
            distances = self.dist_calc.haversine_vectorized(
                cluster_points[:, 0], cluster_points[:, 1],
                facilities[i, 0], facilities[i, 1]
            )
            
            # Demand served
            demand = df.loc[cluster_indices, 'Order Item Quantity'].sum() if 'Order Item Quantity' in df.columns else len(cluster_points)
            
            facility_stats.append({
                'facility_id': i,
                'latitude': facilities[i, 0],
                'longitude': facilities[i, 1],
                'customers_served': len(cluster_points),
                'total_demand': demand,
                'avg_distance_km': distances.mean(),
                'max_distance_km': distances.max(),
                'total_distance_km': distances.sum()
            })
            
            total_distance += distances.sum()
        
        # Calculate costs
        total_facility_cost = n_facilities * self.facility_cost
        avg_distance_per_order = total_distance / len(coords)
        
        return {
            'status': 'success',
            'method': 'K-Means Clustering',
            'n_facilities': n_facilities,
            'facilities': facility_stats,
            'total_customers': len(coords),
            'total_distance_km': round(total_distance, 2),
            'avg_distance_km': round(avg_distance_per_order, 2),
            'total_facility_cost': total_facility_cost,
            'weighted_by_demand': weight_by_demand
        }
    
    def center_of_gravity(self, df: pd.DataFrame, 
                         weight_column: str = 'Order Item Quantity') -> Dict[str, Any]:
        """
        Find single optimal location using Center of Gravity method
        
        Weighted centroid based on demand/volume
        
        Args:
            df: Customer location data
            weight_column: Column to use as weights
            
        Returns:
            Optimal single facility location
        """
        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            return {"error": "Missing Latitude/Longitude columns"}
        
        coords = df[['Latitude', 'Longitude', weight_column]].dropna()
        
        if len(coords) == 0:
            return {"error": "No valid data"}
        
        # Calculate weighted centroid
        weights = coords[weight_column].values
        total_weight = weights.sum()
        
        center_lat = (coords['Latitude'] * weights).sum() / total_weight
        center_lon = (coords['Longitude'] * weights).sum() / total_weight
        
        # Calculate distances from all customers to this center
        distances = self.dist_calc.haversine_vectorized(
            coords['Latitude'].values,
            coords['Longitude'].values,
            center_lat, center_lon
        )
        
        # Weighted distances
        weighted_distances = distances * weights
        
        return {
            'status': 'success',
            'method': 'Center of Gravity',
            'facility_location': {
                'latitude': round(center_lat, 6),
                'longitude': round(center_lon, 6)
            },
            'total_customers': len(coords),
            'total_demand': total_weight,
            'avg_distance_km': round(distances.mean(), 2),
            'weighted_avg_distance_km': round(weighted_distances.sum() / total_weight, 2),
            'max_distance_km': round(distances.max(), 2),
            'total_distance_km': round(distances.sum(), 2)
        }
    
    def p_median_optimization(self, df: pd.DataFrame,
                            n_facilities: int = 5,
                            candidate_locations: Optional[List[Tuple[float, float]]] = None,
                            max_customers: int = 5000) -> Dict[str, Any]:
        """
        P-Median facility location problem using Linear Programming
        
        Minimizes: Sum of distance-weighted assignments
        Subject to: Each customer assigned to exactly one facility
        
        Args:
            df: Customer location data
            n_facilities: Number of facilities (p)
            candidate_locations: Optional list of candidate (lat, lon) locations
            max_customers: Maximum customers for LP (default 5000 for performance)
            
        Returns:
            Optimal facility locations and assignments
        """
        if not PULP_AVAILABLE:
            return {"error": "PuLP not installed. Run: pip install pulp"}
        
        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            return {"error": "Missing Latitude/Longitude columns"}
        
        coords = df[['Latitude', 'Longitude']].dropna()
        if len(coords) < n_facilities:
            return {"error": f"Insufficient data points for {n_facilities} facilities"}
        
        # OPTIMIZATION: Sample customers for LP performance
        # Too many customers = too many constraints = very slow
        if len(coords) > max_customers:
            sample_indices = np.random.choice(len(coords), max_customers, replace=False)
            coords = coords.iloc[sample_indices]
            logger.info(f"Sampled {max_customers} customers from {len(df)} for p-median optimization")
        
        # If no candidate locations provided, use sample of customer locations
        if candidate_locations is None:
            # Use stratified sampling to get candidate locations
            sample_size = min(30, len(coords))  # Reduced from 50 to 30
            candidate_indices = np.linspace(0, len(coords)-1, sample_size, dtype=int)
            candidates = coords.iloc[candidate_indices].values
        else:
            candidates = np.array(candidate_locations)
        
        n_customers = len(coords)
        n_candidates = len(candidates)
        
        # Calculate distance matrix (customers x candidates)
        customer_coords = coords.values
        dist_matrix = cdist(customer_coords, candidates, 
                           metric=lambda u, v: self.dist_calc.haversine(u[0], u[1], v[0], v[1]))
        
        # Demand weights
        if 'Order Item Quantity' in df.columns:
            demand = df.loc[coords.index, 'Order Item Quantity'].fillna(1).values
        else:
            demand = np.ones(n_customers)
        
        # Create LP problem
        prob = pulp.LpProblem("P_Median_Location", pulp.LpMinimize)
        
        # Decision variables
        # x[i][j] = 1 if customer i is assigned to facility j
        x = pulp.LpVariable.dicts("assign",
                                  [(i, j) for i in range(n_customers) for j in range(n_candidates)],
                                  cat='Binary')
        
        # y[j] = 1 if facility j is opened
        y = pulp.LpVariable.dicts("open",
                                  [j for j in range(n_candidates)],
                                  cat='Binary')
        
        # Objective: Minimize weighted distance
        prob += pulp.lpSum([
            demand[i] * dist_matrix[i, j] * x[(i, j)]
            for i in range(n_customers)
            for j in range(n_candidates)
        ]), "Total_Weighted_Distance"
        
        # Constraints
        # 1. Each customer assigned to exactly one facility
        for i in range(n_customers):
            prob += pulp.lpSum([x[(i, j)] for j in range(n_candidates)]) == 1, f"Customer_{i}_Assignment"
        
        # 2. Open exactly n_facilities
        prob += pulp.lpSum([y[j] for j in range(n_candidates)]) == n_facilities, "Facility_Count"
        
        # 3. Can only assign to open facilities
        for i in range(n_customers):
            for j in range(n_candidates):
                prob += x[(i, j)] <= y[j], f"Open_Facility_{i}_{j}"
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if pulp.LpStatus[prob.status] != 'Optimal':
            return {"error": f"Optimization failed: {pulp.LpStatus[prob.status]}"}
        
        # Extract solution
        opened_facilities = [j for j in range(n_candidates) if pulp.value(y[j]) == 1]
        
        facility_stats = []
        total_distance = 0
        
        for fac_idx in opened_facilities:
            assigned_customers = [i for i in range(n_customers) if pulp.value(x[(i, fac_idx)]) == 1]
            
            if not assigned_customers:
                continue
            
            distances = dist_matrix[assigned_customers, fac_idx]
            customer_demand = demand[assigned_customers]
            
            facility_stats.append({
                'facility_id': len(facility_stats),
                'latitude': candidates[fac_idx, 0],
                'longitude': candidates[fac_idx, 1],
                'customers_served': len(assigned_customers),
                'total_demand': customer_demand.sum(),
                'avg_distance_km': distances.mean(),
                'weighted_avg_distance_km': (distances * customer_demand).sum() / customer_demand.sum(),
                'total_distance_km': distances.sum()
            })
            
            total_distance += distances.sum()
        
        return {
            'status': 'optimal',
            'method': 'P-Median Optimization',
            'n_facilities': n_facilities,
            'facilities': facility_stats,
            'total_customers': n_customers,
            'total_distance_km': round(total_distance, 2),
            'avg_distance_km': round(total_distance / n_customers, 2),
            'total_facility_cost': n_facilities * self.facility_cost,
            'objective_value': round(pulp.value(prob.objective), 2)
        }
    
    def coverage_analysis(self, df: pd.DataFrame,
                         facilities: List[Dict],
                         coverage_radius_km: float = 100) -> Dict[str, Any]:
        """
        Analyze coverage of facilities
        
        Args:
            df: Customer locations
            facilities: List of facility dicts with 'latitude', 'longitude'
            coverage_radius_km: Maximum service radius
            
        Returns:
            Coverage statistics
        """
        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            return {"error": "Missing Latitude/Longitude columns"}
        
        coords = df[['Latitude', 'Longitude']].dropna()
        
        # Calculate distance from each customer to each facility
        coverage_stats = []
        customers_covered = set()
        
        for i, facility in enumerate(facilities):
            distances = self.dist_calc.haversine_vectorized(
                coords['Latitude'].values,
                coords['Longitude'].values,
                facility['latitude'],
                facility['longitude']
            )
            
            within_coverage = distances <= coverage_radius_km
            covered_indices = coords.index[within_coverage]
            customers_covered.update(covered_indices)
            
            coverage_stats.append({
                'facility_id': i,
                'customers_in_range': within_coverage.sum(),
                'coverage_pct': round(within_coverage.sum() / len(coords) * 100, 2)
            })
        
        total_coverage = len(customers_covered) / len(coords) * 100
        
        return {
            'coverage_radius_km': coverage_radius_km,
            'total_customers': len(coords),
            'customers_covered': len(customers_covered),
            'total_coverage_pct': round(total_coverage, 2),
            'facility_coverage': coverage_stats,
            'customers_uncovered': len(coords) - len(customers_covered)
        }
    
    def compare_scenarios(self, df: pd.DataFrame,
                         facility_counts: List[int] = [3, 5, 7, 10]) -> Dict[str, Any]:
        """
        Compare different facility count scenarios
        
        Args:
            df: Customer location data
            facility_counts: List of facility counts to compare
            
        Returns:
            Comparison results
        """
        scenarios = {}
        
        for n in facility_counts:
            if n > len(df) / 10:  # Skip if too many facilities
                continue
            
            result = self.kmeans_location(df, n_facilities=n, weight_by_demand=True)
            
            if result.get('status') == 'success':
                scenarios[f'{n}_facilities'] = {
                    'n_facilities': n,
                    'total_distance_km': result['total_distance_km'],
                    'avg_distance_km': result['avg_distance_km'],
                    'total_cost': result['total_facility_cost'],
                    'cost_per_customer': result['total_facility_cost'] / result['total_customers']
                }
        
        # Find best scenario (minimize total cost + distance cost)
        # Assume $1 per km transportation cost
        TRANSPORT_COST_PER_KM = 1.0
        
        for scenario_name, scenario_data in scenarios.items():
            transport_cost = scenario_data['total_distance_km'] * TRANSPORT_COST_PER_KM
            scenario_data['transport_cost'] = transport_cost
            scenario_data['total_system_cost'] = scenario_data['total_cost'] + transport_cost
        
        if scenarios:
            best_scenario = min(scenarios.items(), key=lambda x: x[1]['total_system_cost'])
            
            return {
                'scenarios': scenarios,
                'best_scenario': best_scenario[0],
                'best_n_facilities': best_scenario[1]['n_facilities'],
                'best_total_cost': round(best_scenario[1]['total_system_cost'], 2)
            }
        
        return {"error": "No valid scenarios generated"}
    
    def optimize_by_region(self, df: pd.DataFrame,
                          region_column: str = 'Order Region',
                          facilities_per_region: int = 2) -> Dict[str, Any]:
        """
        Optimize warehouse locations per region
        
        Args:
            df: Order data with region information
            region_column: Column containing region names
            facilities_per_region: Number of facilities per region
            
        Returns:
            Regional optimization results
        """
        if region_column not in df.columns:
            return {"error": f"Column {region_column} not found"}
        
        results = {}
        
        for region in df[region_column].unique():
            if pd.isna(region):
                continue
            
            region_df = df[df[region_column] == region]
            
            if len(region_df) < facilities_per_region * 5:  # Need minimum data
                continue
            
            region_result = self.kmeans_location(
                region_df, 
                n_facilities=facilities_per_region,
                weight_by_demand=True
            )
            
            if region_result.get('status') == 'success':
                results[region] = {
                    'n_facilities': facilities_per_region,
                    'customers': region_result['total_customers'],
                    'avg_distance_km': region_result['avg_distance_km'],
                    'facilities': region_result['facilities']
                }
        
        # Summary
        total_facilities = sum(r['n_facilities'] for r in results.values())
        total_customers = sum(r['customers'] for r in results.values())
        avg_distance = np.mean([r['avg_distance_km'] for r in results.values()])
        
        return {
            'status': 'success',
            'method': 'Regional Optimization',
            'regions': results,
            'summary': {
                'total_regions': len(results),
                'total_facilities': total_facilities,
                'total_customers': total_customers,
                'avg_distance_km': round(avg_distance, 2)
            }
        }


def print_warehouse_report(results: Dict[str, Any]) -> None:
    """Pretty print warehouse optimization results"""
    print("\n" + "=" * 70)
    print("WAREHOUSE LOCATION OPTIMIZATION REPORT")
    print("=" * 70)
    
    method = results.get('method', 'Unknown')
    print(f"\n[PIN] Method: {method}")
    
    if 'facilities' in results:
        print(f"\n[FACILITY] FACILITY LOCATIONS:")
        print(f"   Number of Facilities: {results.get('n_facilities', len(results['facilities']))}")
        print(f"   Total Customers: {results.get('total_customers', 'N/A'):,}")
        
        if 'total_distance_km' in results:
            print(f"   Total Distance: {results['total_distance_km']:,.2f} km")
            print(f"   Avg Distance: {results['avg_distance_km']:.2f} km")
        
        if 'total_facility_cost' in results:
            print(f"   Total Facility Cost: ${results['total_facility_cost']:,.2f}")
        
        print(f"\n   Facility Details:")
        print(f"   {'ID':<4} {'Latitude':>10} {'Longitude':>11} {'Customers':>10} {'Demand':>10} {'Avg Dist (km)':>14}")
        print("   " + "-" * 65)
        
        for fac in results['facilities']:
            print(f"   {fac['facility_id']:<4} {fac['latitude']:>10.4f} {fac['longitude']:>11.4f} "
                  f"{fac['customers_served']:>10,} {fac.get('total_demand', 0):>10,.0f} "
                  f"{fac['avg_distance_km']:>14.2f}")
    
    if 'facility_location' in results:  # Center of Gravity
        loc = results['facility_location']
        print(f"\n[TARGET] OPTIMAL SINGLE LOCATION:")
        print(f"   Latitude:  {loc['latitude']}")
        print(f"   Longitude: {loc['longitude']}")
        print(f"   Total Customers: {results['total_customers']:,}")
        print(f"   Avg Distance: {results['avg_distance_km']:.2f} km")
        print(f"   Weighted Avg Distance: {results['weighted_avg_distance_km']:.2f} km")
    
    if 'scenarios' in results:  # Scenario comparison
        print(f"\n[DATA] SCENARIO COMPARISON:")
        print(f"   {'Scenario':<15} {'Facilities':>11} {'Avg Dist (km)':>14} {'Facility Cost':>14} {'Transport Cost':>15} {'TOTAL':>12}")
        print("   " + "-" * 85)
        
        for name, scenario in results['scenarios'].items():
            print(f"   {name:<15} {scenario['n_facilities']:>11} "
                  f"{scenario['avg_distance_km']:>14.2f} "
                  f"${scenario['total_cost']:>13,.0f} "
                  f"${scenario['transport_cost']:>14,.0f} "
                  f"${scenario['total_system_cost']:>11,.0f}")
        
        print(f"\n   [OK] Best: {results['best_scenario']} with total cost ${results['best_total_cost']:,.2f}")
    
    if 'regions' in results:  # Regional optimization
        print(f"\n🗺️ REGIONAL OPTIMIZATION:")
        print(f"   Total Regions: {results['summary']['total_regions']}")
        print(f"   Total Facilities: {results['summary']['total_facilities']}")
        print(f"   Total Customers: {results['summary']['total_customers']:,}")
        print(f"   Avg Distance: {results['summary']['avg_distance_km']:.2f} km")
    
    print("\n" + "=" * 70)
