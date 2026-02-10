"""
VRP (Vehicle Routing Problem) Optimizer
Extends TSP to multi-vehicle routing with time minimization focus

Approach: Cluster-First, Route-Second
1. Cluster customers into groups (K-Means or capacity-based)
2. Solve TSP for each cluster using best method from benchmark
3. Minimize total time (or balance time across vehicles)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import time
import logging

from .tsp_optimizer import TSPOptimizer
from ..utils.distance_calculator import DistanceCalculator

logger = logging.getLogger(__name__)


class VRPOptimizer:
    """
    VRP Optimizer with time minimization focus
    
    Strategies:
    1. Cluster-TSP: K-Means clustering + TSP per cluster
    2. Balanced VRP: Minimize max route time (load balancing)
    3. Total Time VRP: Minimize total travel time
    """
    
    # Average speed assumptions (km/h)
    SPEEDS = {
        'urban': 30,
        'suburban': 50,
        'highway': 80,
        'default': 40
    }
    
    # Service time per stop (minutes)
    SERVICE_TIME_MIN = 10
    
    def __init__(self, num_vehicles: int = 3, 
                 speed_kmh: float = 40,
                 max_time_per_vehicle_min: float = 480):
        """
        Initialize VRP optimizer
        
        Args:
            num_vehicles: Number of available vehicles
            speed_kmh: Average travel speed in km/h
            max_time_per_vehicle_min: Max route time per vehicle (default 8 hours)
        """
        self.num_vehicles = num_vehicles
        self.speed_kmh = speed_kmh
        self.max_time_min = max_time_per_vehicle_min
        self.tsp_optimizer = TSPOptimizer()
        self.calculator = DistanceCalculator()
    
    def optimize(self, locations: List[Tuple[float, float]],
                 depot_index: int = 0,
                 demands: Optional[List[int]] = None,
                 strategy: str = 'cluster_tsp') -> Dict[str, Any]:
        """
        Run VRP optimization
        
        Args:
            locations: List of (lat, lon) including depot at depot_index
            depot_index: Index of depot in locations
            demands: Optional demand per location
            strategy: 'cluster_tsp', 'balanced', or 'min_total'
            
        Returns:
            VRP solution with routes per vehicle
        """
        n = len(locations)
        if n < 2:
            return {"error": "Need at least 2 locations"}
        
        start_time = time.time()
        
        if strategy == 'cluster_tsp':
            result = self._cluster_tsp_strategy(locations, depot_index)
        elif strategy == 'balanced':
            result = self._balanced_strategy(locations, depot_index)
        elif strategy == 'min_total':
            result = self._min_total_strategy(locations, depot_index)
        else:
            result = self._cluster_tsp_strategy(locations, depot_index)
        
        result['computation_time_ms'] = round((time.time() - start_time) * 1000, 2)
        result['strategy'] = strategy
        result['num_vehicles'] = self.num_vehicles
        result['speed_kmh'] = self.speed_kmh
        
        return result
    
    def _cluster_tsp_strategy(self, locations: List[Tuple[float, float]],
                               depot_index: int) -> Dict[str, Any]:
        """
        Cluster-First, Route-Second strategy
        1. Use K-Means to cluster customers
        2. Solve TSP for each cluster
        """
        # Separate depot from customers
        depot = locations[depot_index]
        customers = [loc for i, loc in enumerate(locations) if i != depot_index]
        customer_indices = [i for i in range(len(locations)) if i != depot_index]
        
        if len(customers) < self.num_vehicles:
            # Fewer customers than vehicles - assign one per vehicle
            return self._assign_simple(locations, depot_index, customers, customer_indices)
        
        # K-Means clustering
        coords = np.array(customers)
        kmeans = KMeans(n_clusters=self.num_vehicles, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        
        # Solve TSP for each cluster
        routes = []
        total_distance = 0
        total_time = 0
        max_time = 0
        
        for vehicle_id in range(self.num_vehicles):
            cluster_mask = labels == vehicle_id
            cluster_customers = [customers[i] for i in range(len(customers)) if cluster_mask[i]]
            cluster_indices = [customer_indices[i] for i in range(len(customers)) if cluster_mask[i]]
            
            if len(cluster_customers) == 0:
                routes.append({
                    'vehicle_id': vehicle_id,
                    'route_indices': [depot_index],
                    'route_coords': [depot],
                    'distance_km': 0,
                    'time_min': 0,
                    'num_stops': 0
                })
                continue
            
            # Add depot to cluster and solve TSP
            cluster_locs = [depot] + cluster_customers
            tsp_result = self.tsp_optimizer.optimize(cluster_locs, depot_index=0, method='2opt')
            
            # Extract route
            tsp_route = tsp_result['methods']['2opt']['route']
            distance = tsp_result['methods']['2opt']['total_distance_km']
            
            # Calculate time
            travel_time = (distance / self.speed_kmh) * 60  # minutes
            service_time = len(cluster_customers) * self.SERVICE_TIME_MIN
            route_time = travel_time + service_time
            
            # Map back to original indices
            route_indices = [depot_index]
            for r in tsp_route[1:-1]:  # Skip first and last (depot)
                route_indices.append(cluster_indices[r - 1])
            route_indices.append(depot_index)
            
            routes.append({
                'vehicle_id': vehicle_id,
                'route_indices': route_indices,
                'route_coords': [locations[i] for i in route_indices],
                'distance_km': round(distance, 2),
                'travel_time_min': round(travel_time, 2),
                'service_time_min': service_time,
                'total_time_min': round(route_time, 2),
                'num_stops': len(cluster_customers)
            })
            
            total_distance += distance
            total_time += route_time
            max_time = max(max_time, route_time)
        
        return {
            'status': 'success',
            'routes': routes,
            'total_distance_km': round(total_distance, 2),
            'total_time_min': round(total_time, 2),
            'max_route_time_min': round(max_time, 2),
            'avg_route_time_min': round(total_time / self.num_vehicles, 2),
            'time_balance_ratio': round(max_time / (total_time / self.num_vehicles), 2) if total_time > 0 else 1.0,
            'vehicles_used': len([r for r in routes if r['num_stops'] > 0])
        }
    
    def _assign_simple(self, locations, depot_index, customers, customer_indices):
        """Simple assignment when customers < vehicles"""
        depot = locations[depot_index]
        routes = []
        total_distance = 0
        total_time = 0
        
        for vehicle_id in range(self.num_vehicles):
            if vehicle_id < len(customers):
                cust = customers[vehicle_id]
                cust_idx = customer_indices[vehicle_id]
                dist = self.calculator.haversine_distance(depot, cust) * 2  # Round trip
                travel_time = (dist / self.speed_kmh) * 60
                service_time = self.SERVICE_TIME_MIN
                
                routes.append({
                    'vehicle_id': vehicle_id,
                    'route_indices': [depot_index, cust_idx, depot_index],
                    'route_coords': [depot, cust, depot],
                    'distance_km': round(dist, 2),
                    'travel_time_min': round(travel_time, 2),
                    'service_time_min': service_time,
                    'total_time_min': round(travel_time + service_time, 2),
                    'num_stops': 1
                })
                total_distance += dist
                total_time += travel_time + service_time
            else:
                routes.append({
                    'vehicle_id': vehicle_id,
                    'route_indices': [depot_index],
                    'route_coords': [depot],
                    'distance_km': 0,
                    'travel_time_min': 0,
                    'service_time_min': 0,
                    'total_time_min': 0,
                    'num_stops': 0
                })
        
        return {
            'status': 'success',
            'routes': routes,
            'total_distance_km': round(total_distance, 2),
            'total_time_min': round(total_time, 2),
            'max_route_time_min': max(r['total_time_min'] for r in routes),
            'avg_route_time_min': round(total_time / len(customers), 2) if customers else 0,
            'vehicles_used': len(customers)
        }
    
    def _balanced_strategy(self, locations: List[Tuple[float, float]],
                           depot_index: int) -> Dict[str, Any]:
        """
        Balanced VRP - minimize the maximum route time
        Uses iterative assignment to balance workload
        """
        depot = locations[depot_index]
        customers = [(i, loc) for i, loc in enumerate(locations) if i != depot_index]
        
        # Sort customers by distance from depot (farthest first)
        customers.sort(key=lambda x: self.calculator.haversine_distance(depot, x[1]), reverse=True)
        
        # Initialize vehicle routes
        vehicle_routes = [[] for _ in range(self.num_vehicles)]
        vehicle_times = [0.0] * self.num_vehicles
        
        # Assign customers to vehicle with minimum current time
        for cust_idx, cust_loc in customers:
            # Find vehicle with minimum time
            min_vehicle = min(range(self.num_vehicles), key=lambda v: vehicle_times[v])
            
            # Calculate added time
            if len(vehicle_routes[min_vehicle]) == 0:
                # First customer - from depot
                added_dist = self.calculator.haversine_distance(depot, cust_loc) * 2
            else:
                # Insert at end of current route
                last_cust = vehicle_routes[min_vehicle][-1][1]
                # Remove return to depot, add new customer, add return
                old_return = self.calculator.haversine_distance(last_cust, depot)
                new_leg = self.calculator.haversine_distance(last_cust, cust_loc)
                new_return = self.calculator.haversine_distance(cust_loc, depot)
                added_dist = new_leg + new_return - old_return
            
            added_time = (added_dist / self.speed_kmh) * 60 + self.SERVICE_TIME_MIN
            
            vehicle_routes[min_vehicle].append((cust_idx, cust_loc))
            vehicle_times[min_vehicle] += added_time
        
        # Build final routes with TSP optimization per vehicle
        routes = []
        total_distance = 0
        total_time = 0
        
        for vehicle_id in range(self.num_vehicles):
            if len(vehicle_routes[vehicle_id]) == 0:
                routes.append({
                    'vehicle_id': vehicle_id,
                    'route_indices': [depot_index],
                    'route_coords': [depot],
                    'distance_km': 0,
                    'total_time_min': 0,
                    'num_stops': 0
                })
                continue
            
            # Optimize route with TSP
            cluster_locs = [depot] + [c[1] for c in vehicle_routes[vehicle_id]]
            cluster_indices = [depot_index] + [c[0] for c in vehicle_routes[vehicle_id]]
            
            tsp_result = self.tsp_optimizer.optimize(cluster_locs, depot_index=0, method='2opt')
            tsp_route = tsp_result['methods']['2opt']['route']
            distance = tsp_result['methods']['2opt']['total_distance_km']
            
            # Map back
            route_indices = [cluster_indices[r] for r in tsp_route]
            
            travel_time = (distance / self.speed_kmh) * 60
            service_time = len(vehicle_routes[vehicle_id]) * self.SERVICE_TIME_MIN
            route_time = travel_time + service_time
            
            routes.append({
                'vehicle_id': vehicle_id,
                'route_indices': route_indices,
                'route_coords': [locations[i] for i in route_indices],
                'distance_km': round(distance, 2),
                'travel_time_min': round(travel_time, 2),
                'service_time_min': service_time,
                'total_time_min': round(route_time, 2),
                'num_stops': len(vehicle_routes[vehicle_id])
            })
            
            total_distance += distance
            total_time += route_time
        
        max_time = max(r['total_time_min'] for r in routes)
        
        return {
            'status': 'success',
            'routes': routes,
            'total_distance_km': round(total_distance, 2),
            'total_time_min': round(total_time, 2),
            'max_route_time_min': round(max_time, 2),
            'avg_route_time_min': round(total_time / self.num_vehicles, 2),
            'time_balance_ratio': round(max_time / (total_time / self.num_vehicles), 2) if total_time > 0 else 1.0,
            'vehicles_used': len([r for r in routes if r['num_stops'] > 0])
        }
    
    def _min_total_strategy(self, locations: List[Tuple[float, float]],
                            depot_index: int) -> Dict[str, Any]:
        """
        Minimize total time using OR-Tools VRP
        """
        n = len(locations)
        depot = locations[depot_index]
        
        # Create time matrix (in minutes)
        time_matrix = self._create_time_matrix(locations)
        
        # OR-Tools setup
        manager = pywrapcp.RoutingIndexManager(n, self.num_vehicles, depot_index)
        routing = pywrapcp.RoutingModel(manager)
        
        def time_callback(from_idx, to_idx):
            from_node = manager.IndexToNode(from_idx)
            to_node = manager.IndexToNode(to_idx)
            return time_matrix[from_node][to_node]
        
        transit_cb = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)
        
        # Time dimension with max constraint
        max_time_int = int(self.max_time_min)
        routing.AddDimension(
            transit_cb,
            30,  # slack
            max_time_int,
            True,
            'Time'
        )
        
        # Search parameters
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.time_limit.seconds = 30
        
        solution = routing.SolveWithParameters(search_params)
        
        if not solution:
            # Fallback to cluster strategy
            logger.warning("OR-Tools VRP failed, falling back to cluster strategy")
            return self._cluster_tsp_strategy(locations, depot_index)
        
        # Extract solution
        routes = []
        total_time = 0
        total_distance = 0
        
        for vehicle_id in range(self.num_vehicles):
            route_indices = []
            index = routing.Start(vehicle_id)
            route_time = 0
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route_indices.append(node)
                prev = index
                index = solution.Value(routing.NextVar(index))
                route_time += routing.GetArcCostForVehicle(prev, index, vehicle_id)
            
            route_indices.append(depot_index)  # Return to depot
            
            # Calculate distance
            distance = sum(
                self.calculator.haversine_distance(locations[route_indices[i]], locations[route_indices[i+1]])
                for i in range(len(route_indices) - 1)
            )
            
            num_stops = len(route_indices) - 2  # Exclude depot start/end
            
            routes.append({
                'vehicle_id': vehicle_id,
                'route_indices': route_indices,
                'route_coords': [locations[i] for i in route_indices],
                'distance_km': round(distance, 2),
                'total_time_min': route_time,
                'num_stops': num_stops
            })
            
            total_time += route_time
            total_distance += distance
        
        max_time = max(r['total_time_min'] for r in routes)
        
        return {
            'status': 'success',
            'routes': routes,
            'total_distance_km': round(total_distance, 2),
            'total_time_min': round(total_time, 2),
            'max_route_time_min': round(max_time, 2),
            'avg_route_time_min': round(total_time / self.num_vehicles, 2),
            'time_balance_ratio': round(max_time / (total_time / self.num_vehicles), 2) if total_time > 0 else 1.0,
            'vehicles_used': len([r for r in routes if r['num_stops'] > 0])
        }
    
    def _create_time_matrix(self, locations: List[Tuple[float, float]]) -> List[List[int]]:
        """Create time matrix in minutes (integer)"""
        n = len(locations)
        matrix = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = self.calculator.haversine_distance(locations[i], locations[j])
                    travel_time = (dist / self.speed_kmh) * 60
                    service_time = self.SERVICE_TIME_MIN if j != 0 else 0  # No service at depot
                    matrix[i][j] = int(travel_time + service_time)
        
        return matrix
    
    def optimize_region(self, df: pd.DataFrame,
                       region_column: str,
                       region_value: str,
                       lat_col: str = 'Latitude',
                       lon_col: str = 'Longitude',
                       max_locations: int = 50,
                       strategy: str = 'all') -> Dict[str, Any]:
        """
        Optimize VRP for a specific region with strategy comparison
        """
        # Filter by region
        region_df = df[df[region_column] == region_value].copy()
        
        if len(region_df) == 0:
            return {"error": f"No data found for {region_column}={region_value}"}
        
        # Get unique locations
        unique_locs = region_df[[lat_col, lon_col]].drop_duplicates()
        
        if len(unique_locs) > max_locations:
            unique_locs = unique_locs.head(max_locations)
        
        # Add depot at centroid
        centroid = (unique_locs[lat_col].mean(), unique_locs[lon_col].mean())
        locations = [centroid] + list(zip(unique_locs[lat_col], unique_locs[lon_col]))
        
        results = {
            'region': {
                'column': region_column,
                'value': region_value,
                'total_orders': len(region_df),
                'unique_locations': len(locations) - 1,
                'depot_location': centroid
            },
            'config': {
                'num_vehicles': self.num_vehicles,
                'speed_kmh': self.speed_kmh,
                'max_time_per_vehicle_min': self.max_time_min,
                'service_time_per_stop_min': self.SERVICE_TIME_MIN
            },
            'strategies': {}
        }
        
        strategies_to_run = ['cluster_tsp', 'balanced', 'min_total'] if strategy == 'all' else [strategy]
        
        for s in strategies_to_run:
            results['strategies'][s] = self.optimize(locations, depot_index=0, strategy=s)
        
        # Benchmark
        if len(results['strategies']) > 1:
            results['benchmark'] = self._benchmark_strategies(results['strategies'])
        
        return results
    
    def _benchmark_strategies(self, strategies: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare strategies"""
        comparisons = []
        
        for name, result in strategies.items():
            if result.get('status') == 'success':
                comparisons.append({
                    'strategy': name,
                    'total_time_min': result['total_time_min'],
                    'max_route_time_min': result['max_route_time_min'],
                    'total_distance_km': result['total_distance_km'],
                    'time_balance_ratio': result.get('time_balance_ratio', 1.0),
                    'computation_time_ms': result.get('computation_time_ms', 0)
                })
        
        # Sort by total time
        comparisons.sort(key=lambda x: x['total_time_min'])
        
        if comparisons:
            best = comparisons[0]
            return {
                'best_for_total_time': best['strategy'],
                'best_total_time_min': best['total_time_min'],
                'comparisons': comparisons
            }
        
        return {'error': 'No valid strategies'}


def print_vrp_report(results: Dict[str, Any]) -> None:
    """Pretty print VRP results"""
    print("\n" + "=" * 70)
    print("VRP OPTIMIZATION REPORT")
    print("=" * 70)
    
    if 'region' in results:
        r = results['region']
        print(f"\n[PIN] Region: {r['value']} ({r['column']})")
        print(f"   Orders: {r['total_orders']:,} | Locations: {r['unique_locations']}")
        depot = r.get('depot_location') or r.get('depot')
        if depot:
            print(f"   Depot: ({depot[0]:.4f}, {depot[1]:.4f})")
    
    if 'config' in results:
        c = results['config']
        print(f"\n[SETTINGS] Configuration:")
        print(f"   Vehicles: {c['num_vehicles']} | Speed: {c['speed_kmh']} km/h")
        print(f"   Max time/vehicle: {c['max_time_per_vehicle_min']} min | Service time: {c['service_time_per_stop_min']} min")
    
    print(f"\n[DATA] RESULTS BY STRATEGY:")
    print("-" * 70)
    
    for name, strat in results.get('strategies', {}).items():
        if strat.get('status') == 'success':
            print(f"\n   [TRUCK] {name.upper()}")
            print(f"      Total Distance: {strat['total_distance_km']:.2f} km")
            print(f"      Total Time: {strat['total_time_min']:.2f} min ({strat['total_time_min']/60:.1f} hours)")
            print(f"      Max Route Time: {strat['max_route_time_min']:.2f} min")
            print(f"      Time Balance: {strat.get('time_balance_ratio', 0):.2f}x")
            print(f"      Vehicles Used: {strat['vehicles_used']}/{results['config']['num_vehicles']}")
            print(f"      Computation: {strat.get('computation_time_ms', 0):.1f} ms")
            
            print(f"\n      Routes:")
            for route in strat['routes']:
                if route['num_stops'] > 0:
                    print(f"         V{route['vehicle_id']}: {route['num_stops']} stops | "
                          f"{route['distance_km']:.1f} km | {route['total_time_min']:.1f} min")
    
    if 'benchmark' in results:
        b = results['benchmark']
        print(f"\n[CHART] BENCHMARK:")
        print("-" * 70)
        print(f"   Best for Total Time: {b['best_for_total_time']} ({b['best_total_time_min']:.2f} min)")
        print(f"\n   Ranking by Total Time:")
        for i, c in enumerate(b['comparisons'], 1):
            marker = "[1st]" if i == 1 else "[2nd]" if i == 2 else "[3rd]"
            print(f"   {marker} {c['strategy']}: {c['total_time_min']:.1f} min | "
                  f"Max: {c['max_route_time_min']:.1f} min | "
                  f"Balance: {c['time_balance_ratio']:.2f}x")
    
    print("\n" + "=" * 70)
