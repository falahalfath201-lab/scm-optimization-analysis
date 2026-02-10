"""
Distance calculation utilities for SCM optimization
Supports Haversine and external routing API calculations
"""

import math
import numpy as np
from typing import Tuple, List, Optional, Union
import logging

from ..config import EARTH_RADIUS_KM

logger = logging.getLogger(__name__)

class DistanceCalculator:
    """
    Calculate distances between geographical points
    """

    def __init__(self):
        """Initialize distance calculator"""
        self.earth_radius = EARTH_RADIUS_KM

    def haversine_distance(self, point1: Tuple[float, float],
                          point2: Tuple[float, float]) -> float:
        """
        Calculate haversine distance between two points

        Args:
            point1: (latitude, longitude) of first point
            point2: (latitude, longitude) of second point

        Returns:
            Distance in kilometers
        """
        lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        distance = self.earth_radius * c
        return distance
    
    def haversine(self, lat1: float, lon1: float, 
                  lat2: float, lon2: float) -> float:
        """
        Calculate haversine distance between two points (individual coords)
        
        Args:
            lat1: Latitude of first point
            lon1: Longitude of first point
            lat2: Latitude of second point
            lon2: Longitude of second point
            
        Returns:
            Distance in kilometers
        """
        return self.haversine_distance((lat1, lon1), (lat2, lon2))
    
    def haversine_vectorized(self, lat1: Union[float, np.ndarray], 
                            lon1: Union[float, np.ndarray],
                            lat2: Union[float, np.ndarray], 
                            lon2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Vectorized haversine distance calculation (works with numpy arrays)
        
        Args:
            lat1: Latitude(s) of first point(s)
            lon1: Longitude(s) of first point(s)
            lat2: Latitude(s) of second point(s)
            lon2: Longitude(s) of second point(s)
            
        Returns:
            Distance(s) in kilometers (scalar or array)
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        
        distance = self.earth_radius * c
        return distance

    def calculate_distance_matrix(self, locations: List[Tuple[float, float]]) -> List[List[float]]:
        """
        Calculate distance matrix for all locations

        Args:
            locations: List of (lat, lon) coordinates

        Returns:
            2D distance matrix in km
        """
        n = len(locations)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                dist = self.haversine_distance(locations[i], locations[j])
                matrix[i][j] = dist
                matrix[j][i] = dist

        return matrix