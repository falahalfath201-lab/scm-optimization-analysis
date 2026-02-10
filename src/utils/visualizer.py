"""
Visualization Utilities
Standardized plotting functions for all analysis modules
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class Visualizer:
    """Centralized visualization utility"""
    
    def __init__(self, output_dir: str = 'outputs/figures'):
        """
        Initialize visualizer
        
        Args:
            output_dir: Base directory for saving figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_figure(self, filename: str, subdirectory: Optional[str] = None, 
                   dpi: int = 300, bbox_inches: str = 'tight'):
        """Save current figure"""
        if subdirectory:
            save_path = self.output_dir / subdirectory / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            save_path = self.output_dir / filename
        
        plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
        print(f"   Saved: {save_path}")
        plt.close()
    
    def plot_cost_comparison(self, scenarios: Dict[str, float], 
                            title: str = "Cost Comparison",
                            ylabel: str = "Total Cost ($)",
                            highlight_best: bool = True,
                            filename: str = "cost_comparison.png",
                            subdirectory: Optional[str] = None):
        """Plot cost comparison bar chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = list(scenarios.keys())
        values = list(scenarios.values())
        
        # Highlight best (lowest)
        colors = ['green' if v == min(values) and highlight_best else 'steelblue' 
                 for v in values]
        
        bars = ax.bar(names, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height:,.0f}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        self.save_figure(filename, subdirectory)
    
    def plot_shipping_mode_distribution(self, mode_counts: Dict[str, int],
                                       title: str = "Shipping Mode Distribution",
                                       filename: str = "shipping_mode_dist.png",
                                       subdirectory: Optional[str] = None):
        """Plot shipping mode distribution pie chart"""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        modes = list(mode_counts.keys())
        counts = list(mode_counts.values())
        colors = sns.color_palette("Set2", len(modes))
        
        wedges, texts, autotexts = ax.pie(counts, labels=modes, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        
        # Style
        for text in texts:
            text.set_fontsize(11)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        self.save_figure(filename, subdirectory)
    
    def plot_delivery_time_improvement(self, baseline: Dict[str, float],
                                       optimized: Dict[str, float],
                                       title: str = "Delivery Time Improvement",
                                       filename: str = "delivery_time_improvement.png",
                                       subdirectory: Optional[str] = None):
        """Plot before/after delivery time comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = list(baseline.keys())
        x = np.arange(len(metrics))
        width = 0.35
        
        baseline_vals = list(baseline.values())
        optimized_vals = list(optimized.values())
        
        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline',
                      color='coral', alpha=0.8)
        bars2 = ax.bar(x + width/2, optimized_vals, width, label='Optimized',
                      color='seagreen', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Days')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        self.save_figure(filename, subdirectory)
    
    def plot_risk_heatmap(self, risk_matrix: pd.DataFrame,
                         title: str = "Risk Heatmap",
                         filename: str = "risk_heatmap.png",
                         subdirectory: Optional[str] = None):
        """Plot risk heatmap"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(risk_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r',
                   cbar_kws={'label': 'Risk Score'}, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        self.save_figure(filename, subdirectory)
    
    def plot_warehouse_locations(self, locations: List[Tuple[float, float]],
                                labels: Optional[List[str]] = None,
                                customers: Optional[List[Tuple[float, float]]] = None,
                                title: str = "Warehouse Locations",
                                filename: str = "warehouse_locations.png",
                                subdirectory: Optional[str] = None):
        """Plot warehouse locations on map"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot customers if provided
        if customers:
            cust_lats, cust_lons = zip(*customers)
            ax.scatter(cust_lons, cust_lats, c='lightblue', s=10, alpha=0.5,
                      label='Customers', marker='.')
        
        # Plot warehouses
        if locations:
            wh_lats, wh_lons = zip(*locations)
            ax.scatter(wh_lons, wh_lats, c='red', s=200, alpha=0.8,
                      label='Warehouses', marker='^', edgecolors='black', linewidths=2)
            
            # Add labels
            if labels:
                for i, (lat, lon) in enumerate(locations):
                    if i < len(labels):
                        ax.annotate(labels[i], (lon, lat), xytext=(5, 5),
                                  textcoords='offset points', fontsize=9,
                                  fontweight='bold')
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        self.save_figure(filename, subdirectory)
    
    def plot_feature_importance(self, importance_dict: Dict[str, float],
                               top_n: int = 10,
                               title: str = "Feature Importance",
                               filename: str = "feature_importance.png",
                               subdirectory: Optional[str] = None):
        """Plot feature importance horizontal bar chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort and get top N
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importances = zip(*sorted_items)
        
        # Reverse for better display
        features = features[::-1]
        importances = importances[::-1]
        
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importances, color='steelblue', alpha=0.8)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.3f}',
                   ha='left', va='center', fontsize=8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        self.save_figure(filename, subdirectory)
    
    def plot_time_series(self, df: pd.DataFrame, 
                        date_col: str, value_col: str,
                        title: str = "Time Series",
                        ylabel: str = "Value",
                        filename: str = "time_series.png",
                        subdirectory: Optional[str] = None):
        """Plot time series"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(df[date_col], df[value_col], color='steelblue', linewidth=2)
        ax.fill_between(df[date_col], df[value_col], alpha=0.3, color='steelblue')
        
        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        self.save_figure(filename, subdirectory)
    
    def plot_metric_comparison_table(self, data: Dict[str, Dict[str, float]],
                                    title: str = "Metrics Comparison",
                                    filename: str = "metrics_table.png",
                                    subdirectory: Optional[str] = None):
        """Plot comparison table as image"""
        df = pd.DataFrame(data).T
        
        fig, ax = plt.subplots(figsize=(12, len(df) * 0.5 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values,
                        rowLabels=df.index,
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        self.save_figure(filename, subdirectory)
    
    def plot_route_map(self, routes: List[Dict[str, Any]],
                      title: str = "Route Optimization",
                      filename: str = "route_map.png",
                      subdirectory: Optional[str] = None):
        """Plot multiple routes on map"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = sns.color_palette("husl", len(routes))
        
        for i, route in enumerate(routes):
            if 'stops' in route:
                stops = route['stops']
                lats, lons = zip(*stops)
                
                # Plot route line
                ax.plot(lons, lats, 'o-', color=colors[i], linewidth=2,
                       markersize=8, label=f"Route {i+1}", alpha=0.7)
                
                # Highlight depot (first stop)
                ax.plot(lons[0], lats[0], '*', color=colors[i], markersize=20,
                       markeredgecolor='black', markeredgewidth=1.5)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        self.save_figure(filename, subdirectory)
