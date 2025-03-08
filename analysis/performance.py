"""
Performance analysis for PoS networks.
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from models.metrics import MetricCategory, get_metrics_by_category
from models.network import NetworkComparison, PerformanceMetrics

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Analyzer for network performance metrics."""

    def __init__(self, storage_client=None):
        """
        Initialize the performance analyzer.
        
        Args:
            storage_client: Client for accessing stored network data
        """
        self.storage_client = storage_client
        self.performance_metrics = get_metrics_by_category(MetricCategory.PERFORMANCE)

    def load_data(self, 
                 network_ids: List[str], 
                 start_time: datetime, 
                 end_time: datetime) -> Dict[str, pd.DataFrame]:
        """
        Load performance data for specified networks and time range.
        
        Args:
            network_ids: List of network IDs to analyze
            start_time: Start of the analysis period
            end_time: End of the analysis period
            
        Returns:
            Dictionary of DataFrames with performance data for each network
        """
        if not self.storage_client:
            logger.error("No storage client configured")
            return {}
            
        network_data = {}
        
        for network_id in network_ids:
            try:
                # Fetch data from storage
                raw_data = self.storage_client.get_metrics(
                    network_id=network_id,
                    metric_category=MetricCategory.PERFORMANCE.value,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if not raw_data:
                    logger.warning(f"No performance data found for network {network_id}")
                    continue
                    
                # Convert to DataFrame for easier analysis
                df = pd.DataFrame(raw_data)
                
                # Ensure timestamp column is datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    
                network_data[network_id] = df
                logger.info(f"Loaded {len(df)} performance records for {network_id}")
                
            except Exception as e:
                logger.error(f"Error loading data for network {network_id}: {str(e)}")
                
        return network_data

    def calculate_summary_stats(self, 
                               network_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Calculate summary statistics for performance metrics.
        
        Args:
            network_data: Dictionary of network performance DataFrames
            
        Returns:
            Dictionary of summary statistics for each network
        """
        summary_stats = {}
        
        for network_id, df in network_data.items():
            if df.empty:
                continue
                
            network_stats = {}
            
            # Calculate basic statistics for numerical columns
            for metric in self.performance_metrics:
                if metric.id in df.columns:
                    series = df[metric.id].dropna()
                    if not series.empty:
                        network_stats[f"{metric.id}_mean"] = series.mean()
                        network_stats[f"{metric.id}_median"] = series.median()
                        network_stats[f"{metric.id}_min"] = series.min()
                        network_stats[f"{metric.id}_max"] = series.max()
                        network_stats[f"{metric.id}_std"] = series.std()
                        network_stats[f"{metric.id}_p95"] = series.quantile(0.95)
                        
            # Calculate TPS stability (coefficient of variation)
            if 'tps' in df.columns:
                tps_series = df['tps'].dropna()
                if not tps_series.empty and tps_series.mean() > 0:
                    network_stats['tps_variation'] = tps_series.std() / tps_series.mean()
                    
            # Calculate block time consistency
            if 'block_time' in df.columns:
                block_time_series = df['block_time'].dropna()
                if not block_time_series.empty:
                    # Jitter: standard deviation of differences between consecutive block times
                    block_time_diffs = block_time_series.diff().dropna()
                    network_stats['block_time_jitter'] = block_time_diffs.std()
                    
            # Calculate correlation between utilization and latency
            if 'network_utilization' in df.columns and 'latency_ms' in df.columns:
                corr = df['network_utilization'].corr(df['latency_ms'])
                if not np.isnan(corr):
                    network_stats['utilization_latency_correlation'] = corr
                    
            summary_stats[network_id] = network_stats
            
        return summary_stats

    def detect_performance_anomalies(self, 
                                    network_data: Dict[str, pd.DataFrame], 
                                    z_threshold: float = 3.0) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect anomalies in performance metrics using Z-score method.
        
        Args:
            network_data: Dictionary of network performance DataFrames
            z_threshold: Z-score threshold for anomaly detection
            
        Returns:
            Dictionary of detected anomalies for each network
        """
        anomalies = {}
        
        for network_id, df in network_data.items():
            if df.empty:
                continue
                
            network_anomalies = []
            
            # Reset index to access timestamp column
            df_reset = df.reset_index()
            
            for metric in self.performance_metrics:
                if metric.id in df.columns:
                    series = df[metric.id].dropna()
                    if len(series) < 10:  # Need enough data for meaningful statistics
                        continue
                        
                    # Calculate Z-scores
                    mean = series.mean()
                    std = series.std()
                    if std == 0:  # Avoid division by zero
                        continue
                        
                    z_scores = (series - mean) / std
                    
                    # Find anomalies
                    if metric.is_higher_better:
                        # For metrics where higher is better, low values are anomalies
                        anomaly_mask = z_scores < -z_threshold
                    else:
                        # For metrics where lower is better, high values are anomalies
                        anomaly_mask = z_scores > z_threshold
                        
                    # Extract anomalies with timestamps
                    if anomaly_mask.any():
                        anomaly_indices = series[anomaly_mask].index
                        for idx in anomaly_indices:
                            timestamp = df_reset.loc[df_reset.index[df.index.get_loc(idx)], 'timestamp']
                            anomaly = {
                                'timestamp': timestamp,
                                'metric': metric.id,
                                'value': series[idx],
                                'z_score': z_scores[idx],
                                'expected': mean,
                                'deviation': abs(series[idx] - mean)
                            }
                            network_anomalies.append(anomaly)
                            
            anomalies[network_id] = network_anomalies
            logger.info(f"Detected {len(network_anomalies)} performance anomalies for {network_id}")
            
        return anomalies

    def compare_networks(self, 
                        network_data: Dict[str, pd.DataFrame],
                        metrics_to_compare: Optional[List[str]] = None) -> NetworkComparison:
        """
        Compare performance metrics across networks.
        
        Args:
            network_data: Dictionary of network performance DataFrames
            metrics_to_compare: List of specific metrics to compare (default: all available)
            
        Returns:
            NetworkComparison object with comparison results
        """
        if not network_data:
            logger.warning("No data available for network comparison")
            return None
            
        # Determine common time period
        common_start = max(df.index.min() for df in network_data.values() if not df.empty)
        common_end = min(df.index.max() for df in network_data.values() if not df.empty)
        
        # Determine metrics to compare
        if not metrics_to_compare:
            # Find metrics available across all networks
            all_metrics = set()
            for df in network_data.values():
                if not df.empty:
                    all_metrics.update(set(df.columns))
                    
            metrics_to_compare = [m.id for m in self.performance_metrics if m.id in all_metrics]
            
        # Calculate average metrics for each network
        comparison_metrics = {}
        
        for network_id, df in network_data.items():
            if df.empty:
                continue
                
            # Filter to common time period
            filtered_df = df[(df.index >= common_start) & (df.index <= common_end)]
            
            network_metrics = {}
            for metric_id in metrics_to_compare:
                if metric_id in filtered_df.columns:
                    series = filtered_df[metric_id].dropna()
                    if not series.empty:
                        network_metrics[metric_id] = series.mean()
                        
            comparison_metrics[network_id] = network_metrics
            
        # Create comparison object
        comparison = NetworkComparison(
            timestamp=datetime.utcnow(),
            networks=list(network_data.keys()),
            metrics=comparison_metrics,
            period_start=common_start,
            period_end=common_end,
            comparison_type=MetricCategory.PERFORMANCE.value
        )
        
        return comparison

    def analyze_trends(self, 
                      network_data: Dict[str, pd.DataFrame],
                      window: str = '1D') -> Dict[str, Dict[str, Any]]:
        """
        Analyze performance trends over time.
        
        Args:
            network_data: Dictionary of network performance DataFrames
            window: Time window for trend analysis (e.g., '1D' for daily)
            
        Returns:
            Dictionary of trend analysis results
        """
        trends = {}
        
        for network_id, df in network_data.items():
            if df.empty:
                continue
                
            network_trends = {}
            
            # Resample data to specified window
            resampled = df.resample(window).mean()
            
            for metric in self.performance_metrics:
                if metric.id in resampled.columns:
                    series = resampled[metric.id].dropna()
                    if len(series) < 3:  # Need enough data points for trend analysis
                        continue
                        
                    # Calculate linear regression
                    x = np.arange(len(series))
                    y = series.values
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    # Determine trend direction and strength
                    trend_direction = "improving" if (slope > 0) == metric.is_higher_better else "degrading"
                    trend_strength = abs(r_value)  # Correlation coefficient as strength indicator
                    
                    metric_trend = {
                        'metric': metric.id,
                        'slope': slope,
                        'r_squared': r_value ** 2,
                        'p_value': p_value,
                        'direction': trend_direction,
                        'strength': trend_strength,
                        'is_significant': p_value < 0.05,
                        'period_change': (series.iloc[-1] - series.iloc[0]) / series.iloc[0] if series.iloc[0] != 0 else 0
                    }
                    
                    network_trends[metric.id] = metric_trend
                    
            trends[network_id] = network_trends
            
        return trends

    def calculate_performance_score(self, 
                                  network_data: Dict[str, pd.DataFrame],
                                  weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate a composite performance score for each network.
        
        Args:
            network_data: Dictionary of network performance DataFrames
            weights: Optional dictionary of metric weights (default: equal weights)
            
        Returns:
            Dictionary of performance scores for each network
        """
        scores = {}
        
        # Define default weights if not provided
        if not weights:
            metrics = [m.id for m in self.performance_metrics]
            weights = {metric: 1.0 / len(metrics) for metric in metrics}
            
        for network_id, df in network_data.items():
            if df.empty:
                continue
                
            # Calculate average for each metric
            metric_values = {}
            for metric_id, weight in weights.items():
                if metric_id in df.columns:
                    series = df[metric_id].dropna()
                    if not series.empty:
                        metric_values[metric_id] = series.mean()
                        
            # Skip if no metrics available
            if not metric_values:
                continue
                
            # Normalize metrics to 0-1 scale
            normalized_values = {}
            for metric in self.performance_metrics:
                if metric.id in metric_values:
                    value = metric_values[metric.id]
                    
                    # Get all values across networks for this metric for normalization
                    all_network_values = []
                    for net_df in network_data.values():
                        if not net_df.empty and metric.id in net_df.columns:
                            series = net_df[metric.id].dropna()
                            if not series.empty:
                                all_network_values.append(series.mean())
                                
                    if all_network_values:
                        min_val = min(all_network_values)
                        max_val = max(all_network_values)
                        
                        if min_val == max_val:
                            normalized = 1.0  # All networks have the same value
                        else:
                            # Normalize based on whether higher is better
                            if metric.is_higher_better:
                                normalized = (value - min_val) / (max_val - min_val)
                            else:
                                normalized = (max_val - value) / (max_val - min_val)
                                
                        normalized_values[metric.id] = normalized
                        
            # Calculate weighted score
            score = 0.0
            total_weight = 0.0
            
            for metric_id, normalized in normalized_values.items():
                weight = weights.get(metric_id, 0.0)
                score += normalized * weight
                total_weight += weight
                
            if total_weight > 0:
                final_score = score / total_weight
                scores[network_id] = final_score
                
        return scores