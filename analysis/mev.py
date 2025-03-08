"""
MEV (Maximal Extractable Value) analysis for PoS networks.
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.metrics import MetricCategory, get_metrics_by_category
from models.network import MEVMetrics, NetworkComparison

logger = logging.getLogger(__name__)


class MEVAnalyzer:
    """Analyzer for MEV metrics in PoS networks."""

    def __init__(self, storage_client=None):
        """
        Initialize the MEV analyzer.
        
        Args:
            storage_client: Client for accessing stored network data
        """
        self.storage_client = storage_client
        self.mev_metrics = get_metrics_by_category(MetricCategory.MEV)

    def load_data(self, 
                 network_ids: List[str], 
                 start_time: datetime, 
                 end_time: datetime) -> Dict[str, pd.DataFrame]:
        """
        Load MEV data for specified networks and time range.
        
        Args:
            network_ids: List of network IDs to analyze
            start_time: Start of the analysis period
            end_time: End of the analysis period
            
        Returns:
            Dictionary of DataFrames with MEV data for each network
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
                    metric_category=MetricCategory.MEV.value,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if not raw_data:
                    logger.warning(f"No MEV data found for network {network_id}")
                    continue
                    
                # Convert to DataFrame for easier analysis
                df = pd.DataFrame(raw_data)
                
                # Ensure timestamp column is datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    
                network_data[network_id] = df
                logger.info(f"Loaded {len(df)} MEV records for {network_id}")
                
            except Exception as e:
                logger.error(f"Error loading MEV data for network {network_id}: {str(e)}")
                
        return network_data

    def categorize_mev_types(self, network_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Categorize MEV by type for each network.
        
        Args:
            network_data: Dictionary of network MEV DataFrames
            
        Returns:
            Dictionary of MEV categorization for each network
        """
        categorization = {}
        
        for network_id, df in network_data.items():
            if df.empty:
                continue
                
            # Look for relevant MEV metrics
            mev_types = {}
            
            # Sandwich attacks
            if 'sandwich_attacks' in df.columns:
                mev_types['Sandwich Attacks'] = df['sandwich_attacks'].sum()
                
            # Frontrunning
            if 'frontrunning_instances' in df.columns:
                mev_types['Frontrunning'] = df['frontrunning_instances'].sum()
                
            # Backrunning
            if 'backrunning_instances' in df.columns:
                mev_types['Backrunning'] = df['backrunning_instances'].sum()
                
            # Arbitrage
            if 'arbitrage_instances' in df.columns:
                mev_types['Arbitrage'] = df['arbitrage_instances'].sum()
            
            # If we have total MEV extracted but missing some types,
            # categorize the remainder as "Other"
            if 'mev_extracted' in df.columns:
                total_mev = df['mev_extracted'].sum()
                categorized_sum = sum(mev_types.values())
                
                if total_mev > categorized_sum:
                    mev_types['Other'] = total_mev - categorized_sum
            
            categorization[network_id] = mev_types
            
        return categorization

    def analyze_mev_impact(self, 
                          network_data: Dict[str, pd.DataFrame],
                          reward_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Dict[str, float]]:
        """
        Analyze the impact of MEV on validator rewards.
        
        Args:
            network_data: Dictionary of network MEV DataFrames
            reward_data: Optional dictionary of validator reward DataFrames
            
        Returns:
            Dictionary of MEV impact metrics for each network
        """
        impact_metrics = {}
        
        for network_id, df in network_data.items():
            if df.empty:
                continue
                
            network_impact = {}
            
            # Calculate average MEV per block
            if 'mev_extracted' in df.columns and 'blocks_with_mev' in df.columns:
                mev_blocks = df['blocks_with_mev'].sum()
                
                if mev_blocks > 0:
                    network_impact['avg_mev_per_block'] = df['mev_extracted'].sum() / mev_blocks
                    
            # Calculate percentage of blocks with MEV
            if 'blocks_with_mev' in df.columns and 'total_blocks' in df.columns:
                total_blocks = df['total_blocks'].sum()
                
                if total_blocks > 0:
                    network_impact['pct_blocks_with_mev'] = (df['blocks_with_mev'].sum() / total_blocks) * 100
                    
            # Calculate MEV to rewards ratio
            if reward_data and network_id in reward_data and not reward_data[network_id].empty:
                reward_df = reward_data[network_id]
                
                if 'validator_rewards' in reward_df.columns and 'mev_extracted' in df.columns:
                    total_rewards = reward_df['validator_rewards'].sum()
                    total_mev = df['mev_extracted'].sum()
                    
                    if total_rewards > 0:
                        network_impact['mev_to_rewards_ratio'] = total_mev / total_rewards
            
            # Calculate MEV concentration (if we have extractor data)
            if 'top_extractors' in df.columns:
                try:
                    # Assuming top_extractors is stored as a JSON string
                    extractors_data = []
                    for extractors_str in df['top_extractors'].dropna():
                        if isinstance(extractors_str, str):
                            import json
                            extractors = json.loads(extractors_str)
                            extractors_data.extend(extractors)
                    
                    if extractors_data:
                        # Calculate Gini coefficient for MEV extraction
                        extractors_df = pd.DataFrame(extractors_data)
                        if 'value' in extractors_df.columns:
                            values = extractors_df['value'].values
                            values = np.sort(values)
                            n = len(values)
                            
                            if n > 0 and np.sum(values) > 0:
                                index = np.arange(1, n + 1)
                                gini = (2 * np.sum(index * values) / (n * np.sum(values))) - (n + 1) / n
                                network_impact['mev_gini_coefficient'] = gini
                except Exception as e:
                    logger.error(f"Error calculating MEV concentration: {str(e)}")
            
            impact_metrics[network_id] = network_impact
            
        return impact_metrics

    def estimate_user_impact(self, network_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Estimate the impact of MEV on users for each network.
        
        Args:
            network_data: Dictionary of network MEV DataFrames
            
        Returns:
            Dictionary of user impact metrics for each network
        """
        user_impact = {}
        
        for network_id, df in network_data.items():
            if df.empty:
                continue
                
            network_impact = {}
            
            # Estimate price impact from sandwich attacks
            if 'sandwich_attacks' in df.columns and 'sandwich_price_impact' in df.columns:
                # Average price impact across all sandwich attacks
                attacks = df['sandwich_attacks'].sum()
                
                if attacks > 0:
                    avg_price_impact = df['sandwich_price_impact'].mean()
                    network_impact['avg_sandwich_price_impact_pct'] = avg_price_impact * 100
                    
            # Estimate total value extracted from users
            if 'mev_extracted' in df.columns:
                network_impact['total_value_extracted'] = df['mev_extracted'].sum()
                
                # If we have transaction count, calculate per-transaction impact
                if 'tx_count' in df.columns:
                    tx_count = df['tx_count'].sum()
                    
                    if tx_count > 0:
                        network_impact['mev_per_transaction'] = df['mev_extracted'].sum() / tx_count
            
            # Estimate user slipage due to MEV
            if 'user_slippage' in df.columns:
                network_impact['avg_user_slippage_pct'] = df['user_slippage'].mean() * 100
                
            user_impact[network_id] = network_impact
            
        return user_impact

    def compare_networks(self, 
                        network_data: Dict[str, pd.DataFrame],
                        metrics_to_compare: Optional[List[str]] = None) -> NetworkComparison:
        """
        Compare MEV metrics across networks.
        
        Args:
            network_data: Dictionary of network MEV DataFrames
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
                    
            metrics_to_compare = [m.id for m in self.mev_metrics if m.id in all_metrics]
            
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
            comparison_type=MetricCategory.MEV.value
        )
        
        return comparison

    def analyze_trends(self, 
                      network_data: Dict[str, pd.DataFrame],
                      window: str = '1D') -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Analyze MEV trends over time.
        
        Args:
            network_data: Dictionary of network MEV DataFrames
            window: Time window for trend analysis (e.g., '1D' for daily)
            
        Returns:
            Dictionary of trend analysis results
        """
        trends = {}
        
        for network_id, df in network_data.items():
            if df.empty:
                continue
                
            # Resample data to specified window
            resampled = df.resample(window).mean()
            
            metric_trends = {}
            for metric in self.mev_metrics:
                if metric.id in resampled.columns:
                    series = resampled[metric.id].dropna()
                    if len(series) < 3:  # Need enough data points for trend analysis
                        continue
                        
                    # Calculate linear regression
                    x = np.arange(len(series))
                    y = series.values
                    
                    # Simple linear regression
                    if len(x) > 1:  # Need at least 2 points for regression
                        A = np.vstack([x, np.ones(len(x))]).T
                        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
                        
                        # Calculate trend metrics
                        trend_info = {
                            'slope': slope,
                            'intercept': intercept,
                            'direction': 'increasing' if slope > 0 else 'decreasing',
                            'pct_change': (series.iloc[-1] / series.iloc[0] - 1) * 100 if series.iloc[0] != 0 else float('inf'),
                            'start_value': series.iloc[0],
                            'end_value': series.iloc[-1],
                            'min_value': series.min(),
                            'max_value': series.max(),
                        }
                        
                        metric_trends[metric.id] = trend_info
            
            trends[network_id] = metric_trends
            
        return trends

    def detect_mev_anomalies(self, 
                            network_data: Dict[str, pd.DataFrame], 
                            z_threshold: float = 3.0) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect anomalies in MEV activity using Z-score method.
        
        Args:
            network_data: Dictionary of network MEV DataFrames
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
            
            for metric in self.mev_metrics:
                if metric.id in df.columns:
                    series = df[metric.id].dropna()
                    if len(series) < 10:  # Need enough data for meaningful statistics
                        continue
                        
                    # Calculate rolling mean and standard deviation
                    window_size = min(30, len(series))  # Use up to 30 days for rolling window
                    rolling_mean = series.rolling(window=window_size, min_periods=5).mean()
                    rolling_std = series.rolling(window=window_size, min_periods=5).std()
                    
                    # Calculate Z-scores
                    z_scores = (series - rolling_mean) / rolling_std
                    
                    # Find anomalies (ignore NaN values from the beginning of the rolling window)
                    anomaly_mask = (z_scores > z_threshold) & ~np.isnan(z_scores)
                    
                    # Extract anomalies with timestamps
                    if anomaly_mask.any():
                        anomaly_indices = series[anomaly_mask].index
                        for idx in anomaly_indices:
                            row_idx = df.index.get_loc(idx)
                            timestamp = df_reset.loc[row_idx, 'timestamp']
                            z_score_val = z_scores[idx]
                            
                            anomaly = {
                                'timestamp': timestamp,
                                'metric': metric.id,
                                'value': series[idx],
                                'z_score': z_score_val,
                                'expected': rolling_mean[idx],
                                'deviation': abs(series[idx] - rolling_mean[idx]),
                                'percent_deviation': abs(series[idx] / rolling_mean[idx] - 1) * 100 if rolling_mean[idx] != 0 else float('inf')
                            }
                            network_anomalies.append(anomaly)
                            
            anomalies[network_id] = network_anomalies
            logger.info(f"Detected {len(network_anomalies)} MEV anomalies for {network_id}")
            
        return anomalies