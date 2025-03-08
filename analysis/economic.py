"""
Economic security analysis for PoS networks.
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from models.metrics import MetricCategory, get_metrics_by_category
from models.network import EconomicMetrics, NetworkComparison

logger = logging.getLogger(__name__)


class EconomicAnalyzer:
    """Analyzer for economic security of PoS networks."""

    def __init__(self, storage_client=None):
        """
        Initialize the economic analyzer.
        
        Args:
            storage_client: Client for accessing stored network data
        """
        self.storage_client = storage_client
        self.economic_metrics = get_metrics_by_category(MetricCategory.ECONOMIC)
        self.security_metrics = get_metrics_by_category(MetricCategory.SECURITY)

    def load_data(self, 
                network_ids: List[str], 
                start_time: datetime, 
                end_time: datetime) -> Dict[str, pd.DataFrame]:
        """
        Load economic data for specified networks and time range.
        
        Args:
            network_ids: List of network IDs to analyze
            start_time: Start of the analysis period
            end_time: End of the analysis period
            
        Returns:
            Dictionary of DataFrames with economic data for each network
        """
        if not self.storage_client:
            logger.error("No storage client configured")
            return {}
            
        network_data = {}
        
        for network_id in network_ids:
            try:
                # Fetch economic metrics
                economic_data = self.storage_client.get_metrics(
                    network_id=network_id,
                    metric_category=MetricCategory.ECONOMIC.value,
                    start_time=start_time,
                    end_time=end_time
                )
                
                # Fetch security metrics
                security_data = self.storage_client.get_metrics(
                    network_id=network_id,
                    metric_category=MetricCategory.SECURITY.value,
                    start_time=start_time,
                    end_time=end_time
                )
                
                # Combine data
                combined_data = economic_data + security_data
                
                if not combined_data:
                    logger.warning(f"No economic data found for network {network_id}")
                    continue
                    
                # Convert to DataFrame for easier analysis
                df = pd.DataFrame(combined_data)
                
                # Pivot to get metrics as columns
                pivoted = df.pivot(index='timestamp', columns='metric_id', values='value')
                
                # Ensure timestamp column is datetime
                pivoted.index = pd.to_datetime(pivoted.index)
                
                network_data[network_id] = pivoted
                logger.info(f"Loaded {len(pivoted)} economic records for {network_id}")
                
            except Exception as e:
                logger.error(f"Error loading economic data for network {network_id}: {str(e)}")
                
        return network_data

    def calculate_security_thresholds(self, network_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Calculate economic security thresholds for each network.
        
        Args:
            network_data: Dictionary of network economic DataFrames
            
        Returns:
            Dictionary of security thresholds for each network
        """
        thresholds = {}
        
        for network_id, df in network_data.items():
            if df.empty:
                continue
                
            network_thresholds = {}
            
            # Calculate cost to attack (33% attack threshold for PoS networks)
            if 'total_staked' in df.columns and 'token_price_usd' in df.columns:
                # Average values over the period
                avg_total_staked = df['total_staked'].mean()
                avg_price = df['token_price_usd'].mean()
                
                # Cost to acquire 33% of stake
                attack_stake_pct = 0.33
                cost_to_attack = avg_total_staked * attack_stake_pct * avg_price
                network_thresholds['cost_to_attack_usd'] = cost_to_attack
                network_thresholds['attack_stake_pct'] = attack_stake_pct * 100
                
            # Nakamoto coefficient (minimum entities to control 33%)
            if 'nakamoto_coefficient' in df.columns:
                network_thresholds['nakamoto_coefficient'] = df['nakamoto_coefficient'].mean()
                
            # Delegation concentration (Gini coefficient)
            if 'delegation_concentration' in df.columns:
                network_thresholds['delegation_concentration'] = df['delegation_concentration'].mean()
                
            # Validator count
            if 'validator_count' in df.columns:
                network_thresholds['validator_count'] = df['validator_count'].mean()
                
            # Staking ratio
            if 'staking_ratio' in df.columns:
                network_thresholds['staking_ratio'] = df['staking_ratio'].mean() * 100  # Convert to percentage
                
            # Estimate required market cap to maintain security
            if 'cost_to_attack_usd' in network_thresholds:
                # Heuristic: market cap should be at least 10x the cost to attack
                network_thresholds['min_secure_market_cap'] = network_thresholds['cost_to_attack_usd'] * 10
                
            thresholds[network_id] = network_thresholds
            
        return thresholds

    def analyze_stake_distribution(self, 
                                 network_ids: List[str], 
                                 timestamp: Optional[datetime] = None) -> Dict[str, Dict[str, Any]]:
        """
        Analyze stake distribution across validators.
        
        Args:
            network_ids: List of network IDs to analyze
            timestamp: Optional specific timestamp for analysis
            
        Returns:
            Dictionary of stake distribution analysis for each network
        """
        if not self.storage_client:
            logger.error("No storage client configured")
            return {}
            
        distribution_analysis = {}
        
        for network_id in network_ids:
            try:
                # Get validator metrics
                validators = self.storage_client.get_validator_metrics(
                    network_id=network_id,
                    start_time=timestamp - timedelta(hours=1) if timestamp else None,
                    end_time=timestamp + timedelta(hours=1) if timestamp else None,
                    limit=1000  # Get a large number of validators
                )
                
                if not validators:
                    logger.warning(f"No validator data found for network {network_id}")
                    continue
                
                # Extract stake amounts
                stakes = []
                for validator in validators:
                    validator_metrics = validator.get('metrics', {})
                    stake = validator_metrics.get('stake_amount') or validator_metrics.get('voting_power')
                    if stake:
                        stakes.append(stake)
                
                if not stakes:
                    logger.warning(f"No stake data found for network {network_id}")
                    continue
                
                # Sort stakes
                stakes = sorted(stakes, reverse=True)
                total_stake = sum(stakes)
                
                # Calculate distribution metrics
                stake_analysis = {
                    'total_validators': len(stakes),
                    'total_stake': total_stake,
                    'max_stake': max(stakes),
                    'min_stake': min(stakes),
                    'avg_stake': np.mean(stakes),
                    'median_stake': np.median(stakes),
                }
                
                # Calculate stake concentration
                if total_stake > 0:
                    # Percentage of stake held by top validators
                    stake_analysis['top_1_stake_pct'] = stakes[0] / total_stake * 100 if len(stakes) >= 1 else 0
                    stake_analysis['top_5_stake_pct'] = sum(stakes[:5]) / total_stake * 100 if len(stakes) >= 5 else 0
                    stake_analysis['top_10_stake_pct'] = sum(stakes[:10]) / total_stake * 100 if len(stakes) >= 10 else 0
                    stake_analysis['top_20_stake_pct'] = sum(stakes[:20]) / total_stake * 100 if len(stakes) >= 20 else 0
                    
                    # Calculate Gini coefficient
                    stakes_array = np.array(stakes)
                    n = len(stakes_array)
                    stakes_sorted = np.sort(stakes_array)
                    index = np.arange(1, n + 1)
                    gini = (2 * np.sum(index * stakes_sorted) / (n * np.sum(stakes_sorted))) - (n + 1) / n
                    stake_analysis['gini_coefficient'] = gini
                    
                    # Calculate Nakamoto coefficient (minimum validators for 33% control)
                    cumulative_stake = 0
                    for i, stake in enumerate(stakes):
                        cumulative_stake += stake
                        if cumulative_stake / total_stake >= 0.33:
                            stake_analysis['nakamoto_coefficient'] = i + 1
                            break
                    
                distribution_analysis[network_id] = stake_analysis
                
            except Exception as e:
                logger.error(f"Error analyzing stake distribution for {network_id}: {str(e)}")
                
        return distribution_analysis

    def calculate_reward_metrics(self, network_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Calculate staking reward metrics for each network.
        
        Args:
            network_data: Dictionary of network economic DataFrames
            
        Returns:
            Dictionary of reward metrics for each network
        """
        reward_metrics = {}
        
        for network_id, df in network_data.items():
            if df.empty:
                continue
                
            network_rewards = {}
            
            # Staking APR/APY
            if 'staking_rewards_apr' in df.columns:
                network_rewards['apr'] = df['staking_rewards_apr'].mean() * 100  # Convert to percentage
                network_rewards['apy'] = ((1 + df['staking_rewards_apr'].mean() / 365) ** 365 - 1) * 100  # Convert to percentage
                
                # APR volatility
                network_rewards['apr_volatility'] = df['staking_rewards_apr'].std() * 100  # Standard deviation as percentage
                
            # Reward-to-risk ratio
            if 'staking_rewards_apr' in df.columns and 'token_price_usd' in df.columns:
                # Calculate price volatility
                price_returns = df['token_price_usd'].pct_change().dropna()
                price_volatility = price_returns.std() * np.sqrt(365)  # Annualized volatility
                
                if price_volatility > 0:
                    reward_risk_ratio = df['staking_rewards_apr'].mean() / price_volatility
                    network_rewards['reward_risk_ratio'] = reward_risk_ratio
                    
            # Real yield (APR - inflation)
            if 'staking_rewards_apr' in df.columns and 'inflation_rate' in df.columns:
                nominal_apr = df['staking_rewards_apr'].mean()
                inflation = df['inflation_rate'].mean()
                network_rewards['real_yield'] = (nominal_apr - inflation) * 100  # Convert to percentage
                
            # Relative yield (compared to total staking ratio)
            if 'staking_rewards_apr' in df.columns and 'staking_ratio' in df.columns:
                apr = df['staking_rewards_apr'].mean()
                staking_ratio = df['staking_ratio'].mean()
                
                if staking_ratio > 0:
                    network_rewards['yield_efficiency'] = apr / staking_ratio
                    
            reward_metrics[network_id] = network_rewards
            
        return reward_metrics

    def compare_networks(self, 
                        network_data: Dict[str, pd.DataFrame],
                        metrics_to_compare: Optional[List[str]] = None) -> NetworkComparison:
        """
        Compare economic metrics across networks.
        
        Args:
            network_data: Dictionary of network economic DataFrames
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
                    
            economic_metric_ids = [m.id for m in self.economic_metrics]
            security_metric_ids = [m.id for m in self.security_metrics]
            metrics_to_compare = [m for m in all_metrics if m in economic_metric_ids or m in security_metric_ids]
            
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
            comparison_type=MetricCategory.ECONOMIC.value
        )
        
        return comparison

    def calculate_security_score(self, 
                               network_data: Dict[str, pd.DataFrame],
                               weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate a composite economic security score for each network.
        
        Args:
            network_data: Dictionary of network economic DataFrames
            weights: Optional dictionary of metric weights (default: equal weights)
            
        Returns:
            Dictionary of security scores for each network
        """
        scores = {}
        
        # Define default weights if not provided
        if not weights:
            default_weights = {
                'nakamoto_coefficient': 0.25,  # Higher is better
                'delegation_concentration': 0.20,  # Lower is better (inverted in normalization)
                'staking_ratio': 0.20,  # Higher is better
                'validator_count': 0.15,  # Higher is better
                'cost_to_attack_usd': 0.20,  # Higher is better
            }
            weights = default_weights
            
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
            for metric_id, value in metric_values.items():
                # Get all values across networks for this metric for normalization
                all_network_values = []
                for net_df in network_data.values():
                    if not net_df.empty and metric_id in net_df.columns:
                        series = net_df[metric_id].dropna()
                        if not series.empty:
                            all_network_values.append(series.mean())
                            
                if all_network_values:
                    min_val = min(all_network_values)
                    max_val = max(all_network_values)
                    
                    if min_val == max_val:
                        normalized = 1.0  # All networks have the same value
                    else:
                        # For metrics where lower is better, invert the normalization
                        if metric_id == 'delegation_concentration':
                            normalized = (max_val - value) / (max_val - min_val)
                        else:
                            normalized = (value - min_val) / (max_val - min_val)
                            
                    normalized_values[metric_id] = normalized
                    
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

    def analyze_trends(self, 
                      network_data: Dict[str, pd.DataFrame],
                      window: str = '1D') -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Analyze economic security trends over time.
        
        Args:
            network_data: Dictionary of network economic DataFrames
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
            
            # Combine economic and security metrics
            all_metrics = self.economic_metrics + self.security_metrics
            metric_trends = {}
            
            for metric in all_metrics:
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
                    
                    metric_trends[metric.id] = metric_trend
                    
            trends[network_id] = metric_trends
            
        return trends