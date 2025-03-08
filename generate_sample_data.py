"""
Generate comprehensive sample data for the Network Performance Benchmarking Tool.
"""
import logging
import random
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data():
    """Generate sample metrics data for demonstration across all features."""
    from data.storage import DataStorage
    storage = DataStorage()
    
    # First, clear existing data
    conn = sqlite3.connect(storage.db_path)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM metrics')
    cursor.execute('DELETE FROM network_stats')
    cursor.execute('DELETE FROM validator_metrics')
    conn.commit()
    conn.close()
    
    networks = ["ethereum", "solana"]
    
    # Generate data for the past 30 days with hourly granularity
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    # Define network characteristics
    network_characteristics = {
        "ethereum": {
            "tps_range": (10, 30),
            "block_time_range": (12, 14),
            "finality_time_range": (12 * 60, 15 * 60),  # 12-15 minutes in seconds
            "utilization_range": (0.6, 0.9),
            "validator_count_range": (700000, 750000),
            "total_staked_range": (20000000, 25000000),
            "staking_ratio_range": (0.18, 0.22),
            "staking_apr_range": (0.03, 0.05),
            "token_price_range": (3000, 4000),
            "mev_extracted_range": (200000, 300000),
        },
        "solana": {
            "tps_range": (1500, 3000),
            "block_time_range": (0.4, 0.6),
            "finality_time_range": (1, 3),  # 1-3 seconds
            "utilization_range": (0.4, 0.7),
            "validator_count_range": (1500, 1700),
            "total_staked_range": (350000000, 400000000),
            "staking_ratio_range": (0.65, 0.75),
            "staking_apr_range": (0.06, 0.08),
            "token_price_range": (100, 150),
            "mev_extracted_range": (50000, 100000),
        }
    }
    
    # Performance metrics
    performance_metrics = [
        "tps", "block_time", "finality_time", "network_utilization", 
        "gas_used", "gas_limit", "pending_transactions"
    ]
    
    # Economic metrics
    economic_metrics = [
        "total_staked", "staking_ratio", "staking_rewards_apr", 
        "validator_count", "active_validators", "token_price_usd", "market_cap_usd"
    ]
    
    # MEV metrics
    mev_metrics = [
        "mev_extracted", "sandwich_attacks", "frontrunning_instances", 
        "backrunning_instances", "arbitrage_instances", "mev_to_rewards_ratio"
    ]
    
    # Security metrics
    security_metrics = [
        "nakamoto_coefficient", "delegation_concentration", "cost_to_attack_usd"
    ]
    
    logger.info("Generating comprehensive sample data...")
    
    # Generate hourly data points
    hourly_timestamps = []
    current_time = start_time
    while current_time <= end_time:
        hourly_timestamps.append(current_time)
        current_time += timedelta(hours=1)
    
    # Add some randomness but maintain trends
    for network in networks:
        # Base values that will have trends applied
        chars = network_characteristics[network]
        
        # Create trend patterns
        days = 30
        hour_points = days * 24
        
        # Sinusoidal pattern for TPS to show daily cycles
        tps_base = np.linspace(chars["tps_range"][0], chars["tps_range"][1], hour_points)
        tps_trend = tps_base + np.sin(np.linspace(0, 2*np.pi*10, hour_points)) * (chars["tps_range"][1] - chars["tps_range"][0])/5
        
        # Increasing trend for token price
        price_trend = np.linspace(chars["token_price_range"][0], chars["token_price_range"][1], hour_points)
        
        # Gradually increasing staking ratio
        staking_ratio_trend = np.linspace(chars["staking_ratio_range"][0], chars["staking_ratio_range"][1], hour_points)
        
        # Random fluctuations in MEV extraction with occasional spikes
        mev_base = np.linspace(chars["mev_extracted_range"][0], chars["mev_extracted_range"][1], hour_points)
        mev_spikes = np.zeros(hour_points)
        for _ in range(5):  # Add 5 random spikes
            spike_idx = random.randint(0, hour_points-1)
            mev_spikes[spike_idx] = random.uniform(chars["mev_extracted_range"][1], chars["mev_extracted_range"][1]*1.5)
        mev_trend = mev_base + mev_spikes
        
        # Generate data for all timestamps
        for idx, timestamp in enumerate(hourly_timestamps):
            timestamp_str = timestamp.isoformat()
            
            # PERFORMANCE METRICS
            storage._execute_query(
                'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                (network, "tps", timestamp_str, tps_trend[idx % hour_points] * (1 + random.uniform(-0.1, 0.1)), "performance")
            )
            
            storage._execute_query(
                'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                (network, "block_time", timestamp_str, random.uniform(chars["block_time_range"][0], chars["block_time_range"][1]), "performance")
            )
            
            storage._execute_query(
                'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                (network, "finality_time", timestamp_str, random.uniform(chars["finality_time_range"][0], chars["finality_time_range"][1]), "performance")
            )
            
            storage._execute_query(
                'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                (network, "network_utilization", timestamp_str, random.uniform(chars["utilization_range"][0], chars["utilization_range"][1]), "performance")
            )
            
            if network == "ethereum":
                gas_limit = 30000000
                gas_used = gas_limit * random.uniform(0.7, 0.95)
                storage._execute_query(
                    'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                    (network, "gas_limit", timestamp_str, gas_limit, "performance")
                )
                storage._execute_query(
                    'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                    (network, "gas_used", timestamp_str, gas_used, "performance")
                )
            
            storage._execute_query(
                'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                (network, "pending_transactions", timestamp_str, random.randint(1000, 5000), "performance")
            )
            
            # ECONOMIC METRICS
            storage._execute_query(
                'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                (network, "total_staked", timestamp_str, chars["total_staked_range"][0] + 
                 (idx / hour_points) * (chars["total_staked_range"][1] - chars["total_staked_range"][0]), "economic")
            )
            
            storage._execute_query(
                'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                (network, "staking_ratio", timestamp_str, staking_ratio_trend[idx % hour_points], "economic")
            )
            
            storage._execute_query(
                'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                (network, "staking_rewards_apr", timestamp_str, random.uniform(chars["staking_apr_range"][0], chars["staking_apr_range"][1]), "economic")
            )
            
            storage._execute_query(
                'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                (network, "validator_count", timestamp_str, random.randint(int(chars["validator_count_range"][0]), 
                                                                        int(chars["validator_count_range"][1])), "economic")
            )
            
            storage._execute_query(
                'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                (network, "active_validators", timestamp_str, random.randint(int(chars["validator_count_range"][0] * 0.95), 
                                                                        int(chars["validator_count_range"][1] * 0.98)), "economic")
            )
            
            token_price = price_trend[idx % hour_points] * (1 + random.uniform(-0.03, 0.03))
            storage._execute_query(
                'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                (network, "token_price_usd", timestamp_str, token_price, "economic")
            )
            
            # Calculate market cap based on total supply and price
            total_supply = chars["total_staked_range"][1] / chars["staking_ratio_range"][1]
            market_cap = total_supply * token_price
            storage._execute_query(
                'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                (network, "market_cap_usd", timestamp_str, market_cap, "economic")
            )
            
            # MEV METRICS
            mev_value = mev_trend[idx % hour_points] * (1 + random.uniform(-0.1, 0.1))
            storage._execute_query(
                'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                (network, "mev_extracted", timestamp_str, mev_value, "mev")
            )
            
            # Generate related MEV metrics
            sandwich_attacks = int(mev_value * random.uniform(0.001, 0.002))
            frontrunning = int(mev_value * random.uniform(0.002, 0.003))
            backrunning = int(mev_value * random.uniform(0.0015, 0.0025))
            arbitrage = int(mev_value * random.uniform(0.003, 0.004))
            
            storage._execute_query(
                'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                (network, "sandwich_attacks", timestamp_str, sandwich_attacks, "mev")
            )
            
            storage._execute_query(
                'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                (network, "frontrunning_instances", timestamp_str, frontrunning, "mev")
            )
            
            storage._execute_query(
                'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                (network, "backrunning_instances", timestamp_str, backrunning, "mev")
            )
            
            storage._execute_query(
                'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                (network, "arbitrage_instances", timestamp_str, arbitrage, "mev")
            )
            
            # MEV to rewards ratio
            daily_rewards = chars["total_staked_range"][0] * chars["staking_apr_range"][0] / 365
            mev_ratio = mev_value / daily_rewards if daily_rewards > 0 else 0
            storage._execute_query(
                'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                (network, "mev_to_rewards_ratio", timestamp_str, mev_ratio, "mev")
            )
            
            # SECURITY METRICS
            nakamoto = 20 if network == "ethereum" else 10  # Higher is better
            storage._execute_query(
                'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                (network, "nakamoto_coefficient", timestamp_str, nakamoto + random.randint(-2, 2), "security")
            )
            
            concentration = 0.45 if network == "ethereum" else 0.65  # Lower is better
            storage._execute_query(
                'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                (network, "delegation_concentration", timestamp_str, concentration + random.uniform(-0.05, 0.05), "security")
            )
            
            # Cost to attack in millions USD
            cost_to_attack = market_cap * chars["staking_ratio_range"][0] * 0.33 / 1000000
            storage._execute_query(
                'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                (network, "cost_to_attack_usd", timestamp_str, cost_to_attack, "security")
            )
    
    # Generate validator data (simplified)
    for network in networks:
        validator_count = int(network_characteristics[network]["validator_count_range"][0])
        
        # Create a sample of validators
        for i in range(1, min(validator_count, 50)):  # Limit to 50 validators for sample
            validator_id = f"{network}_validator_{i}"
            stake_amount = random.uniform(1000, 100000)
            uptime = random.uniform(95, 100)
            
            validator_data = {
                "address": validator_id,
                "stake_amount": stake_amount,
                "is_active": random.random() > 0.05,  # 95% chance of being active
                "uptime_percentage": uptime,
                "blocks_proposed": random.randint(100, 1000),
                "blocks_missed": random.randint(0, 50),
                "reward_rate": random.uniform(0.03, 0.05),
                "commission_rate": random.uniform(0.01, 0.1),
                "slash_count": random.randint(0, 2),
                "rank": i
            }
            
            # Store validator metrics
            storage._execute_query(
                'INSERT INTO validator_metrics (network_id, validator_id, timestamp, metrics) VALUES (?, ?, ?, ?)',
                (network, validator_id, end_time.isoformat(), str(validator_data))
            )
    
    logger.info(f"Sample data generation complete! Generated data for {len(hourly_timestamps)} time points across {len(networks)} networks.")

if __name__ == "__main__":
    generate_sample_data()