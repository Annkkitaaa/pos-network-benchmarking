#!/usr/bin/env python
"""
Data collection script for PoS network benchmarking.
"""
import logging
import time
from datetime import datetime

from config.load import load_config, create_default_configs
from collectors.ethereum import EthereumCollector
from collectors.solana import SolanaCollector
from collectors.cosmos import CosmosCollector
from data.storage import DataStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """Run data collection for all configured networks."""
    # Load configurations
    create_default_configs()
    network_config = load_config("config/networks.yaml")
    
    # Initialize storage
    storage = DataStorage()
    
    # Initialize collectors
    collectors = {}
    
    if network_config["networks"].get("ethereum", {}).get("enabled", False):
        collectors["ethereum"] = EthereumCollector(
            config=network_config["networks"]["ethereum"],
            storage_client=storage
        )
        
    if network_config["networks"].get("solana", {}).get("enabled", False):
        collectors["solana"] = SolanaCollector(
            config=network_config["networks"]["solana"],
            storage_client=storage
        )
        
    if network_config["networks"].get("cosmos", {}).get("enabled", False):
        collectors["cosmos"] = CosmosCollector(
            config=network_config["networks"]["cosmos"],
            storage_client=storage
        )
        
    logger.info(f"Initialized {len(collectors)} collectors")
    
    # Collect data from all networks
    for network_id, collector in collectors.items():
        try:
            logger.info(f"Starting collection for {network_id}")
            metrics = collector.collect_all_metrics()
            logger.info(f"Collected {len(metrics) if isinstance(metrics, dict) else 'N/A'} metrics for {network_id}")
        except Exception as e:
            logger.error(f"Error collecting data for {network_id}: {str(e)}")
    
    logger.info("Data collection completed")

if __name__ == "__main__":
    main()