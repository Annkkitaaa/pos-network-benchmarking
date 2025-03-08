"""
Configuration loading utilities.
"""
import logging
import os
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    try:
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file not found: {config_path}")
            return {}
            
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        # Process environment variables in configuration strings
        return process_env_vars(config)
        
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}")
        return {}


def process_env_vars(config: Any) -> Any:
    """
    Process environment variables in configuration values.
    
    Args:
        config: Configuration object (dict, list, or scalar)
        
    Returns:
        Configuration with environment variables replaced
    """
    if isinstance(config, dict):
        return {key: process_env_vars(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [process_env_vars(item) for item in config]
    elif isinstance(config, str):
        return replace_env_vars(config)
    else:
        return config


def replace_env_vars(value: str) -> str:
    """
    Replace environment variables in a string.
    
    Args:
        value: String that may contain environment variable references
        
    Returns:
        String with environment variables replaced
    """
    # Skip processing if the string doesn't contain ${
    if '${' not in value:
        return value
        
    # Find all environment variables in the string
    i = 0
    result = ""
    
    while i < len(value):
        # Find the start of an environment variable
        start = value.find('${', i)
        if start == -1:
            # No more environment variables
            result += value[i:]
            break
            
        # Add everything up to the start of the environment variable
        result += value[i:start]
        
        # Find the end of the environment variable
        end = value.find('}', start)
        if end == -1:
            # No closing brace, treat as literal
            result += value[start:]
            break
            
        # Extract the environment variable name
        env_var = value[start+2:end]
        
        # Replace with the environment variable value or an empty string
        env_value = os.environ.get(env_var, '')
        result += env_value
        
        # Move past the environment variable
        i = end + 1
        
    return result


def create_default_configs():
    """Create default configuration files if they don't exist."""
    config_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create app_config.yaml if it doesn't exist
    app_config_path = os.path.join(config_dir, 'app_config.yaml')
    if not os.path.exists(app_config_path):
        default_app_config = {
            'app': {
                'name': 'PoS Network Benchmarking Tool',
                'version': '0.1.0',
                'log_level': 'INFO',
                'data_dir': 'data',
            },
            'collection': {
                'interval_minutes': 15,
                'history_days': 30,
            },
            'dashboard': {
                'theme': 'light',
                'default_time_range': '7d',
                'refresh_interval_seconds': 300,
            },
        }
        
        with open(app_config_path, 'w') as file:
            yaml.dump(default_app_config, file, default_flow_style=False)
            logger.info(f"Created default app configuration: {app_config_path}")
    
    # Create networks.yaml if it doesn't exist
    networks_config_path = os.path.join(config_dir, 'networks.yaml')
    if not os.path.exists(networks_config_path):
        default_networks_config = {
            'networks': {
                'ethereum': {
                    'enabled': True,
                    'endpoints': [
                        {
                            'name': 'Infura',
                            'url': 'https://mainnet.infura.io/v3/${INFURA_API_KEY}',
                            'priority': 1,
                        },
                    ],
                    'explorer': {
                        'url': 'https://api.etherscan.io/api',
                        'api_key': '${ETHERSCAN_API_KEY}',
                    },
                    'metrics': [
                        'validator_count',
                        'total_staked',
                        'reward_rate',
                        'avg_block_time',
                        'transaction_throughput',
                    ],
                },
                'solana': {
                    'enabled': True,
                    'endpoints': [
                        {
                            'name': 'Official RPC',
                            'url': 'https://api.mainnet-beta.solana.com',
                            'priority': 1,
                        },
                    ],
                    'explorer': {
                        'url': 'https://public-api.solscan.io',
                        'api_key': '${SOLSCAN_API_KEY}',
                    },
                    'metrics': [
                        'validator_count',
                        'total_staked',
                        'reward_rate',
                        'avg_block_time',
                        'transaction_throughput',
                    ],
                },
            },
            'collection': {
                'interval_minutes': 15,
                'history_days': 30,
                'timeout_seconds': 30,
                'retry_attempts': 3,
            },
        }
        
        with open(networks_config_path, 'w') as file:
            yaml.dump(default_networks_config, file, default_flow_style=False)
            logger.info(f"Created default networks configuration: {networks_config_path}")