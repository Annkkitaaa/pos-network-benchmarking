"""
Base Collector module for fetching data from Proof-of-Stake networks.
"""
import abc
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class NetworkCollector(abc.ABC):
    """Base class for all network data collectors."""

    def __init__(
        self,
        network_id: str,
        config: Dict[str, Any],
        storage_client=None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize the network collector.

        Args:
            network_id: Unique identifier for the network
            config: Network configuration dictionary
            storage_client: Client for storing collected data
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.network_id = network_id
        self.config = config
        self.storage_client = storage_client
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # Validate configuration
        self._validate_config()
        
        # Set up endpoints
        self.endpoints = self._setup_endpoints()
        
        logger.info(f"Initialized {network_id} collector with {len(self.endpoints)} endpoints")

    def _validate_config(self) -> None:
        """Validate the collector configuration."""
        required_keys = ["endpoints", "metrics"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        if not self.config.get("enabled", False):
            logger.warning(f"Network {self.network_id} is disabled in configuration")

    def _setup_endpoints(self) -> List[Dict[str, Any]]:
        """Set up and validate network endpoints."""
        if not self.config["endpoints"]:
            raise ValueError(f"No endpoints configured for network {self.network_id}")
            
        # Sort endpoints by priority
        return sorted(self.config["endpoints"], key=lambda x: x.get("priority", 999))

    def _make_request(
        self, endpoint_index: int, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """
        Make an HTTP request to the network endpoint.
        
        Args:
            endpoint_index: Index of the endpoint to use
            method: API method to call
            params: Request parameters
            
        Returns:
            Tuple containing success flag and response data or error message
        """
        if endpoint_index >= len(self.endpoints):
            return False, "No valid endpoints available"
            
        endpoint = self.endpoints[endpoint_index]
        url = f"{endpoint['url']}/{method.lstrip('/')}"
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    url, params=params, timeout=self.timeout
                )
                response.raise_for_status()
                return True, response.json()
            except RequestException as e:
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}"
                )
                time.sleep(2 ** attempt)  # Exponential backoff
        
        # If all attempts with current endpoint failed, try next endpoint
        logger.error(f"All attempts failed with endpoint {endpoint['name']}")
        return self._make_request(endpoint_index + 1, method, params)

    @abc.abstractmethod
    def collect_network_stats(self) -> Dict[str, Any]:
        """
        Collect general network statistics.
        
        Returns:
            Dictionary of network statistics
        """
        pass
        
    @abc.abstractmethod
    def collect_validator_metrics(self) -> List[Dict[str, Any]]:
        """
        Collect validator-specific metrics.
        
        Returns:
            List of validator metrics
        """
        pass
        
    @abc.abstractmethod
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """
        Collect network performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        pass
        
    @abc.abstractmethod
    def collect_economic_metrics(self) -> Dict[str, Any]:
        """
        Collect economic metrics like rewards and total stake.
        
        Returns:
            Dictionary of economic metrics
        """
        pass

    def collect_all_metrics(self) -> Dict[str, Any]:
        """
        Collect all available metrics for the network.
        
        Returns:
            Dictionary containing all collected metrics
        """
        try:
            logger.info(f"Starting data collection for {self.network_id}")
            
            timestamp = datetime.utcnow().isoformat()
            
            # Collect data from different categories
            network_stats = self.collect_network_stats()
            validator_metrics = self.collect_validator_metrics()
            performance_metrics = self.collect_performance_metrics()
            economic_metrics = self.collect_economic_metrics()
            
            # Combine all metrics
            metrics = {
                "network_id": self.network_id,
                "timestamp": timestamp,
                "network_stats": network_stats,
                "validator_metrics": validator_metrics,
                "performance_metrics": performance_metrics,
                "economic_metrics": economic_metrics,
            }
            
            # Save to storage if available
            if self.storage_client:
                self.storage_client.save_metrics(metrics)
                
            logger.info(f"Successfully collected all metrics for {self.network_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics for {self.network_id}: {str(e)}")
            raise