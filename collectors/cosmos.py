"""
Cosmos network data collector.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

from .base import NetworkCollector

logger = logging.getLogger(__name__)


class CosmosCollector(NetworkCollector):
    """Collector for Cosmos Hub PoS network data."""

    def __init__(
        self,
        config: Dict[str, Any],
        storage_client=None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """Initialize the Cosmos collector."""
        super().__init__("cosmos", config, storage_client, timeout, max_retries)
        self.explorer_url = config.get("explorer", {}).get("url")
        self.explorer_api_key = config.get("explorer", {}).get("api_key")

    def _make_rpc_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a REST request to Cosmos endpoint.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Response data or error
        """
        if not self.endpoints:
            raise ConnectionError("No Cosmos endpoints configured")
            
        # Try each endpoint until successful
        for i, endpoint_config in enumerate(self.endpoints):
            try:
                url = f"{endpoint_config['url']}/{endpoint.lstrip('/')}"
                
                for attempt in range(self.max_retries):
                    try:
                        response = requests.get(
                            url,
                            params=params,
                            timeout=self.timeout
                        )
                        response.raise_for_status()
                        return response.json()
                        
                    except requests.RequestException as e:
                        logger.warning(
                            f"Request failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}"
                        )
                        time.sleep(2 ** attempt)  # Exponential backoff
                        
            except Exception as e:
                logger.error(f"Error with endpoint {endpoint_config['name']}: {str(e)}")
                
            # If we reached here, the current endpoint failed
            if i < len(self.endpoints) - 1:
                logger.warning(f"Trying next endpoint after failure with {endpoint_config['name']}")
                
        # If all endpoints failed
        logger.error("All Cosmos endpoints failed")
        return {}

    def _fetch_from_explorer(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fetch data from blockchain explorer API."""
        if not self.explorer_url:
            logger.warning("Explorer API not configured")
            return {}
            
        headers = {}
        if self.explorer_api_key:
            headers["x-api-key"] = self.explorer_api_key
            
        url = f"{self.explorer_url}/{endpoint.lstrip('/')}"
        
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Explorer API request failed: {str(e)}")
            return {}

    def collect_network_stats(self) -> Dict[str, Any]:
        """Collect general Cosmos network statistics."""
        stats = {}
        
        try:
            # Get latest block
            latest_block = self._make_rpc_request("blocks/latest")
            if latest_block and "block" in latest_block:
                block_data = latest_block["block"]
                stats["latest_block_height"] = int(block_data["header"]["height"])
                stats["latest_block_time"] = block_data["header"]["time"]
                stats["chain_id"] = block_data["header"]["chain_id"]
                
                # Calculate timestamp
                try:
                    block_time = datetime.fromisoformat(stats["latest_block_time"].replace("Z", "+00:00"))
                    stats["latest_block_timestamp"] = block_time.timestamp()
                except (ValueError, AttributeError):
                    pass
                
            # Get node info
            node_info = self._make_rpc_request("node_info")
            if node_info and "application_version" in node_info:
                stats["application_name"] = node_info["application_version"].get("name")
                stats["application_version"] = node_info["application_version"].get("version")
                stats["network"] = node_info.get("network")
                
            # Get syncing status
            syncing = self._make_rpc_request("syncing")
            if syncing is not None:
                stats["is_syncing"] = bool(syncing)
                
            # Calculate block time from recent blocks
            recent_blocks = []
            if "latest_block_height" in stats:
                current_height = stats["latest_block_height"]
                for i in range(10):
                    height = current_height - i
                    if height <= 0:
                        break
                        
                    block = self._make_rpc_request(f"blocks/{height}")
                    if block and "block" in block:
                        recent_blocks.append(block["block"])
                        
            if len(recent_blocks) >= 2:
                try:
                    # Calculate average block time
                    times = []
                    for i in range(len(recent_blocks) - 1):
                        time1 = datetime.fromisoformat(recent_blocks[i]["header"]["time"].replace("Z", "+00:00"))
                        time2 = datetime.fromisoformat(recent_blocks[i+1]["header"]["time"].replace("Z", "+00:00"))
                        time_diff = (time1 - time2).total_seconds()
                        times.append(time_diff)
                        
                    avg_block_time = sum(times) / len(times)
                    stats["avg_block_time"] = avg_block_time
                    
                    # Calculate TPS
                    total_txs = sum(len(block.get("data", {}).get("txs", [])) for block in recent_blocks)
                    stats["avg_tps"] = total_txs / sum(times) if sum(times) > 0 else 0
                    
                except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
                    logger.warning(f"Error calculating block time: {str(e)}")
                
            # Get additional network stats from explorer
            if self.explorer_url:
                network_info = self._fetch_from_explorer("status")
                if network_info:
                    stats.update(network_info.get("result", {}))
                    
            return stats
            
        except Exception as e:
            logger.error(f"Error collecting Cosmos network stats: {str(e)}")
            return {"error": str(e)}

    def collect_validator_metrics(self) -> List[Dict[str, Any]]:
        """Collect Cosmos validator metrics."""
        try:
            # Get validators list
            validators_response = self._make_rpc_request("validators")
            if not validators_response or "validators" not in validators_response:
                return []
                
            validators = validators_response["validators"]
            
            # Collect detailed metrics for each validator
            validator_metrics = []
            
            for validator in validators:
                validator_address = validator.get("address")
                if not validator_address:
                    continue
                    
                # Extract basic info
                validator_info = {
                    "address": validator_address,
                    "moniker": validator.get("description", {}).get("moniker", ""),
                    "identity": validator.get("description", {}).get("identity", ""),
                    "website": validator.get("description", {}).get("website", ""),
                    "voting_power": int(validator.get("voting_power", 0)),
                    "commission_rate": float(validator.get("commission", {}).get("commission_rates", {}).get("rate", 0)),
                    "max_commission": float(validator.get("commission", {}).get("commission_rates", {}).get("max_rate", 0)),
                    "min_self_delegation": int(validator.get("min_self_delegation", 0)),
                    "jailed": validator.get("jailed", False),
                    "status": validator.get("status", ""),
                    "tokens": int(validator.get("tokens", 0)),
                    "delegator_shares": float(validator.get("delegator_shares", 0)),
                }
                
                # Try to get additional info from explorer
                if self.explorer_url:
                    explorer_info = self._fetch_from_explorer(f"validators/{validator_address}")
                    if explorer_info and "validator" in explorer_info:
                        validator_data = explorer_info["validator"]
                        validator_info.update({
                            "uptime": validator_data.get("uptime", 0),
                            "self_delegation": validator_data.get("self_delegation", 0),
                            "total_delegation": validator_data.get("total_delegation", 0),
                            "num_delegators": validator_data.get("num_delegators", 0),
                        })
                
                validator_metrics.append(validator_info)
                
            # Sort by voting power
            validator_metrics.sort(key=lambda x: x.get("voting_power", 0), reverse=True)
            
            # Add ranking
            for i, validator in enumerate(validator_metrics):
                validator["rank"] = i + 1
                
            return validator_metrics
            
        except Exception as e:
            logger.error(f"Error collecting Cosmos validator metrics: {str(e)}")
            return []

    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect Cosmos performance metrics."""
        metrics = {}
        
        try:
            # Get latest block for current height
            latest_block = self._make_rpc_request("blocks/latest")
            if latest_block and "block" in latest_block:
                current_height = int(latest_block["block"]["header"]["height"])
                metrics["current_height"] = current_height
                
                # Get blocks for the last hour
                blocks = []
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=1)
                
                # Convert to ISO format
                start_time_str = start_time.isoformat() + "Z"
                end_time_str = end_time.isoformat() + "Z"
                
                # Get blocks in range
                blocks_response = self._make_rpc_request("blocks", {
                    "minHeight": 1,
                    "maxHeight": current_height,
                    "page": 1,
                    "limit": 100
                })
                
                if blocks_response and "block_metas" in blocks_response:
                    blocks = blocks_response["block_metas"]
                
                # Calculate performance metrics
                if blocks:
                    # Total transactions
                    tx_count = sum(block.get("num_txs", 0) for block in blocks)
                    metrics["tx_count"] = tx_count
                    
                    # Calculate block times
                    block_times = []
                    for i in range(len(blocks) - 1):
                        try:
                            time1 = datetime.fromisoformat(blocks[i]["header"]["time"].replace("Z", "+00:00"))
                            time2 = datetime.fromisoformat(blocks[i+1]["header"]["time"].replace("Z", "+00:00"))
                            delta = (time1 - time2).total_seconds()
                            if delta > 0:  # Skip negative values which indicate out-of-order blocks
                                block_times.append(delta)
                        except (ValueError, KeyError):
                            continue
                    
                    if block_times:
                        metrics["avg_block_time"] = sum(block_times) / len(block_times)
                        metrics["min_block_time"] = min(block_times)
                        metrics["max_block_time"] = max(block_times)
                        
                        # Calculate TPS
                        total_time = sum(block_times)
                        metrics["tps"] = tx_count / total_time if total_time > 0 else 0
                
                # Check for chain halts
                chain_status = self._make_rpc_request("status")
                if chain_status and "sync_info" in chain_status:
                    latest_block_time_str = chain_status["sync_info"].get("latest_block_time")
                    if latest_block_time_str:
                        try:
                            latest_block_time = datetime.fromisoformat(latest_block_time_str.replace("Z", "+00:00"))
                            time_since_last_block = (datetime.utcnow() - latest_block_time).total_seconds()
                            metrics["time_since_last_block"] = time_since_last_block
                            
                            # Flag potential issues
                            metrics["potential_halt"] = time_since_last_block > 30  # If no block for 30+ seconds
                        except ValueError:
                            pass
            
            # Get network load from mempool
            mempool = self._make_rpc_request("num_unconfirmed_txs")
            if mempool and "total" in mempool:
                metrics["pending_txs"] = int(mempool["total"])
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting Cosmos performance metrics: {str(e)}")
            return {"error": str(e)}

    def collect_economic_metrics(self) -> Dict[str, Any]:
        """Collect Cosmos economic metrics."""
        metrics = {}
        
        try:
            # Get total supply
            supply = self._make_rpc_request("supply/total")
            if supply and "result" in supply:
                for token in supply["result"]:
                    if token["denom"] == "uatom":  # Cosmos Hub uses uatom (micro ATOM)
                        metrics["total_supply"] = int(token["amount"]) / 1_000_000  # Convert to ATOM
                        break
            
            # Get staking parameters
            staking_params = self._make_rpc_request("staking/parameters")
            if staking_params and "result" in staking_params:
                metrics["unbonding_time"] = staking_params["result"].get("unbonding_time")
                metrics["max_validators"] = int(staking_params["result"].get("max_validators", 0))
                metrics["bond_denom"] = staking_params["result"].get("bond_denom")
            
            # Get staking pool info
            staking_pool = self._make_rpc_request("staking/pool")
            if staking_pool and "result" in staking_pool:
                bonded = int(staking_pool["result"].get("bonded_tokens", 0)) / 1_000_000  # Convert to ATOM
                not_bonded = int(staking_pool["result"].get("not_bonded_tokens", 0)) / 1_000_000  # Convert to ATOM
                
                metrics["total_bonded"] = bonded
                metrics["total_not_bonded"] = not_bonded
                
                if "total_supply" in metrics and metrics["total_supply"] > 0:
                    metrics["staking_ratio"] = bonded / metrics["total_supply"]
            
            # Get inflation rate
            mint_params = self._make_rpc_request("minting/parameters")
            if mint_params and "result" in mint_params:
                metrics["inflation"] = float(mint_params["result"].get("inflation", 0))
                metrics["annual_provisions"] = float(mint_params["result"].get("annual_provisions", 0)) / 1_000_000
            
            # Calculate staking APR
            if "annual_provisions" in metrics and "total_bonded" in metrics and metrics["total_bonded"] > 0:
                metrics["staking_apr"] = metrics["annual_provisions"] / metrics["total_bonded"]
            
            # Get additional economic data from explorer
            if self.explorer_url:
                economic_data = self._fetch_from_explorer("market/summary")
                if economic_data and "result" in economic_data:
                    price_data = economic_data["result"]
                    metrics.update({
                        "price_usd": price_data.get("price_usd", 0),
                        "market_cap_usd": price_data.get("market_cap_usd", 0),
                        "volume_24h_usd": price_data.get("volume_24h_usd", 0),
                    })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting Cosmos economic metrics: {str(e)}")
            return {"error": str(e)}

    def collect_mev_metrics(self) -> Dict[str, Any]:
        """Collect MEV-related metrics from Cosmos."""
        # MEV in Cosmos chains has different patterns than Ethereum
        # This is a simplified placeholder
        
        # In a real implementation, you would analyze:
        # 1. Transaction ordering within blocks
        # 2. DEX arbitrage opportunities (e.g., on Osmosis)
        # 3. Validator-extracted MEV patterns
        
        metrics = {
            "mev_detected_blocks": 0,
            "mev_extracted_value": 0,
            "arbitrage_instances": 0,
            "sandwich_attacks": 0,
        }
        
        return metrics