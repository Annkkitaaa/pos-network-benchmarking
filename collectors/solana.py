"""
Solana network data collector.
"""
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from base64 import b64decode

from .base import NetworkCollector

logger = logging.getLogger(__name__)


class SolanaCollector(NetworkCollector):
    """Collector for Solana PoS network data."""

    def __init__(
        self,
        config: Dict[str, Any],
        storage_client=None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """Initialize the Solana collector."""
        super().__init__("solana", config, storage_client, timeout, max_retries)
        self.explorer_url = config.get("explorer", {}).get("url")
        self.explorer_api_key = config.get("explorer", {}).get("api_key")

    def _make_rpc_request(self, method: str, params: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Make a JSON-RPC request to Solana endpoint.
        
        Args:
            method: RPC method name
            params: RPC parameters
            
        Returns:
            Response data or error
        """
        if not self.endpoints:
            raise ConnectionError("No Solana endpoints configured")
            
        # Prepare request payload
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or []
        }
        
        headers = {
            "Content-Type": "application/json",
        }
        
        # Try each endpoint until successful
        for i, endpoint in enumerate(self.endpoints):
            try:
                for attempt in range(self.max_retries):
                    try:
                        response = requests.post(
                            endpoint["url"],
                            json=payload,
                            headers=headers,
                            timeout=self.timeout
                        )
                        response.raise_for_status()
                        result = response.json()
                        
                        if "result" in result:
                            return result["result"]
                        elif "error" in result:
                            logger.warning(f"RPC error: {result['error']}")
                            break
                        
                    except requests.RequestException as e:
                        logger.warning(
                            f"Request failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}"
                        )
                        time.sleep(2 ** attempt)  # Exponential backoff
                        
            except Exception as e:
                logger.error(f"Error with endpoint {endpoint['name']}: {str(e)}")
                
            # If we reached here, the current endpoint failed
            if i < len(self.endpoints) - 1:
                logger.warning(f"Trying next endpoint after failure with {endpoint['name']}")
                
        # If all endpoints failed
        logger.error("All Solana endpoints failed")
        return {}

    def _fetch_from_explorer(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fetch data from blockchain explorer API."""
        if not self.explorer_url or not self.explorer_api_key:
            logger.warning("Explorer API not configured")
            return {}
            
        headers = {
            "Content-Type": "application/json",
        }
        
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
        """Collect general Solana network statistics."""
        stats = {}
        
        try:
            # Get current slot
            current_slot = self._make_rpc_request("getSlot")
            if current_slot:
                stats["latest_slot"] = current_slot
                
            # Get genesis hash
            genesis_hash = self._make_rpc_request("getGenesisHash")
            if genesis_hash:
                stats["genesis_hash"] = genesis_hash
                
            # Get latest block info
            latest_block = self._make_rpc_request("getLatestBlockhash")
            if latest_block:
                stats["latest_blockhash"] = latest_block.get("blockhash")
                stats["latest_block_height"] = latest_block.get("lastValidBlockHeight")
                
            # Get cluster nodes
            cluster_nodes = self._make_rpc_request("getClusterNodes")
            if cluster_nodes:
                stats["node_count"] = len(cluster_nodes)
                
            # Get version
            version = self._make_rpc_request("getVersion")
            if version:
                stats["solana_version"] = f"{version.get('solana-core', '')}"
                
            # Calculate TPS
            # We need recent performance samples
            performance_samples = self._make_rpc_request("getRecentPerformanceSamples", [5])
            if performance_samples and len(performance_samples) > 0:
                total_transactions = sum(sample.get("numTransactions", 0) for sample in performance_samples)
                total_slots = sum(sample.get("numSlots", 0) for sample in performance_samples)
                
                # Calculate average block time based on samples
                if total_slots > 0:
                    # Solana aims for 400ms slot time
                    sample_time = 0.4 * total_slots  # in seconds
                    stats["avg_tps"] = total_transactions / sample_time if sample_time > 0 else 0
                    stats["avg_slot_time"] = sample_time / total_slots if total_slots > 0 else 0.4
                
            # Get epoch info
            epoch_info = self._make_rpc_request("getEpochInfo")
            if epoch_info:
                stats["epoch"] = epoch_info.get("epoch")
                stats["slot_index"] = epoch_info.get("slotIndex")
                stats["slots_in_epoch"] = epoch_info.get("slotsInEpoch")
                stats["epoch_progress"] = (epoch_info.get("slotIndex", 0) / epoch_info.get("slotsInEpoch", 1)) * 100
                
            # Get additional stats from explorer API if available
            if self.explorer_url:
                chain_stats = self._fetch_from_explorer("chain/stats")
                if chain_stats:
                    stats["total_transactions"] = chain_stats.get("totalTransactions")
                    stats["tps_24h"] = chain_stats.get("tps24h")
                    
            return stats
            
        except Exception as e:
            logger.error(f"Error collecting Solana network stats: {str(e)}")
            return {"error": str(e)}

    def collect_validator_metrics(self) -> List[Dict[str, Any]]:
        """Collect Solana validator metrics."""
        try:
            # Get vote accounts
            vote_accounts = self._make_rpc_request("getVoteAccounts")
            if not vote_accounts:
                return []
                
            validators = []
            
            # Combine current and delinquent validators
            all_validators = vote_accounts.get("current", []) + vote_accounts.get("delinquent", [])
            
            for validator in all_validators:
                validator_info = {
                    "pubkey": validator.get("votePubkey"),
                    "node_pubkey": validator.get("nodePubkey"),
                    "activated_stake": validator.get("activatedStake", 0) / 1_000_000_000,  # Convert to SOL
                    "is_active": validator in vote_accounts.get("current", []),
                    "commission": validator.get("commission"),
                    "last_vote": validator.get("lastVote"),
                    "root_slot": validator.get("rootSlot"),
                }
                
                validators.append(validator_info)
                
            # Sort by stake amount
            validators.sort(key=lambda x: x["activated_stake"], reverse=True)
            
            # Calculate and add ranks
            for i, validator in enumerate(validators):
                validator["rank"] = i + 1
                
            # Get additional validator info from explorer API if available
            if self.explorer_url and validators:
                # This is a simplified approach - in a real implementation,
                # you would batch these requests or have a more efficient API
                for validator in validators[:20]:  # Limit to top 20 for demonstration
                    validator_pubkey = validator["pubkey"]
                    validator_details = self._fetch_from_explorer(f"validator/{validator_pubkey}")
                    
                    if validator_details:
                        validator["skip_rate"] = validator_details.get("skipRate")
                        validator["uptime"] = validator_details.get("uptime")
                        
            return validators
            
        except Exception as e:
            logger.error(f"Error collecting Solana validator metrics: {str(e)}")
            return []

    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect Solana performance metrics."""
        metrics = {}
        
        try:
            # Get recent performance samples
            performance_samples = self._make_rpc_request("getRecentPerformanceSamples", [10])
            if performance_samples:
                # Calculate average values
                total_tx = sum(sample.get("numTransactions", 0) for sample in performance_samples)
                total_slots = sum(sample.get("numSlots", 0) for sample in performance_samples)
                total_tx_per_slot = total_tx / total_slots if total_slots > 0 else 0
                
                metrics["avg_tx_per_slot"] = total_tx_per_slot
                metrics["avg_tx_count"] = total_tx / len(performance_samples) if performance_samples else 0
                metrics["avg_slot_count"] = total_slots / len(performance_samples) if performance_samples else 0
                
                # Calculate TPS based on performance samples
                sample_time = 0.4 * total_slots  # 400ms per slot
                metrics["tps"] = total_tx / sample_time if sample_time > 0 else 0
                
                # First slot in samples
                first_slot = performance_samples[-1].get("slot", 0) if performance_samples else 0
                # Last slot in samples
                last_slot = performance_samples[0].get("slot", 0) if performance_samples else 0
                # Duration between first and last slot
                slot_duration = (last_slot - first_slot) * 0.4  # 400ms per slot
                
                if slot_duration > 0:
                    metrics["block_time"] = slot_duration / (len(performance_samples) - 1) if len(performance_samples) > 1 else 0.4
                else:
                    metrics["block_time"] = 0.4  # Solana's target slot time
                    
                # Add additional metrics like network congestion based on dropped tx
                total_dropped = sum(sample.get("numTransactionsDropped", 0) for sample in performance_samples)
                if total_tx > 0:
                    metrics["tx_drop_rate"] = total_dropped / (total_tx + total_dropped)
                else:
                    metrics["tx_drop_rate"] = 0
                    
            # Get supply info for additional context
            supply_info = self._make_rpc_request("getSupply")
            if supply_info and "value" in supply_info:
                total_supply = supply_info["value"].get("total", 0) / 1_000_000_000  # Convert to SOL
                circulating = supply_info["value"].get("circulating", 0) / 1_000_000_000  # Convert to SOL
                metrics["total_supply"] = total_supply
                metrics["circulating_supply"] = circulating
                metrics["circulating_ratio"] = (circulating / total_supply) if total_supply > 0 else 0
                
            # Fetch health
            health = self._make_rpc_request("getHealth")
            metrics["is_healthy"] = health == "ok"
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting Solana performance metrics: {str(e)}")
            return {"error": str(e)}

    def collect_economic_metrics(self) -> Dict[str, Any]:
        """Collect Solana economic metrics."""
        metrics = {}
        
        try:
            # Get inflation info
            inflation = self._make_rpc_request("getInflationReward", [[]])
            if inflation:
                metrics["inflation_rate"] = inflation[0].get("rate", 0) if inflation and len(inflation) > 0 else 0
                
            # Get vote accounts for staking info
            vote_accounts = self._make_rpc_request("getVoteAccounts")
            if vote_accounts:
                # Calculate total stake
                total_stake = sum(
                    validator.get("activatedStake", 0) 
                    for validator in vote_accounts.get("current", []) + vote_accounts.get("delinquent", [])
                ) / 1_000_000_000  # Convert lamports to SOL
                
                metrics["total_staked"] = total_stake
                metrics["validator_count"] = len(vote_accounts.get("current", []))
                metrics["delinquent_count"] = len(vote_accounts.get("delinquent", []))
                
                # Get supply for stake ratio calculation
                supply_info = self._make_rpc_request("getSupply")
                if supply_info and "value" in supply_info:
                    total_supply = supply_info["value"].get("total", 0) / 1_000_000_000  # Convert to SOL
                    metrics["staking_ratio"] = total_stake / total_supply if total_supply > 0 else 0
                    
            # Estimate staking APY
            # This is a simplified calculation - actual Solana APY varies based on multiple factors
            if "staking_ratio" in metrics and "inflation_rate" in metrics:
                metrics["staking_apy"] = metrics["inflation_rate"] / metrics["staking_ratio"] if metrics["staking_ratio"] > 0 else 0
                
            # Get token price and market info from external source if needed
            # This would require integration with a price API like CoinGecko
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting Solana economic metrics: {str(e)}")
            return {"error": str(e)}

    def collect_mev_metrics(self) -> Dict[str, Any]:
        """Collect MEV-related metrics from Solana."""
        # MEV in Solana is different from Ethereum due to its architecture
        # This is a simplified placeholder - a real implementation would need
        # specialized analysis of transaction ordering and MEV extraction patterns
        
        metrics = {
            "mev_detected_blocks": 0,
            "mev_extracted_value": 0,
            "sandwich_attacks": 0,
            "frontrunning_instances": 0,
            "backrunning_instances": 0,
            "arbitrage_instances": 0,
        }
        
        # In a real implementation, you would analyze recent blocks for:
        # 1. JIT transactions and MEV opportunities
        # 2. Arbitrage transactions (especially on Serum/Raydium)
        # 3. Transaction prioritization patterns
        
        return metrics