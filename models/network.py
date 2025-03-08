"""
Network models for representing blockchain network data.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


@dataclass
class NetworkStats:
    """General network statistics."""
    network_id: str
    timestamp: datetime
    latest_block: int
    latest_block_timestamp: int
    avg_block_time: float
    avg_tps: float
    node_count: Optional[int] = None
    peer_count: Optional[int] = None
    network_version: Optional[str] = None
    is_syncing: bool = False
    chain_id: Optional[int] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EconomicMetrics:
    """Network economic metrics."""
    network_id: str
    timestamp: datetime
    total_staked: float
    staking_ratio: float
    staking_rewards_apr: float
    total_validators: int
    active_validators: int
    token_price_usd: Optional[float] = None
    market_cap_usd: Optional[float] = None
    total_supply: Optional[float] = None
    circulating_supply: Optional[float] = None
    delegation_concentration: Optional[float] = None  # Gini coefficient
    min_stake_amount: Optional[float] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MEVMetrics:
    """Maximal Extractable Value metrics."""
    network_id: str
    timestamp: datetime
    mev_extracted_24h: float
    sandwich_attacks_count: int
    frontrunning_instances: int
    backrunning_instances: int
    arbitrage_instances: int
    avg_mev_per_block: float
    pct_blocks_with_mev: float
    mev_to_rewards_ratio: float
    top_extractors: List[Dict[str, Any]] = field(default_factory=list)
    extra_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkComparison:
    """Comparison metrics between networks."""
    timestamp: datetime
    networks: List[str]
    metrics: Dict[str, Dict[str, float]]
    period_start: datetime
    period_end: datetime
    comparison_type: str  # 'performance', 'economic', 'security', 'mev'
    notes: Optional[str] = None


@dataclass
class ValidatorMetrics:
    """Validator-specific metrics."""
    network_id: str
    validator_id: str
    address: str
    stake_amount: float
    is_active: bool
    uptime_percentage: float
    blocks_proposed: int
    blocks_missed: int
    reward_rate: float
    commission_rate: Optional[float] = None
    voting_power: Optional[float] = None
    slash_count: int = 0
    jailed_until: Optional[datetime] = None
    rank: Optional[int] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Network performance metrics."""
    network_id: str
    timestamp: datetime
    tps: float
    latency_ms: float
    block_time: float
    block_size: float
    gas_used: Optional[float] = None
    gas_limit: Optional[float] = None
    pending_transactions: int = 0
    network_utilization: float = 0.0
    finality_time: Optional[float] = None
    consensus_time: Optional[float] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)