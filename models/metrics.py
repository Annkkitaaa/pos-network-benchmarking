"""
Metrics definitions and calculation utilities.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class MetricCategory(Enum):
    """Categories of metrics to organize and group them."""
    PERFORMANCE = "performance"
    ECONOMIC = "economic"
    SECURITY = "security"
    MEV = "mev"
    VALIDATOR = "validator"
    NETWORK = "network"


class MetricAggregation(Enum):
    """Available aggregation methods for metrics."""
    AVG = "avg"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    COUNT = "count"
    LAST = "last"
    FIRST = "first"
    STD_DEV = "std_dev"
    PERCENTILE_95 = "percentile_95"
    PERCENTILE_99 = "percentile_99"


@dataclass
class Metric:
    """Definition of a metric with metadata."""
    id: str
    name: str
    description: str
    category: MetricCategory
    unit: str
    aggregation: MetricAggregation = MetricAggregation.AVG
    is_higher_better: bool = True
    tags: List[str] = None
    networks: Set[str] = None  # Networks where this metric is available
    calculation_fn: Optional[Callable] = None
    dependencies: List[str] = None  # Other metrics required for calculation
    thresholds: Dict[str, float] = None  # Thresholds for different levels (e.g., "warning", "critical")
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.networks is None:
            self.networks = set()
        if self.dependencies is None:
            self.dependencies = []
        if self.thresholds is None:
            self.thresholds = {}


# Performance Metrics
TPS = Metric(
    id="tps",
    name="Transactions Per Second",
    description="Average number of transactions processed per second",
    category=MetricCategory.PERFORMANCE,
    unit="tx/s",
    is_higher_better=True,
    tags=["throughput", "scalability"]
)

BLOCK_TIME = Metric(
    id="block_time",
    name="Block Time",
    description="Average time between blocks",
    category=MetricCategory.PERFORMANCE,
    unit="seconds",
    is_higher_better=False,
    tags=["latency", "responsiveness"]
)

FINALITY_TIME = Metric(
    id="finality_time",
    name="Finality Time",
    description="Time until a transaction is considered final",
    category=MetricCategory.PERFORMANCE,
    unit="seconds",
    is_higher_better=False,
    tags=["finality", "security"]
)

NETWORK_UTILIZATION = Metric(
    id="network_utilization",
    name="Network Utilization",
    description="Percentage of network capacity currently being used",
    category=MetricCategory.PERFORMANCE,
    unit="%",
    is_higher_better=True,  # Debatable
    thresholds={"warning": 85, "critical": 95},
    tags=["capacity", "congestion"]
)

# Economic Metrics
STAKING_RATIO = Metric(
    id="staking_ratio",
    name="Staking Ratio",
    description="Percentage of total supply that is staked",
    category=MetricCategory.ECONOMIC,
    unit="%",
    is_higher_better=True,
    tags=["security", "participation"]
)

STAKING_REWARDS = Metric(
    id="staking_rewards",
    name="Staking Rewards",
    description="Annual percentage return for stakers",
    category=MetricCategory.ECONOMIC,
    unit="% APR",
    is_higher_better=True,
    tags=["rewards", "incentives"]
)

VALIDATOR_COUNT = Metric(
    id="validator_count",
    name="Validator Count",
    description="Number of active validators",
    category=MetricCategory.SECURITY,
    unit="validators",
    aggregation=MetricAggregation.LAST,
    is_higher_better=True,
    tags=["decentralization", "security"]
)

# MEV Metrics
MEV_EXTRACTED = Metric(
    id="mev_extracted",
    name="MEV Extracted",
    description="Value extracted through MEV opportunities",
    category=MetricCategory.MEV,
    unit="USD",
    is_higher_better=False,
    tags=["mev", "fairness"]
)

SANDWICH_ATTACKS = Metric(
    id="sandwich_attacks",
    name="Sandwich Attacks",
    description="Number of sandwich attacks detected",
    category=MetricCategory.MEV,
    unit="count",
    is_higher_better=False,
    tags=["mev", "attacks"]
)

# Security Metrics
NAKAMOTO_COEFFICIENT = Metric(
    id="nakamoto_coefficient",
    name="Nakamoto Coefficient",
    description="Minimum entities needed to compromise the network",
    category=MetricCategory.SECURITY,
    unit="entities",
    is_higher_better=True,
    tags=["decentralization", "security"]
)

DELEGATION_CONCENTRATION = Metric(
    id="delegation_concentration",
    name="Delegation Concentration",
    description="Gini coefficient of stake distribution",
    category=MetricCategory.SECURITY,
    unit="coefficient",
    is_higher_better=False,
    tags=["decentralization", "concentration"]
)


# All metrics dictionary for easy lookup
ALL_METRICS = {
    TPS.id: TPS,
    BLOCK_TIME.id: BLOCK_TIME,
    FINALITY_TIME.id: FINALITY_TIME,
    NETWORK_UTILIZATION.id: NETWORK_UTILIZATION,
    STAKING_RATIO.id: STAKING_RATIO,
    STAKING_REWARDS.id: STAKING_REWARDS,
    VALIDATOR_COUNT.id: VALIDATOR_COUNT,
    MEV_EXTRACTED.id: MEV_EXTRACTED,
    SANDWICH_ATTACKS.id: SANDWICH_ATTACKS,
    NAKAMOTO_COEFFICIENT.id: NAKAMOTO_COEFFICIENT,
    DELEGATION_CONCENTRATION.id: DELEGATION_CONCENTRATION,
}


def calculate_metric(
    metric_id: str, 
    network_data: Dict[str, Any],
    timestamp: Optional[str] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate a specific metric from raw network data.
    
    Args:
        metric_id: ID of the metric to calculate
        network_data: Raw network data
        timestamp: Optional timestamp for the calculation
        
    Returns:
        Tuple of (metric value, metadata)
    """
    if metric_id not in ALL_METRICS:
        raise ValueError(f"Unknown metric: {metric_id}")
        
    metric = ALL_METRICS[metric_id]
    
    # Use custom calculation function if available
    if metric.calculation_fn:
        return metric.calculation_fn(network_data, timestamp)
        
    # Otherwise, try to extract directly from data
    # This is a simplified approach - in practice, you'd have more robust extraction
    if metric_id in network_data:
        return network_data[metric_id], {}
        
    # Handle specific metrics that need calculation
    if metric_id == "network_utilization":
        if "gas_used" in network_data and "gas_limit" in network_data and network_data["gas_limit"] > 0:
            return (network_data["gas_used"] / network_data["gas_limit"]) * 100, {}
            
    # For metrics not directly available
    return None, {"error": "Metric not available in data"}


def get_metrics_by_category(category: MetricCategory) -> List[Metric]:
    """Get all metrics belonging to a specific category."""
    return [m for m in ALL_METRICS.values() if m.category == category]


def get_available_metrics_for_network(network_id: str) -> List[Metric]:
    """Get metrics available for a specific network."""
    return [m for m in ALL_METRICS.values() if not m.networks or network_id in m.networks]