# Network Performance Benchmarking Tool for Proof-of-Stake Networks

A comprehensive data collection and analysis tool for comparing performance metrics across multiple Proof-of-Stake blockchain networks. This project enables detailed analysis of network performance, economic security metrics, and MEV (Maximal Extractable Value) data.

![image](https://github.com/user-attachments/assets/69377c89-6113-4fbf-a087-a2c38476cfed)


## Features

- **Multi-Network Data Collection**: Collect real-time data from Ethereum, Solana, Cosmos, and more
- **Performance Analysis**: Compare transaction throughput, block times, and finality metrics
- **Economic Security Evaluation**: Analyze staking ratios, validator distributions, and security thresholds
- **MEV Analysis**: Detect and categorize MEV activity across different networks
- **Interactive Dashboard**: Visualize and explore all metrics through a user-friendly Streamlit interface

## Installation

### Prerequisites

- Python 3.8+
- Pip package manager
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Annkkitaaa/pos-network-benchmarking.git
cd pos-network-benchmarking
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file for your API keys:
```
# Ethereum API Keys
ALCHEMY_API_KEY=your_alchemy_key_here
ETHERSCAN_API_KEY=your_etherscan_key_here

# Solana API Keys
SOLSCAN_API_KEY=your_solscan_key_here (optional)
ALCHEMY_SOLANA_KEY=your_alchemy_solana_key_here

# Cosmos API Keys
MINTSCAN_API_KEY=your_mintscan_key_here (optional)
```

## Usage

### Option 1: Using Sample Data (For Testing/Demo)

For a quick start or demonstration, you can use the sample data generator to create realistic network data:

1. Generate sample data:
```bash
python generate_sample_data.py
```

2. Run the dashboard:
```bash
streamlit run app.py
```

3. Access the dashboard in your browser at http://localhost:8501

The sample data generator creates a comprehensive set of realistic metrics for:
- Performance metrics (TPS, block time, finality time)
- Economic security metrics (staking ratio, validator distribution)
- MEV metrics (extracted value, attack patterns)

This approach is perfect for:
- Demonstrating the tool's capabilities
- Development and testing
- Presentations where live API access isn't available
- Teaching about blockchain performance metrics

### Option 2: Using Live Network Data

For real-world analysis, you'll want to collect actual data from blockchain networks:

1. Configure networks in `config/networks.yaml`

2. Run the data collection script:
```bash
python collection.py
```
This will fetch data from all configured networks and store it in the SQLite database.

3. Schedule regular data collection (optional):
   - **Linux/Mac**: Add to crontab
     ```
     */15 * * * * cd /path/to/pos-network-benchmarking && python collection.py
     ```
   - **Windows**: Use Task Scheduler
     - Create a task that runs `collection.py` every 15 minutes

4. Run the dashboard:
```bash
streamlit run app.py
```

5. Access the dashboard in your browser at http://localhost:8501

This approach provides:
- Real-time network performance insights
- Accurate historical trends
- Up-to-date economic security analysis
- Current MEV detection and impact assessment

## Project Structure

```
pos-network-benchmarking/
├── README.md                        # Project documentation
├── requirements.txt                 # Project dependencies
├── app.py                           # Main Streamlit dashboard application
├── collection.py                    # Data collection script for scheduled runs
│
├── config/                          # Configuration files
│   ├── __init__.py
│   ├── app_config.yaml              # Application configuration
│   ├── networks.yaml                # Network endpoints configuration
│   └── load.py                      # Configuration loading utilities
│
├── collectors/                      # Network data collectors
│   ├── __init__.py
│   ├── base.py                      # Base collector abstract class
│   ├── ethereum.py                  # Ethereum-specific collector
│   ├── solana.py                    # Solana-specific collector
│   └── cosmos.py                    # Cosmos-specific collector
│
├── analysis/                        # Analysis modules
│   ├── __init__.py
│   ├── performance.py               # Performance metrics analysis
│   ├── economic.py                  # Economic security analysis
│   └── mev.py                       # MEV analysis
│
├── models/                          # Data models
│   ├── __init__.py
│   ├── network.py                   # Network data models
│   └── metrics.py                   # Metrics definitions
│
├── data/                            # Data storage
│   ├── __init__.py
│   ├── storage.py                   # Data storage interface
│   └── metrics.db                   # SQLite database (auto-created)
│
├── generate_sample_data.py          # Sample data generator
                
```

## Customization

### Adding New Networks

1. Create a new collector class in `collectors/`:
```python
# collectors/new_network.py
from .base import NetworkCollector

class NewNetworkCollector(NetworkCollector):
    """Collector for New Network."""
    
    def __init__(self, config, storage_client=None, timeout=30, max_retries=3):
        super().__init__("new_network", config, storage_client, timeout, max_retries)
        # Network-specific initialization
        
    # Implement required collector methods
    def collect_network_stats(self):
        # Implementation
        
    def collect_validator_metrics(self):
        # Implementation
        
    def collect_performance_metrics(self):
        # Implementation
        
    def collect_economic_metrics(self):
        # Implementation
```

2. Update `config/networks.yaml` to include the new network:
```yaml
networks:
  # Existing networks...
  
  new_network:
    enabled: true
    endpoints:
      - name: "Primary Endpoint"
        url: "https://new-network-api.example.com"
        priority: 1
    explorer:
      url: "https://new-network-explorer.example.com/api"
      api_key: "${NEW_NETWORK_API_KEY}"
    metrics:
      - validator_count
      - total_staked
      - reward_rate
      - avg_block_time
      - transaction_throughput
```

3. Update `collection.py` to initialize the new collector:
```python
if network_config["networks"].get("new_network", {}).get("enabled", False):
    collectors["new_network"] = NewNetworkCollector(
        config=network_config["networks"]["new_network"],
        storage_client=storage
    )
```

### Adding New Metrics

1. Define new metrics in `models/metrics.py`:
```python
NEW_METRIC = Metric(
    id="new_metric",
    name="New Metric Name",
    description="Description of the new metric",
    category=MetricCategory.PERFORMANCE,  # Or appropriate category
    unit="units",
    is_higher_better=True,  # Or False depending on the metric
    tags=["tag1", "tag2"]
)

# Add to ALL_METRICS dictionary
ALL_METRICS["new_metric"] = NEW_METRIC
```

2. Update collector implementations to collect the new metric.

3. Update dashboard visualizations in `app.py` if needed.

## Troubleshooting

### Database Issues

If you encounter issues with the database:

1. Check database existence and permissions:
```bash
ls -la data/metrics.db
```

2. Generate fresh sample data:
```bash
python generate_sample_data.py
```

### API Connection Issues

If collectors can't connect to blockchain APIs:

1. Verify API keys in your `.env` file
2. Check network connectivity to API endpoints
3. Look for rate limiting issues in the logs
4. Try alternative API providers in your configuration

### Dashboard Display Issues

If metrics aren't showing in the dashboard:

1. Verify data exists in the database using the test script
2. Check for errors in the Streamlit console output
3. Try clearing the Streamlit cache by clicking "Clear Cache" in the app menu
4. Restart the Streamlit server

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

