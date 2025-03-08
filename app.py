"""
Network Performance Benchmarking Tool - Streamlit Dashboard
"""
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analysis.economic import EconomicAnalyzer
from analysis.mev import MEVAnalyzer
from analysis.performance import PerformanceAnalyzer
from collectors.ethereum import EthereumCollector
from collectors.solana import SolanaCollector
from collectors.cosmos import CosmosCollector
from config.load import load_config
from data.storage import DataStorage
from models.metrics import MetricCategory, get_metrics_by_category

# Page configuration
st.set_page_config(
    page_title="PoS Network Benchmarking Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
@st.cache_resource
def load_app_config():
    """Load application configuration."""
    return load_config("config/app_config.yaml")


@st.cache_resource
def load_network_config():
    """Load network configuration."""
    return load_config("config/networks.yaml")


@st.cache_resource
def initialize_storage():
    """Initialize data storage client."""
    return DataStorage()


@st.cache_resource
def initialize_collectors(network_config):
    """Initialize network collectors."""
    collectors = {}
    storage = initialize_storage()
    
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
        
    return collectors


@st.cache_resource
def initialize_analyzers():
    """Initialize data analyzers."""
    storage = initialize_storage()
    
    analyzers = {
        "performance": PerformanceAnalyzer(storage_client=storage),
        "economic": EconomicAnalyzer(storage_client=storage),
        "mev": MEVAnalyzer(storage_client=storage)
    }
    
    return analyzers


# Load configurations and initialize components
app_config = load_app_config()
network_config = load_network_config()
storage = initialize_storage()
collectors = initialize_collectors(network_config)
analyzers = initialize_analyzers()

# Sidebar
st.sidebar.title("PoS Network Benchmarking")

# Network selection
available_networks = list(collectors.keys())
selected_networks = st.sidebar.multiselect(
    "Select Networks to Compare",
    options=available_networks,
    default=available_networks[:2] if len(available_networks) >= 2 else available_networks
)

# Time range selection
time_range_options = {
    "Last 24 Hours": timedelta(days=1),
    "Last 7 Days": timedelta(days=7),
    "Last 30 Days": timedelta(days=30),
    "Last 90 Days": timedelta(days=90),
    "Custom Range": "custom"
}

selected_time_range = st.sidebar.selectbox(
    "Select Time Range",
    options=list(time_range_options.keys()),
    index=1  # Default to 7 days
)

if selected_time_range == "Custom Range":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
        
    start_time = datetime.combine(start_date, datetime.min.time())
    end_time = datetime.combine(end_date, datetime.max.time())
else:
    end_time = datetime.now()
    start_time = end_time - time_range_options[selected_time_range]

# Analysis type selection
analysis_types = {
    "Performance Analysis": "performance",
    "Economic Security": "economic",
    "MEV Analysis": "mev",
    "Comprehensive Comparison": "comprehensive"
}

selected_analysis = st.sidebar.radio(
    "Analysis Type",
    options=list(analysis_types.keys())
)

# Metric selection
if selected_analysis != "Comprehensive Comparison":
    analysis_category = MetricCategory(analysis_types[selected_analysis].upper())
    available_metrics = get_metrics_by_category(analysis_category)
    metric_options = {m.name: m.id for m in available_metrics}
    
    selected_metrics = st.sidebar.multiselect(
        "Select Metrics to Display",
        options=list(metric_options.keys()),
        default=list(metric_options.keys())[:min(5, len(metric_options))]
    )
    selected_metric_ids = [metric_options[name] for name in selected_metrics]
else:
    selected_metrics = []
    selected_metric_ids = []

# Button to refresh data
if st.sidebar.button("🔄 Refresh Data"):
    st.experimental_rerun()

# Dashboard header
st.title("Proof-of-Stake Network Benchmarking Tool")
st.markdown(f"Analyzing {', '.join(selected_networks)} from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")

# Check if networks are selected
if not selected_networks:
    st.warning("Please select at least one network to analyze.")
    st.stop()
    
# Load data for selected networks and time range
@st.cache_data(ttl=3600)
def load_analysis_data(networks, start, end, analysis_type):
    """Load and cache analysis data."""
    if analysis_type == "performance":
        return analyzers["performance"].load_data(networks, start, end)
    elif analysis_type == "economic":
        return analyzers["economic"].load_data(networks, start, end)
    elif analysis_type == "mev":
        return analyzers["mev"].load_data(networks, start, end)
    elif analysis_type == "comprehensive":
        # Load all types of data
        performance_data = analyzers["performance"].load_data(networks, start, end)
        economic_data = analyzers["economic"].load_data(networks, start, end)
        mev_data = analyzers["mev"].load_data(networks, start, end)
        
        # Combine data for each network
        combined_data = {}
        for network in networks:
            network_dfs = []
            
            if network in performance_data and not performance_data[network].empty:
                network_dfs.append(performance_data[network])
                
            if network in economic_data and not economic_data[network].empty:
                network_dfs.append(economic_data[network])
                
            if network in mev_data and not mev_data[network].empty:
                network_dfs.append(mev_data[network])
                
            if network_dfs:
                # Combine all DataFrames for this network
                combined_data[network] = pd.concat(network_dfs, axis=1, join="outer")
                
        return combined_data
    
    return {}

# Load data
with st.spinner("Loading data for analysis..."):
    analysis_type = analysis_types[selected_analysis]
    network_data = load_analysis_data(selected_networks, start_time, end_time, analysis_type)

# Check if data was loaded successfully
if not network_data or all(df.empty for df in network_data.values() if df is not None):
    st.error("No data available for the selected networks and time range.")
    st.stop()

# Display analysis based on selection
if selected_analysis == "Performance Analysis":
    st.header("Performance Analysis")
    
    # Performance metrics over time
    st.subheader("Performance Metrics Over Time")
    
    for metric_id in selected_metric_ids:
        metric = next((m for m in get_metrics_by_category(MetricCategory.PERFORMANCE) if m.id == metric_id), None)
        if not metric:
            continue
            
        fig = go.Figure()
        
        for network, df in network_data.items():
            if df is not None and not df.empty and metric_id in df.columns:
                # Reset index to get timestamp as column
                plot_df = df.reset_index()
                fig.add_trace(go.Scatter(
                    x=plot_df['timestamp'],
                    y=plot_df[metric_id],
                    mode='lines',
                    name=network,
                    hovertemplate=f"{network}: %{{y:.2f}} {metric.unit}<extra></extra>"
                ))
                
        fig.update_layout(
            title=f"{metric.name} over Time",
            xaxis_title="Time",
            yaxis_title=f"{metric.name} ({metric.unit})",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance comparison
    st.subheader("Network Performance Comparison")
    
    # Create summary statistics
    summary_stats = analyzers["performance"].calculate_summary_stats(network_data)
    
    if summary_stats:
        # Create comparison table
        comparison_data = []
        
        for network, stats in summary_stats.items():
            row = {"Network": network}
            
            for metric_id in selected_metric_ids:
                metric = next((m for m in get_metrics_by_category(MetricCategory.PERFORMANCE) if m.id == metric_id), None)
                if not metric:
                    continue
                    
                avg_key = f"{metric_id}_mean"
                if avg_key in stats:
                    row[metric.name] = stats[avg_key]
                    
            comparison_data.append(row)
            
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df.set_index("Network"), use_container_width=True)
            
            # Radar chart for performance comparison
            if len(selected_networks) > 1 and len(selected_metric_ids) > 2:
                st.subheader("Performance Radar Chart")
                
                # Normalize metrics for radar chart
                radar_data = []
                
                for network, stats in summary_stats.items():
                    row = {"Network": network}
                    
                    for metric_id in selected_metric_ids:
                        metric = next((m for m in get_metrics_by_category(MetricCategory.PERFORMANCE) if m.id == metric_id), None)
                        if not metric:
                            continue
                            
                        avg_key = f"{metric_id}_mean"
                        if avg_key in stats:
                            # Get metric value
                            value = stats[avg_key]
                            
                            # Get all values for this metric across networks for normalization
                            all_values = [s.get(avg_key, 0) for s in summary_stats.values() if avg_key in s]
                            
                            if all_values:
                                min_val = min(all_values)
                                max_val = max(all_values)
                                
                                if min_val == max_val:
                                    normalized = 1.0
                                else:
                                    # Normalize based on whether higher is better
                                    if metric.is_higher_better:
                                        normalized = (value - min_val) / (max_val - min_val)
                                    else:
                                        normalized = (max_val - value) / (max_val - min_val)
                                        
                                row[metric.name] = normalized
                                
                    radar_data.append(row)
                    
                if radar_data:
                    radar_df = pd.DataFrame(radar_data)
                    
                    # Create radar chart
                    fig = go.Figure()
                    
                    categories = [col for col in radar_df.columns if col != "Network"]
                    
                    for _, row in radar_df.iterrows():
                        fig.add_trace(go.Scatterpolar(
                            r=[row[cat] for cat in categories],
                            theta=categories,
                            fill='toself',
                            name=row["Network"]
                        ))
                        
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        showlegend=True,
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Performance anomalies
    st.subheader("Performance Anomalies")
    
    anomalies = analyzers["performance"].detect_performance_anomalies(network_data)
    
    if anomalies:
        for network, network_anomalies in anomalies.items():
            if network_anomalies:
                st.markdown(f"**{network}**")
                
                anomaly_data = []
                for anomaly in network_anomalies:
                    metric = next((m for m in get_metrics_by_category(MetricCategory.PERFORMANCE) 
                                 if m.id == anomaly["metric"]), None)
                    if not metric:
                        continue
                        
                    anomaly_data.append({
                        "Timestamp": anomaly["timestamp"],
                        "Metric": metric.name,
                        "Value": anomaly["value"],
                        "Expected": anomaly["expected"],
                        "Deviation": anomaly["deviation"],
                        "Z-Score": anomaly["z_score"]
                    })
                    
                if anomaly_data:
                    anomaly_df = pd.DataFrame(anomaly_data)
                    st.dataframe(anomaly_df, use_container_width=True)
                else:
                    st.info("No anomalies detected.")
            else:
                st.info(f"No anomalies detected for {network}.")
    else:
        st.info("No anomalies detected across any networks.")
        
    # Performance score
    st.subheader("Overall Performance Score")
    
    performance_scores = analyzers["performance"].calculate_performance_score(network_data)
    
    if performance_scores:
        # Create bar chart of scores
        score_data = {"Network": [], "Score": []}
        
        for network, score in performance_scores.items():
            score_data["Network"].append(network)
            score_data["Score"].append(score)
            
        score_df = pd.DataFrame(score_data)
        
        fig = px.bar(
            score_df,
            x="Network",
            y="Score",
            color="Network",
            text="Score",
            labels={"Score": "Performance Score (0-1)"},
            height=400
        )
        
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(
            title="Overall Performance Score by Network",
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
# Economic Security Analysis
elif selected_analysis == "Economic Security":
    st.header("Economic Security Analysis")
    
    # Economic metrics over time
    st.subheader("Economic Metrics Over Time")
    
    for metric_id in selected_metric_ids:
        metric = next((m for m in get_metrics_by_category(MetricCategory.ECONOMIC) if m.id == metric_id), None)
        if not metric:
            continue
            
        fig = go.Figure()
        
        for network, df in network_data.items():
            if df is not None and not df.empty and metric_id in df.columns:
                # Reset index to get timestamp as column
                plot_df = df.reset_index()
                fig.add_trace(go.Scatter(
                    x=plot_df['timestamp'],
                    y=plot_df[metric_id],
                    mode='lines',
                    name=network,
                    hovertemplate=f"{network}: %{{y:.2f}} {metric.unit}<extra></extra>"
                ))
                
        fig.update_layout(
            title=f"{metric.name} over Time",
            xaxis_title="Time",
            yaxis_title=f"{metric.name} ({metric.unit})",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Staking distribution analysis
    st.subheader("Staking Distribution Analysis")
    
    # Placeholder for staking distribution visualization
    # In a real implementation, you would fetch and display validator stake distribution
    
    # Security threshold analysis
    st.subheader("Economic Security Thresholds")
    
    # Calculate Nakamoto coefficients
    # This is a simplified placeholder
    security_data = {"Network": [], "Nakamoto Coefficient": [], "Stake to Attack (%)": [], "Cost to Attack ($)": []}
    
    for network, df in network_data.items():
        if df is not None and not df.empty:
            # These would be calculated from actual data in a real implementation
            nakamoto = np.random.randint(10, 100)
            stake_pct = 33.3
            cost = np.random.randint(100000000, 10000000000) / 1000000
            
            security_data["Network"].append(network)
            security_data["Nakamoto Coefficient"].append(nakamoto)
            security_data["Stake to Attack (%)"].append(stake_pct)
            security_data["Cost to Attack ($)"].append(cost)
            
    if security_data["Network"]:
        security_df = pd.DataFrame(security_data)
        st.dataframe(security_df.set_index("Network"), use_container_width=True)
        
        # Bar chart of Nakamoto coefficients
        fig = px.bar(
            security_df,
            x="Network",
            y="Nakamoto Coefficient",
            color="Network",
            text="Nakamoto Coefficient",
            labels={"Nakamoto Coefficient": "Minimum entities to control 33%"},
            height=400
        )
        
        fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        fig.update_layout(title="Nakamoto Coefficient by Network")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Bar chart of cost to attack
        fig = px.bar(
            security_df,
            x="Network",
            y="Cost to Attack ($)",
            color="Network",
            text="Cost to Attack ($)",
            labels={"Cost to Attack ($)": "Cost to Attack (Million $)"},
            height=400
        )
        
        fig.update_traces(texttemplate='%{text:.1f}M', textposition='outside')
        fig.update_layout(title="Estimated Cost to Attack (33%) by Network")
        
        st.plotly_chart(fig, use_container_width=True)
        
# MEV Analysis
elif selected_analysis == "MEV Analysis":
    st.header("MEV Analysis")
    
    # MEV metrics over time
    st.subheader("MEV Metrics Over Time")
    
    for metric_id in selected_metric_ids:
        metric = next((m for m in get_metrics_by_category(MetricCategory.MEV) if m.id == metric_id), None)
        if not metric:
            continue
            
        fig = go.Figure()
        
        for network, df in network_data.items():
            if df is not None and not df.empty and metric_id in df.columns:
                # Reset index to get timestamp as column
                plot_df = df.reset_index()
                fig.add_trace(go.Scatter(
                    x=plot_df['timestamp'],
                    y=plot_df[metric_id],
                    mode='lines',
                    name=network,
                    hovertemplate=f"{network}: %{{y:.2f}} {metric.unit}<extra></extra>"
                ))
                
        fig.update_layout(
            title=f"{metric.name} over Time",
            xaxis_title="Time",
            yaxis_title=f"{metric.name} ({metric.unit})",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # MEV extraction by type
    st.subheader("MEV Extraction by Type")
    
    # Placeholder for MEV extraction by type visualization
    # In a real implementation, you would categorize and display MEV by type
    
    mev_types = ["Sandwich Attacks", "Arbitrage", "Liquidations", "Other"]
    
    mev_data = {"Network": [], "Type": [], "Value Extracted": []}
    
    for network in selected_networks:
        # These would be calculated from actual data in a real implementation
        for mev_type in mev_types:
            value = np.random.randint(100000, 10000000) / 1000
            
            mev_data["Network"].append(network)
            mev_data["Type"].append(mev_type)
            mev_data["Value Extracted"].append(value)
            
    if mev_data["Network"]:
        mev_df = pd.DataFrame(mev_data)
        
        # Stacked bar chart of MEV by type
        fig = px.bar(
            mev_df,
            x="Network",
            y="Value Extracted",
            color="Type",
            barmode="stack",
            labels={"Value Extracted": "Value Extracted (K $)"},
            height=500
        )
        
        fig.update_layout(title="MEV Extraction by Type and Network")
        
        st.plotly_chart(fig, use_container_width=True)
        
    # MEV impact on users
    st.subheader("MEV Impact on User Experience")
    
    # Placeholder for user impact visualization
    # In a real implementation, you would calculate and display metrics like price impact
    
# Comprehensive Comparison
else:
    st.header("Comprehensive Network Comparison")
    
    # Overall network metrics dashboard
    st.subheader("Key Metrics Dashboard")
    
    # Create metric cards
    col1, col2, col3 = st.columns(3)
    
    # Performance metrics
    with col1:
        st.markdown("### Performance")
        
        for network in selected_networks:
            if network in network_data and network_data[network] is not None and not network_data[network].empty:
                df = network_data[network]
                
                # TPS
                if "tps" in df.columns:
                    tps = df["tps"].mean()
                    st.metric(f"{network} - TPS", f"{tps:.2f} tx/s")
                
                # Block time
                if "block_time" in df.columns:
                    block_time = df["block_time"].mean()
                    st.metric(f"{network} - Block Time", f"{block_time:.2f} s")
                    
                # Finality
                if "finality_time" in df.columns:
                    finality = df["finality_time"].mean()
                    st.metric(f"{network} - Finality", f"{finality:.2f} s")
    
    # Economic metrics
    with col2:
        st.markdown("### Economics")
        
        for network in selected_networks:
            if network in network_data and network_data[network] is not None and not network_data[network].empty:
                df = network_data[network]
                
                # Staking ratio
                if "staking_ratio" in df.columns:
                    ratio = df["staking_ratio"].mean() * 100
                    st.metric(f"{network} - Staked", f"{ratio:.2f}%")
                
                # Rewards
                if "staking_rewards" in df.columns:
                    rewards = df["staking_rewards"].mean()
                    st.metric(f"{network} - APR", f"{rewards:.2f}%")
                    
                # Validators
                if "validator_count" in df.columns:
                    validators = df["validator_count"].mean()
                    st.metric(f"{network} - Validators", f"{validators:.0f}")
    
    # MEV metrics
    with col3:
        st.markdown("### MEV")
        
        for network in selected_networks:
            if network in network_data and network_data[network] is not None and not network_data[network].empty:
                df = network_data[network]
                
                # MEV extracted
                if "mev_extracted" in df.columns:
                    mev = df["mev_extracted"].mean() / 1000
                    st.metric(f"{network} - MEV/day", f"${mev:.2f}K")
                
                # Sandwich attacks
                if "sandwich_attacks" in df.columns:
                    attacks = df["sandwich_attacks"].mean()
                    st.metric(f"{network} - Sandwich Attacks", f"{attacks:.0f}/day")
    
    # Comprehensive comparison charts
    st.subheader("Network Comparison Over Time")
    
    # Create tabs for different metric categories
    tabs = st.tabs(["Performance", "Economics", "MEV & Security"])
    
    # Performance tab
    with tabs[0]:
        # Performance metrics over time
        performance_metrics = [m for m in get_metrics_by_category(MetricCategory.PERFORMANCE)]
        selected_perf_metrics = performance_metrics[:min(3, len(performance_metrics))]
        
        for metric in selected_perf_metrics:
            fig = go.Figure()
            
            for network, df in network_data.items():
                if df is not None and not df.empty and metric.id in df.columns:
                    # Reset index to get timestamp as column
                    plot_df = df.reset_index()
                    fig.add_trace(go.Scatter(
                        x=plot_df['timestamp'],
                        y=plot_df[metric.id],
                        mode='lines',
                        name=network,
                        hovertemplate=f"{network}: %{{y:.2f}} {metric.unit}<extra></extra>"
                    ))
                    
            fig.update_layout(
                title=f"{metric.name} over Time",
                xaxis_title="Time",
                yaxis_title=f"{metric.name} ({metric.unit})",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Economics tab
    with tabs[1]:
        # Economic metrics over time
        economic_metrics = [m for m in get_metrics_by_category(MetricCategory.ECONOMIC)]
        selected_econ_metrics = economic_metrics[:min(3, len(economic_metrics))]
        
        for metric in selected_econ_metrics:
            fig = go.Figure()
            
            for network, df in network_data.items():
                if df is not None and not df.empty and metric.id in df.columns:
                    # Reset index to get timestamp as column
                    plot_df = df.reset_index()
                    fig.add_trace(go.Scatter(
                        x=plot_df['timestamp'],
                        y=plot_df[metric.id],
                        mode='lines',
                        name=network,
                        hovertemplate=f"{network}: %{{y:.2f}} {metric.unit}<extra></extra>"
                    ))
                    
            fig.update_layout(
                title=f"{metric.name} over Time",
                xaxis_title="Time",
                yaxis_title=f"{metric.name} ({metric.unit})",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # MEV & Security tab
    with tabs[2]:
        # MEV metrics over time
        mev_metrics = [m for m in get_metrics_by_category(MetricCategory.MEV)]
        selected_mev_metrics = mev_metrics[:min(2, len(mev_metrics))]
        
        for metric in selected_mev_metrics:
            fig = go.Figure()
            
            for network, df in network_data.items():
                if df is not None and not df.empty and metric.id in df.columns:
                    # Reset index to get timestamp as column
                    plot_df = df.reset_index()
                    fig.add_trace(go.Scatter(
                        x=plot_df['timestamp'],
                        y=plot_df[metric.id],
                        mode='lines',
                        name=network,
                        hovertemplate=f"{network}: %{{y:.2f}} {metric.unit}<extra></extra>"
                    ))
                    
            fig.update_layout(
                title=f"{metric.name} over Time",
                xaxis_title="Time",
                yaxis_title=f"{metric.name} ({metric.unit})",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        # Security metrics over time
        security_metrics = [m for m in get_metrics_by_category(MetricCategory.SECURITY)]
        selected_sec_metrics = security_metrics[:min(2, len(security_metrics))]
        
        for metric in selected_sec_metrics:
            fig = go.Figure()
            
            for network, df in network_data.items():
                if df is not None and not df.empty and metric.id in df.columns:
                    # Reset index to get timestamp as column
                    plot_df = df.reset_index()
                    fig.add_trace(go.Scatter(
                        x=plot_df['timestamp'],
                        y=plot_df[metric.id],
                        mode='lines',
                        name=network,
                        hovertemplate=f"{network}: %{{y:.2f}} {metric.unit}<extra></extra>"
                    ))
                    
            fig.update_layout(
                title=f"{metric.name} over Time",
                xaxis_title="Time",
                yaxis_title=f"{metric.name} ({metric.unit})",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Overall network ranking
    st.subheader("Overall Network Ranking")
    
    # Calculate overall scores
    # This is a simplified placeholder
    overall_data = {"Network": [], "Performance Score": [], "Economics Score": [], "Security Score": [], "Overall Score": []}
    
    for network in selected_networks:
        # These would be calculated from actual data in a real implementation
        perf_score = np.random.random()
        econ_score = np.random.random()
        sec_score = np.random.random()
        overall = (perf_score + econ_score + sec_score) / 3
        
        overall_data["Network"].append(network)
        overall_data["Performance Score"].append(perf_score)
        overall_data["Economics Score"].append(econ_score)
        overall_data["Security Score"].append(sec_score)
        overall_data["Overall Score"].append(overall)
        
    if overall_data["Network"]:
        overall_df = pd.DataFrame(overall_data)
        overall_df = overall_df.sort_values("Overall Score", ascending=False)
        
        st.dataframe(overall_df.set_index("Network").style.format("{:.3f}"), use_container_width=True)
        
        # Radar chart for overall comparison
        fig = go.Figure()
        
        categories = ["Performance Score", "Economics Score", "Security Score"]
        
        for _, row in overall_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[cat] for cat in categories],
                theta=categories,
                fill='toself',
                name=row["Network"]
            ))
            
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Bar chart of overall scores
        fig = px.bar(
            overall_df,
            x="Network",
            y="Overall Score",
            color="Network",
            text="Overall Score",
            height=400
        )
        
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(
            title="Overall Network Score",
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Network Performance Benchmarking Tool - Build your own version of this project on [GitHub](https://github.com/yourusername/pos-network-benchmarking)")