"""
Data storage for network performance metrics.
"""
import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class DataStorage:
    """Storage interface for network performance metrics."""

    def __init__(self, db_path: str = "data/metrics.db"):
        """
        Initialize the data storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_directory()
        self._init_database()
        
    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
    def _init_database(self):
        """Initialize database tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create metrics table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                network_id TEXT NOT NULL,
                metric_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                value REAL,
                metadata TEXT,
                category TEXT
            )
            ''')
            
            # Create network_stats table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS network_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                network_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                stats TEXT NOT NULL
            )
            ''')
            
            # Create validator_metrics table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS validator_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                network_id TEXT NOT NULL,
                validator_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metrics TEXT NOT NULL
            )
            ''')
            
            # Create indices for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_network_time ON metrics(network_id, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_category ON metrics(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_network_stats_time ON network_stats(network_id, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_validator_metrics ON validator_metrics(network_id, validator_id, timestamp)')
            
            conn.commit()
            logger.info("Database initialized successfully")
            
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {str(e)}")
            
        finally:
            if conn:
                conn.close()
                
    def save_metrics(self, metrics_data: Dict[str, Any]) -> bool:
        """
        Save collected metrics to storage.
        
        Args:
            metrics_data: Dictionary of collected metrics
            
        Returns:
            Success flag
        """
        try:
            network_id = metrics_data.get("network_id")
            timestamp = metrics_data.get("timestamp")
            
            if not network_id or not timestamp:
                logger.error("Missing required fields (network_id, timestamp)")
                return False
                
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Save network stats
            if "network_stats" in metrics_data:
                cursor.execute(
                    'INSERT INTO network_stats (network_id, timestamp, stats) VALUES (?, ?, ?)',
                    (network_id, timestamp, json.dumps(metrics_data["network_stats"]))
                )
                
                # Also save individual metrics for easy querying
                for metric_id, value in metrics_data["network_stats"].items():
                    if isinstance(value, (int, float)):
                        cursor.execute(
                            'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                            (network_id, metric_id, timestamp, value, "network")
                        )
            
            # Save performance metrics
            if "performance_metrics" in metrics_data:
                for metric_id, value in metrics_data["performance_metrics"].items():
                    if isinstance(value, (int, float)):
                        cursor.execute(
                            'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                            (network_id, metric_id, timestamp, value, "performance")
                        )
            
            # Save economic metrics
            if "economic_metrics" in metrics_data:
                for metric_id, value in metrics_data["economic_metrics"].items():
                    if isinstance(value, (int, float)):
                        cursor.execute(
                            'INSERT INTO metrics (network_id, metric_id, timestamp, value, category) VALUES (?, ?, ?, ?, ?)',
                            (network_id, metric_id, timestamp, value, "economic")
                        )
            
            # Save validator metrics
            if "validator_metrics" in metrics_data:
                for validator in metrics_data["validator_metrics"]:
                    validator_id = validator.get("pubkey") or validator.get("address") or validator.get("validator_id")
                    if validator_id:
                        cursor.execute(
                            'INSERT INTO validator_metrics (network_id, validator_id, timestamp, metrics) VALUES (?, ?, ?, ?)',
                            (network_id, validator_id, timestamp, json.dumps(validator))
                        )
            
            conn.commit()
            logger.info(f"Saved metrics for {network_id} at {timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            return False
            
        finally:
            if conn:
                conn.close()
                
    def get_metrics(
        self,
        network_id: str,
        metric_ids: Optional[List[str]] = None,
        metric_category: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Retrieve metrics from storage.
        
        Args:
            network_id: Network ID
            metric_ids: Optional list of specific metric IDs to retrieve
            metric_category: Optional category filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of records to retrieve
            
        Returns:
            List of metric records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            cursor = conn.cursor()
            
            query = 'SELECT * FROM metrics WHERE network_id = ?'
            params = [network_id]
            
            if metric_ids:
                placeholders = ','.join(['?'] * len(metric_ids))
                query += f' AND metric_id IN ({placeholders})'
                params.extend(metric_ids)
                
            if metric_category:
                query += ' AND category = ?'
                params.append(metric_category)
                
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time.isoformat())
                
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time.isoformat())
                
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            results = []
            for row in rows:
                results.append(dict(row))
                
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving metrics: {str(e)}")
            return []
            
        finally:
            if conn:
                conn.close()
                
    def get_network_stats(
        self,
        network_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve network stats from storage.
        
        Args:
            network_id: Network ID
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of records to retrieve
            
        Returns:
            List of network stat records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = 'SELECT * FROM network_stats WHERE network_id = ?'
            params = [network_id]
            
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time.isoformat())
                
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time.isoformat())
                
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Parse JSON stats
            results = []
            for row in rows:
                record = dict(row)
                record['stats'] = json.loads(record['stats'])
                results.append(record)
                
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving network stats: {str(e)}")
            return []
            
        finally:
            if conn:
                conn.close()
                
    def get_validator_metrics(
        self,
        network_id: str,
        validator_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve validator metrics from storage.
        
        Args:
            network_id: Network ID
            validator_id: Optional validator ID filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of records to retrieve
            
        Returns:
            List of validator metric records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = 'SELECT * FROM validator_metrics WHERE network_id = ?'
            params = [network_id]
            
            if validator_id:
                query += ' AND validator_id = ?'
                params.append(validator_id)
                
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time.isoformat())
                
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time.isoformat())
                
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Parse JSON metrics
            results = []
            for row in rows:
                record = dict(row)
                record['metrics'] = json.loads(record['metrics'])
                results.append(record)
                
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving validator metrics: {str(e)}")
            return []
            
        finally:
            if conn:
                conn.close()
                
    def get_metrics_dataframe(
        self,
        network_id: str,
        metric_ids: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Retrieve metrics as a pandas DataFrame with a timestamp index.
        
        Args:
            network_id: Network ID
            metric_ids: List of metric IDs to retrieve
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            DataFrame with metrics data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Build query for selected metrics
            placeholders = ','.join(['?'] * len(metric_ids))
            query = f'''
            SELECT timestamp, metric_id, value
            FROM metrics 
            WHERE network_id = ? AND metric_id IN ({placeholders})
            '''
            params = [network_id] + metric_ids
            
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time.isoformat())
                
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time.isoformat())
                
            query += ' ORDER BY timestamp'
            
            # Load data into DataFrame
            df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                return pd.DataFrame()
                
            # Pivot to get metrics as columns
            pivot_df = df.pivot(index='timestamp', columns='metric_id', values='value')
            
            # Convert timestamp to datetime index
            pivot_df.index = pd.to_datetime(pivot_df.index)
            
            return pivot_df
            
        except Exception as e:
            logger.error(f"Error creating metrics DataFrame: {str(e)}")
            return pd.DataFrame()
            
        finally:
            if conn:
                conn.close()
                
    def get_network_list(self) -> List[str]:
        """
        Get list of all networks in the database.
        
        Returns:
            List of network IDs
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT DISTINCT network_id FROM metrics')
            networks = [row[0] for row in cursor.fetchall()]
            
            return networks
            
        except Exception as e:
            logger.error(f"Error getting network list: {str(e)}")
            return []
            
        finally:
            if conn:
                conn.close()
                
    def get_available_metrics(self, network_id: str) -> Dict[str, List[str]]:
        """
        Get available metrics for a specific network.
        
        Args:
            network_id: Network ID
            
        Returns:
            Dictionary of metric categories and their available metric IDs
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT DISTINCT category, metric_id FROM metrics WHERE network_id = ?',
                (network_id,)
            )
            
            results = {}
            for category, metric_id in cursor.fetchall():
                if category not in results:
                    results[category] = []
                results[category].append(metric_id)
                
            return results
            
        except Exception as e:
            logger.error(f"Error getting available metrics: {str(e)}")
            return {}
            
        finally:
            if conn:
                conn.close()