
"""
Persistence layer for state management system
Handles long-term storage, indexing, and retrieval of state data
"""

import asyncio
import json
import sqlite3
import pickle
import gzip
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from pydantic import BaseModel
from loguru import logger

from .state_holder import AutonomousState, StateSnapshot


class StateIndex(BaseModel):
    """Index entry for fast state lookup"""
    
    state_id: str
    iteration_number: int
    timestamp: datetime
    boss_state: str
    system_phase: str
    file_path: str
    compressed: bool = False
    checksum: str
    size_bytes: int
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class StatePersistence:
    """
    Advanced persistence layer for state management
    Handles compression, indexing, archival, and efficient retrieval
    """
    
    def __init__(self, storage_path: str = "state_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Directory structure
        self.states_dir = self.storage_path / "states"
        self.archives_dir = self.storage_path / "archives"  
        self.indexes_dir = self.storage_path / "indexes"
        
        for directory in [self.states_dir, self.archives_dir, self.indexes_dir]:
            directory.mkdir(exist_ok=True)
        
        # Database for metadata and indexing
        self.db_path = self.storage_path / "persistence.db"
        
        # Configuration
        self.compress_after_days = 1  # Compress states older than 1 day
        self.archive_after_days = 30  # Archive states older than 30 days
        self.max_states_per_file = 100  # For batch storage
        
        # Initialize database
        asyncio.create_task(self._initialize_database())
        
        logger.info(f"StatePersistence initialized with storage at {self.storage_path}")
    
    async def _initialize_database(self):
        """Initialize SQLite database for metadata and indexing"""
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Main states table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS state_index (
                state_id TEXT PRIMARY KEY,
                iteration_number INTEGER,
                timestamp TEXT,
                boss_state TEXT,
                system_phase TEXT,
                file_path TEXT,
                compressed BOOLEAN,
                checksum TEXT,
                size_bytes INTEGER,
                created_at TEXT,
                accessed_at TEXT,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        # Partitioned tables by date for better performance
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_summary (
                date TEXT PRIMARY KEY,
                total_states INTEGER,
                total_size_bytes INTEGER,
                avg_iteration_duration REAL,
                success_rate REAL,
                error_count INTEGER,
                unique_boss_states TEXT,
                created_at TEXT
            )
        """)
        
        # Archive tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS archives (
                archive_id TEXT PRIMARY KEY,
                archive_path TEXT,
                start_date TEXT,
                end_date TEXT,
                state_count INTEGER,
                compressed_size_bytes INTEGER,
                created_at TEXT
            )
        """)
        
        # Performance indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON state_index(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_iteration ON state_index(iteration_number)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_boss_state ON state_index(boss_state)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_phase ON state_index(system_phase)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON state_index(file_path)")
        
        conn.commit()
        conn.close()
        
        logger.info("Persistence database initialized")
    
    async def store_state(self, state: AutonomousState, compress: bool = False) -> bool:
        """
        Store a single state with optional compression
        
        Args:
            state: State to store
            compress: Whether to compress the state data
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare file path
            date_str = state.timestamp.strftime("%Y-%m-%d")
            file_name = f"{state.state_id}.{'pkl.gz' if compress else 'pkl'}"
            file_path = self.states_dir / date_str / file_name
            
            # Create date directory
            file_path.parent.mkdir(exist_ok=True)
            
            # Serialize state
            state_data = state.dict()
            serialized_data = pickle.dumps(state_data)
            
            # Compress if requested
            if compress:
                serialized_data = gzip.compress(serialized_data)
            
            # Calculate checksum
            checksum = hashlib.sha256(serialized_data).hexdigest()
            
            # Write to file
            with open(file_path, 'wb') as f:
                f.write(serialized_data)
            
            # Update index
            await self._update_index(StateIndex(
                state_id=state.state_id,
                iteration_number=state.iteration_number,
                timestamp=state.timestamp,
                boss_state=state.boss_state,
                system_phase=state.system_phase,
                file_path=str(file_path),
                compressed=compress,
                checksum=checksum,
                size_bytes=len(serialized_data)
            ))
            
            logger.debug(f"State {state.state_id} stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error storing state {state.state_id}: {e}")
            return False
    
    async def load_state(self, state_id: str) -> Optional[AutonomousState]:
        """
        Load a state by ID
        
        Args:
            state_id: ID of state to load
        
        Returns:
            Loaded state or None if not found
        """
        try:
            # Get index entry
            index_entry = await self._get_index_entry(state_id)
            if not index_entry:
                logger.warning(f"State {state_id} not found in index")
                return None
            
            # Check if file exists
            file_path = Path(index_entry.file_path)
            if not file_path.exists():
                logger.warning(f"State file not found: {file_path}")
                return None
            
            # Read and deserialize
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Decompress if necessary
            if index_entry.compressed:
                data = gzip.decompress(data)
            
            # Verify checksum
            actual_checksum = hashlib.sha256(data if not index_entry.compressed else gzip.compress(pickle.dumps(pickle.loads(data)))).hexdigest()
            if actual_checksum != index_entry.checksum:
                logger.warning(f"Checksum mismatch for state {state_id}")
            
            # Deserialize
            state_data = pickle.loads(data)
            state = AutonomousState(**state_data)
            
            # Update access statistics
            await self._update_access_stats(state_id)
            
            logger.debug(f"State {state_id} loaded successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error loading state {state_id}: {e}")
            return None
    
    async def batch_store_states(self, states: List[AutonomousState], compress_batch: bool = False) -> int:
        """
        Store multiple states in a single operation for efficiency
        
        Args:
            states: List of states to store
            compress_batch: Whether to compress the batch
        
        Returns:
            Number of states successfully stored
        """
        try:
            if not states:
                return 0
            
            stored_count = 0
            
            # Group states by date for organized storage
            date_groups = {}
            for state in states:
                date_str = state.timestamp.strftime("%Y-%m-%d")
                if date_str not in date_groups:
                    date_groups[date_str] = []
                date_groups[date_str].append(state)
            
            # Store each date group
            for date_str, date_states in date_groups.items():
                batch_file_path = self.states_dir / date_str / f"batch_{int(datetime.utcnow().timestamp())}.pkl"
                batch_file_path.parent.mkdir(exist_ok=True)
                
                # Prepare batch data
                batch_data = {
                    "metadata": {
                        "date": date_str,
                        "state_count": len(date_states),
                        "created_at": datetime.utcnow().isoformat(),
                        "compressed": compress_batch
                    },
                    "states": [state.dict() for state in date_states]
                }
                
                # Serialize
                serialized_batch = pickle.dumps(batch_data)
                if compress_batch:
                    serialized_batch = gzip.compress(serialized_batch)
                
                # Write batch file
                with open(batch_file_path, 'wb') as f:
                    f.write(serialized_batch)
                
                # Update index for each state
                for state in date_states:
                    checksum = hashlib.sha256(pickle.dumps(state.dict())).hexdigest()
                    await self._update_index(StateIndex(
                        state_id=state.state_id,
                        iteration_number=state.iteration_number,
                        timestamp=state.timestamp,
                        boss_state=state.boss_state,
                        system_phase=state.system_phase,
                        file_path=str(batch_file_path),
                        compressed=compress_batch,
                        checksum=checksum,
                        size_bytes=len(pickle.dumps(state.dict()))
                    ))
                    stored_count += 1
            
            logger.info(f"Batch stored {stored_count} states across {len(date_groups)} date groups")
            return stored_count
            
        except Exception as e:
            logger.error(f"Error in batch store: {e}")
            return 0
    
    async def query_states_by_criteria(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        boss_states: Optional[List[str]] = None,
        system_phases: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[AutonomousState]:
        """
        Query states based on various criteria
        
        Args:
            start_time: Start time filter
            end_time: End time filter  
            boss_states: List of boss states to filter by
            system_phases: List of system phases to filter by
            limit: Maximum number of results
        
        Returns:
            List of matching states
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Build query
            where_clauses = []
            params = []
            
            if start_time:
                where_clauses.append("timestamp >= ?")
                params.append(start_time.isoformat())
            
            if end_time:
                where_clauses.append("timestamp <= ?")
                params.append(end_time.isoformat())
            
            if boss_states:
                placeholders = ",".join(["?" for _ in boss_states])
                where_clauses.append(f"boss_state IN ({placeholders})")
                params.extend(boss_states)
            
            if system_phases:
                placeholders = ",".join(["?" for _ in system_phases])
                where_clauses.append(f"system_phase IN ({placeholders})")
                params.extend(system_phases)
            
            # Construct SQL
            sql = "SELECT state_id FROM state_index"
            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)
            sql += " ORDER BY timestamp DESC"
            
            if limit:
                sql += f" LIMIT {limit}"
            
            cursor.execute(sql, params)
            results = cursor.fetchall()
            conn.close()
            
            # Load states
            states = []
            for (state_id,) in results:
                state = await self.load_state(state_id)
                if state:
                    states.append(state)
            
            logger.info(f"Query returned {len(states)} states")
            return states
            
        except Exception as e:
            logger.error(f"Error querying states: {e}")
            return []
    
    async def compress_old_states(self, days_old: int = None) -> int:
        """
        Compress states older than specified days
        
        Args:
            days_old: Number of days (defaults to self.compress_after_days)
        
        Returns:
            Number of states compressed
        """
        try:
            if days_old is None:
                days_old = self.compress_after_days
            
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Find uncompressed states older than cutoff
            cursor.execute("""
                SELECT state_id, file_path FROM state_index 
                WHERE timestamp < ? AND compressed = FALSE
            """, (cutoff_date.isoformat(),))
            
            results = cursor.fetchall()
            conn.close()
            
            compressed_count = 0
            
            for state_id, file_path in results:
                try:
                    # Load uncompressed state
                    with open(file_path, 'rb') as f:
                        data = f.read()
                    
                    # Compress
                    compressed_data = gzip.compress(data)
                    
                    # Write compressed version
                    compressed_path = file_path.replace('.pkl', '.pkl.gz')
                    with open(compressed_path, 'wb') as f:
                        f.write(compressed_data)
                    
                    # Update index
                    conn = sqlite3.connect(str(self.db_path))
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE state_index 
                        SET file_path = ?, compressed = TRUE, size_bytes = ?
                        WHERE state_id = ?
                    """, (compressed_path, len(compressed_data), state_id))
                    conn.commit()
                    conn.close()
                    
                    # Remove original file
                    Path(file_path).unlink()
                    
                    compressed_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error compressing state {state_id}: {e}")
            
            logger.info(f"Compressed {compressed_count} old states")
            return compressed_count
            
        except Exception as e:
            logger.error(f"Error compressing old states: {e}")
            return 0
    
    async def archive_old_states(self, days_old: int = None) -> bool:
        """
        Archive states older than specified days into compressed archives
        
        Args:
            days_old: Number of days (defaults to self.archive_after_days)
        
        Returns:
            True if successful
        """
        try:
            if days_old is None:
                days_old = self.archive_after_days
            
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # Query states to archive
            states_to_archive = await self.query_states_by_criteria(
                end_time=cutoff_date,
                limit=None
            )
            
            if not states_to_archive:
                logger.info("No states to archive")
                return True
            
            # Create archive
            archive_id = f"archive_{int(cutoff_date.timestamp())}"
            archive_path = self.archives_dir / f"{archive_id}.tar.gz"
            
            # Prepare archive data
            archive_data = {
                "metadata": {
                    "archive_id": archive_id,
                    "created_at": datetime.utcnow().isoformat(),
                    "cutoff_date": cutoff_date.isoformat(),
                    "state_count": len(states_to_archive)
                },
                "states": [state.dict() for state in states_to_archive]
            }
            
            # Serialize and compress
            serialized = pickle.dumps(archive_data)
            compressed = gzip.compress(serialized)
            
            # Write archive
            with open(archive_path, 'wb') as f:
                f.write(compressed)
            
            # Update archive tracking
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO archives 
                (archive_id, archive_path, start_date, end_date, state_count, compressed_size_bytes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                archive_id,
                str(archive_path),
                states_to_archive[-1].timestamp.isoformat(),  # Oldest
                states_to_archive[0].timestamp.isoformat(),   # Newest
                len(states_to_archive),
                len(compressed),
                datetime.utcnow().isoformat()
            ))
            conn.commit()
            conn.close()
            
            # Remove archived states from regular storage
            for state in states_to_archive:
                await self._remove_state_from_index(state.state_id)
            
            logger.info(f"Archived {len(states_to_archive)} states to {archive_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error archiving states: {e}")
            return False
    
    async def get_storage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Basic statistics
            cursor.execute("SELECT COUNT(*), SUM(size_bytes), MIN(timestamp), MAX(timestamp) FROM state_index")
            total_states, total_size, min_time, max_time = cursor.fetchone()
            
            # Compression statistics
            cursor.execute("SELECT COUNT(*) FROM state_index WHERE compressed = TRUE")
            compressed_states = cursor.fetchone()[0]
            
            # Archive statistics
            cursor.execute("SELECT COUNT(*), SUM(state_count), SUM(compressed_size_bytes) FROM archives")
            archive_count, archived_states, archived_size = cursor.fetchone()
            
            # Daily breakdown
            cursor.execute("""
                SELECT DATE(timestamp) as date, COUNT(*), SUM(size_bytes)
                FROM state_index 
                GROUP BY DATE(timestamp) 
                ORDER BY date DESC 
                LIMIT 30
            """)
            daily_breakdown = cursor.fetchall()
            
            conn.close()
            
            # Calculate directory sizes
            def get_directory_size(path: Path) -> int:
                return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            
            states_size = get_directory_size(self.states_dir)
            archives_size = get_directory_size(self.archives_dir)
            
            return {
                "active_states": {
                    "count": total_states or 0,
                    "total_size_bytes": total_size or 0,
                    "compressed_count": compressed_states or 0,
                    "disk_size_bytes": states_size,
                    "time_range": {
                        "earliest": min_time,
                        "latest": max_time
                    }
                },
                "archives": {
                    "archive_count": archive_count or 0,
                    "archived_states_count": archived_states or 0,
                    "total_size_bytes": archived_size or 0,
                    "disk_size_bytes": archives_size
                },
                "daily_breakdown": [
                    {"date": date, "count": count, "size_bytes": size}
                    for date, count, size in daily_breakdown
                ],
                "compression_ratio": (compressed_states / total_states * 100) if total_states else 0,
                "total_disk_usage_bytes": states_size + archives_size
            }
            
        except Exception as e:
            logger.error(f"Error getting storage statistics: {e}")
            return {"error": str(e)}
    
    async def _update_index(self, index_entry: StateIndex):
        """Update the state index with new entry"""
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO state_index 
            (state_id, iteration_number, timestamp, boss_state, system_phase,
             file_path, compressed, checksum, size_bytes, created_at, accessed_at, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT access_count FROM state_index WHERE state_id = ?), 0))
        """, (
            index_entry.state_id,
            index_entry.iteration_number,
            index_entry.timestamp.isoformat(),
            index_entry.boss_state,
            index_entry.system_phase,
            index_entry.file_path,
            index_entry.compressed,
            index_entry.checksum,
            index_entry.size_bytes,
            datetime.utcnow().isoformat(),
            datetime.utcnow().isoformat(),
            index_entry.state_id  # For the COALESCE subquery
        ))
        
        conn.commit()
        conn.close()
    
    async def _get_index_entry(self, state_id: str) -> Optional[StateIndex]:
        """Get index entry for a state"""
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT state_id, iteration_number, timestamp, boss_state, system_phase,
                       file_path, compressed, checksum, size_bytes
                FROM state_index 
                WHERE state_id = ?
            """, (state_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return StateIndex(
                    state_id=result[0],
                    iteration_number=result[1],
                    timestamp=datetime.fromisoformat(result[2]),
                    boss_state=result[3],
                    system_phase=result[4],
                    file_path=result[5],
                    compressed=bool(result[6]),
                    checksum=result[7],
                    size_bytes=result[8]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting index entry for {state_id}: {e}")
            return None
    
    async def _update_access_stats(self, state_id: str):
        """Update access statistics for a state"""
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE state_index 
                SET accessed_at = ?, access_count = access_count + 1
                WHERE state_id = ?
            """, (datetime.utcnow().isoformat(), state_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Error updating access stats for {state_id}: {e}")
    
    async def _remove_state_from_index(self, state_id: str):
        """Remove a state from the index (used during archival)"""
        
        try:
            # Get file path first to clean up file
            index_entry = await self._get_index_entry(state_id)
            if index_entry and Path(index_entry.file_path).exists():
                Path(index_entry.file_path).unlink()
            
            # Remove from index
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM state_index WHERE state_id = ?", (state_id,))
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Error removing state {state_id} from index: {e}")
