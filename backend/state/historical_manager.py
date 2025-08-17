
"""
Historical state management for long-term storage and retrieval
Provides sophisticated querying and pattern analysis of historical states
"""

import asyncio
import json
import sqlite3
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field
from loguru import logger
from pathlib import Path

from .state_holder import AutonomousState, StateSnapshot


class StateQuery(BaseModel):
    """Query structure for historical state retrieval"""
    
    # Time-based filters
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    last_n_hours: Optional[int] = None
    last_n_days: Optional[int] = None
    
    # State-based filters
    boss_states: Optional[List[str]] = None
    system_phases: Optional[List[str]] = None
    iteration_range: Optional[Tuple[int, int]] = None
    
    # Performance filters
    min_confidence: Optional[float] = None
    max_error_count: Optional[int] = None
    success_only: bool = False
    
    # Content filters
    has_decision: bool = False
    agent_count_range: Optional[Tuple[int, int]] = None
    task_count_range: Optional[Tuple[int, int]] = None
    
    # Search terms
    search_terms: Optional[List[str]] = None
    
    # Result options
    limit: Optional[int] = 100
    order_by: str = "timestamp"
    ascending: bool = False


class StatePattern(BaseModel):
    """Identified pattern in historical states"""
    
    pattern_id: str
    pattern_type: str = Field(..., description="Type of pattern (recurring, sequential, conditional)")
    description: str = Field(..., description="Human-readable description of pattern")
    
    # Pattern characteristics
    frequency: float = Field(..., description="How often this pattern occurs")
    confidence: float = Field(..., description="Confidence in pattern identification")
    states_involved: int = Field(..., description="Number of states this pattern spans")
    
    # Trigger conditions
    trigger_conditions: List[str] = Field(default_factory=list)
    preconditions: List[str] = Field(default_factory=list)
    
    # Outcomes
    typical_outcomes: List[str] = Field(default_factory=list)
    success_rate: float = Field(..., description="Success rate when this pattern occurs")
    
    # Example occurrences
    example_state_ids: List[str] = Field(default_factory=list)
    
    # Actionable insights
    insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Metadata
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    last_seen: Optional[datetime] = None
    occurrence_count: int = 0


class HistoricalStateManager:
    """
    Manages long-term storage and sophisticated analysis of historical states
    Provides pattern recognition, trend analysis, and predictive insights
    """
    
    def __init__(self, storage_path: str = "state_history", db_name: str = "states.db"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.db_path = self.storage_path / db_name
        self.state_files_path = self.storage_path / "state_files"
        self.state_files_path.mkdir(exist_ok=True)
        
        # Pattern storage
        self.identified_patterns: Dict[str, StatePattern] = {}
        
        # Initialize database
        asyncio.create_task(self._initialize_database())
        
        logger.info(f"HistoricalStateManager initialized with storage at {self.storage_path}")
    
    async def _initialize_database(self):
        """Initialize SQLite database for state metadata"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # States table for metadata and fast querying
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS states (
                state_id TEXT PRIMARY KEY,
                iteration_number INTEGER,
                timestamp TEXT,
                boss_state TEXT,
                system_phase TEXT,
                agent_count INTEGER,
                active_agent_count INTEGER,
                task_count INTEGER,
                active_task_count INTEGER,
                execution_success BOOLEAN,
                error_count INTEGER,
                confidence_score REAL,
                success_rate REAL,
                file_path TEXT
            )
        """)
        
        # Patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT,
                description TEXT,
                frequency REAL,
                confidence REAL,
                success_rate REAL,
                discovered_at TEXT,
                last_seen TEXT,
                occurrence_count INTEGER,
                pattern_data TEXT
            )
        """)
        
        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                state_id TEXT,
                metric_name TEXT,
                metric_value REAL,
                timestamp TEXT,
                FOREIGN KEY(state_id) REFERENCES states(state_id)
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON states(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_boss_state ON states(boss_state)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_iteration ON states(iteration_number)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_phase ON states(system_phase)")
        
        conn.commit()
        conn.close()
        
        logger.info("Database initialized successfully")
    
    async def store_state(self, state: AutonomousState) -> bool:
        """Store a state in long-term historical storage"""
        try:
            # Store full state as pickle file
            state_file_path = self.state_files_path / f"{state.state_id}.pkl"
            with open(state_file_path, 'wb') as f:
                pickle.dump(state.dict(), f)
            
            # Store metadata in database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO states 
                (state_id, iteration_number, timestamp, boss_state, system_phase,
                 agent_count, active_agent_count, task_count, active_task_count,
                 execution_success, error_count, confidence_score, success_rate, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                state.state_id,
                state.iteration_number,
                state.timestamp.isoformat(),
                state.boss_state,
                state.system_phase,
                len(state.agents),
                len(state.active_agents),
                len(state.tasks),
                len(state.active_tasks),
                state.execution_success,
                state.error_count,
                state.confidence_score,
                state.success_indicators.get("success_rate"),
                str(state_file_path)
            ))
            
            # Store metrics
            for metric_name, metric_value in state.metrics.items():
                if isinstance(metric_value, (int, float)):
                    cursor.execute("""
                        INSERT INTO metrics (state_id, metric_name, metric_value, timestamp)
                        VALUES (?, ?, ?, ?)
                    """, (state.state_id, metric_name, metric_value, state.timestamp.isoformat()))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"State {state.state_id} stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error storing state {state.state_id}: {e}")
            return False
    
    async def query_states(self, query: StateQuery) -> List[AutonomousState]:
        """Query historical states based on criteria"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Build SQL query
            where_clauses = []
            params = []
            
            # Time filters
            if query.start_time:
                where_clauses.append("timestamp >= ?")
                params.append(query.start_time.isoformat())
            
            if query.end_time:
                where_clauses.append("timestamp <= ?")
                params.append(query.end_time.isoformat())
            
            if query.last_n_hours:
                cutoff = datetime.utcnow() - timedelta(hours=query.last_n_hours)
                where_clauses.append("timestamp >= ?")
                params.append(cutoff.isoformat())
            
            if query.last_n_days:
                cutoff = datetime.utcnow() - timedelta(days=query.last_n_days)
                where_clauses.append("timestamp >= ?")
                params.append(cutoff.isoformat())
            
            # State filters
            if query.boss_states:
                placeholders = ",".join(["?" for _ in query.boss_states])
                where_clauses.append(f"boss_state IN ({placeholders})")
                params.extend(query.boss_states)
            
            if query.system_phases:
                placeholders = ",".join(["?" for _ in query.system_phases])
                where_clauses.append(f"system_phase IN ({placeholders})")
                params.extend(query.system_phases)
            
            if query.iteration_range:
                where_clauses.append("iteration_number BETWEEN ? AND ?")
                params.extend(query.iteration_range)
            
            # Performance filters
            if query.min_confidence is not None:
                where_clauses.append("confidence_score >= ?")
                params.append(query.min_confidence)
            
            if query.max_error_count is not None:
                where_clauses.append("error_count <= ?")
                params.append(query.max_error_count)
            
            if query.success_only:
                where_clauses.append("execution_success = 1")
            
            if query.has_decision:
                where_clauses.append("confidence_score IS NOT NULL")
            
            if query.agent_count_range:
                where_clauses.append("agent_count BETWEEN ? AND ?")
                params.extend(query.agent_count_range)
            
            if query.task_count_range:
                where_clauses.append("task_count BETWEEN ? AND ?")
                params.extend(query.task_count_range)
            
            # Build final query
            sql = "SELECT state_id, file_path FROM states"
            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)
            
            # Order and limit
            sql += f" ORDER BY {query.order_by}"
            if not query.ascending:
                sql += " DESC"
            
            if query.limit:
                sql += f" LIMIT {query.limit}"
            
            cursor.execute(sql, params)
            results = cursor.fetchall()
            conn.close()
            
            # Load full states from files
            states = []
            for state_id, file_path in results:
                try:
                    with open(file_path, 'rb') as f:
                        state_data = pickle.load(f)
                    
                    # Convert back to AutonomousState
                    state = AutonomousState(**state_data)
                    
                    # Apply text search if specified
                    if query.search_terms:
                        state_text = json.dumps(state_data, default=str).lower()
                        if all(term.lower() in state_text for term in query.search_terms):
                            states.append(state)
                    else:
                        states.append(state)
                        
                except Exception as e:
                    logger.warning(f"Error loading state file {file_path}: {e}")
            
            logger.info(f"Query returned {len(states)} states")
            return states
            
        except Exception as e:
            logger.error(f"Error querying states: {e}")
            return []
    
    async def analyze_patterns(self, lookback_days: int = 7) -> List[StatePattern]:
        """Analyze historical states to identify patterns"""
        try:
            # Query recent states for pattern analysis
            query = StateQuery(
                last_n_days=lookback_days,
                limit=1000,
                order_by="timestamp",
                ascending=True
            )
            
            states = await self.query_states(query)
            if len(states) < 10:
                logger.warning("Not enough states for pattern analysis")
                return []
            
            patterns = []
            
            # Pattern 1: Recurring state sequences
            patterns.extend(await self._identify_sequence_patterns(states))
            
            # Pattern 2: Conditional patterns (if X then Y)
            patterns.extend(await self._identify_conditional_patterns(states))
            
            # Pattern 3: Performance correlation patterns
            patterns.extend(await self._identify_performance_patterns(states))
            
            # Pattern 4: Temporal patterns (time-based patterns)
            patterns.extend(await self._identify_temporal_patterns(states))
            
            # Store patterns in database
            await self._store_patterns(patterns)
            
            logger.info(f"Identified {len(patterns)} patterns in historical data")
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return []
    
    async def _identify_sequence_patterns(self, states: List[AutonomousState]) -> List[StatePattern]:
        """Identify recurring sequences of states"""
        patterns = []
        
        # Look for sequences of 2-4 consecutive states
        for seq_len in range(2, 5):
            sequences = {}
            
            for i in range(len(states) - seq_len + 1):
                sequence = tuple(states[i + j].boss_state for j in range(seq_len))
                
                if sequence not in sequences:
                    sequences[sequence] = []
                sequences[sequence].append(i)
            
            # Identify frequent sequences
            for sequence, occurrences in sequences.items():
                if len(occurrences) >= 3:  # Minimum 3 occurrences
                    frequency = len(occurrences) / (len(states) - seq_len + 1)
                    
                    # Calculate success rate
                    successful = 0
                    for start_idx in occurrences:
                        end_idx = start_idx + seq_len - 1
                        if end_idx < len(states) and states[end_idx].execution_success:
                            successful += 1
                    
                    success_rate = successful / len(occurrences)
                    
                    pattern = StatePattern(
                        pattern_id=f"seq_{hash(sequence)}",
                        pattern_type="sequential",
                        description=f"Recurring sequence: {' → '.join(sequence)}",
                        frequency=frequency,
                        confidence=min(0.9, frequency * 2),  # Simple confidence calculation
                        states_involved=seq_len,
                        success_rate=success_rate,
                        typical_outcomes=[f"Sequence typically {'succeeds' if success_rate > 0.5 else 'fails'}"],
                        example_state_ids=[states[idx].state_id for idx in occurrences[:3]],
                        occurrence_count=len(occurrences),
                        insights=[f"This {seq_len}-state sequence occurs {frequency:.1%} of the time"]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _identify_conditional_patterns(self, states: List[AutonomousState]) -> List[StatePattern]:
        """Identify conditional patterns (if X then Y)"""
        patterns = []
        
        # Analyze state transitions based on conditions
        transitions = {}
        
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            
            # Create condition based on current state characteristics
            condition = (
                current_state.boss_state,
                len(current_state.active_agents) > 2,
                current_state.error_count > 0,
                current_state.confidence_score is not None and current_state.confidence_score > 0.8
            )
            
            outcome = next_state.boss_state
            
            if condition not in transitions:
                transitions[condition] = {}
            
            if outcome not in transitions[condition]:
                transitions[condition][outcome] = 0
            transitions[condition][outcome] += 1
        
        # Identify strong conditional patterns
        for condition, outcomes in transitions.items():
            total_occurrences = sum(outcomes.values())
            if total_occurrences >= 5:  # Minimum occurrences
                
                # Find most common outcome
                most_common_outcome = max(outcomes.items(), key=lambda x: x[1])
                outcome_probability = most_common_outcome[1] / total_occurrences
                
                if outcome_probability > 0.7:  # Strong correlation
                    boss_state, has_many_agents, has_errors, high_confidence = condition
                    
                    condition_desc = f"Boss in {boss_state}"
                    if has_many_agents:
                        condition_desc += " with many active agents"
                    if has_errors:
                        condition_desc += " with errors present"
                    if high_confidence:
                        condition_desc += " with high confidence"
                    
                    pattern = StatePattern(
                        pattern_id=f"cond_{hash(condition)}",
                        pattern_type="conditional",
                        description=f"When {condition_desc}, next state is usually {most_common_outcome[0]}",
                        frequency=total_occurrences / len(states),
                        confidence=outcome_probability,
                        states_involved=2,
                        success_rate=0.5,  # Default, would need more analysis
                        trigger_conditions=[condition_desc],
                        typical_outcomes=[f"Next state: {most_common_outcome[0]}"],
                        occurrence_count=total_occurrences,
                        insights=[f"Strong conditional pattern with {outcome_probability:.1%} probability"]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _identify_performance_patterns(self, states: List[AutonomousState]) -> List[StatePattern]:
        """Identify patterns related to performance metrics"""
        patterns = []
        
        # Analyze performance correlations
        high_performance_states = [s for s in states if s.execution_success and s.confidence_score and s.confidence_score > 0.8]
        low_performance_states = [s for s in states if s.execution_success is False or s.error_count > 2]
        
        if len(high_performance_states) >= 5:
            # Analyze characteristics of high performance states
            common_boss_states = {}
            common_agent_counts = {}
            
            for state in high_performance_states:
                common_boss_states[state.boss_state] = common_boss_states.get(state.boss_state, 0) + 1
                agent_count_range = "few" if len(state.active_agents) <= 2 else "many"
                common_agent_counts[agent_count_range] = common_agent_counts.get(agent_count_range, 0) + 1
            
            # Create performance pattern
            most_common_state = max(common_boss_states.items(), key=lambda x: x[1])
            most_common_agent_count = max(common_agent_counts.items(), key=lambda x: x[1])
            
            pattern = StatePattern(
                pattern_id="perf_high",
                pattern_type="performance",
                description=f"High performance typically occurs in {most_common_state[0]} state with {most_common_agent_count[0]} agents",
                frequency=len(high_performance_states) / len(states),
                confidence=0.8,
                states_involved=1,
                success_rate=1.0,
                typical_outcomes=["High performance", "Successful execution"],
                occurrence_count=len(high_performance_states),
                insights=[
                    f"High performance most common in {most_common_state[0]} state",
                    f"Optimal agent count appears to be {most_common_agent_count[0]}"
                ]
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _identify_temporal_patterns(self, states: List[AutonomousState]) -> List[StatePattern]:
        """Identify time-based patterns"""
        patterns = []
        
        # Analyze patterns by time of day, if timestamps span multiple days
        if len(states) > 0:
            time_span = states[-1].timestamp - states[0].timestamp
            if time_span.days > 1:
                
                # Group by hour of day
                hourly_performance = {}
                for state in states:
                    hour = state.timestamp.hour
                    if hour not in hourly_performance:
                        hourly_performance[hour] = {"total": 0, "successful": 0}
                    
                    hourly_performance[hour]["total"] += 1
                    if state.execution_success:
                        hourly_performance[hour]["successful"] += 1
                
                # Find patterns in hourly performance
                for hour, perf in hourly_performance.items():
                    if perf["total"] >= 5:  # Minimum samples
                        success_rate = perf["successful"] / perf["total"]
                        if success_rate > 0.8 or success_rate < 0.3:  # Notable performance
                            pattern = StatePattern(
                                pattern_id=f"temporal_{hour}",
                                pattern_type="temporal",
                                description=f"{'High' if success_rate > 0.8 else 'Low'} performance at hour {hour}:00",
                                frequency=perf["total"] / len(states),
                                confidence=0.7,
                                states_involved=1,
                                success_rate=success_rate,
                                typical_outcomes=[f"Performance tends to be {'good' if success_rate > 0.8 else 'poor'} at this time"],
                                occurrence_count=perf["total"],
                                insights=[f"Time-based performance pattern at {hour}:00"]
                            )
                            patterns.append(pattern)
        
        return patterns
    
    async def _store_patterns(self, patterns: List[StatePattern]):
        """Store identified patterns in database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            for pattern in patterns:
                cursor.execute("""
                    INSERT OR REPLACE INTO patterns 
                    (pattern_id, pattern_type, description, frequency, confidence, 
                     success_rate, discovered_at, occurrence_count, pattern_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_id,
                    pattern.pattern_type,
                    pattern.description,
                    pattern.frequency,
                    pattern.confidence,
                    pattern.success_rate,
                    pattern.discovered_at.isoformat(),
                    pattern.occurrence_count,
                    json.dumps(pattern.dict(), default=str)
                ))
                
                # Update in-memory storage
                self.identified_patterns[pattern.pattern_id] = pattern
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored {len(patterns)} patterns")
            
        except Exception as e:
            logger.error(f"Error storing patterns: {e}")
    
    async def get_stored_patterns(self) -> List[StatePattern]:
        """Retrieve stored patterns"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("SELECT pattern_data FROM patterns ORDER BY discovered_at DESC")
            results = cursor.fetchall()
            conn.close()
            
            patterns = []
            for (pattern_data,) in results:
                pattern_dict = json.loads(pattern_data)
                pattern = StatePattern(**pattern_dict)
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error retrieving patterns: {e}")
            return []
    
    async def get_trend_analysis(self, metric_name: str, days: int = 7) -> Dict[str, Any]:
        """Analyze trends for a specific metric"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cutoff = datetime.utcnow() - timedelta(days=days)
            cursor.execute("""
                SELECT timestamp, metric_value 
                FROM metrics 
                WHERE metric_name = ? AND timestamp >= ?
                ORDER BY timestamp
            """, (metric_name, cutoff.isoformat()))
            
            results = cursor.fetchall()
            conn.close()
            
            if len(results) < 2:
                return {"error": "Insufficient data for trend analysis"}
            
            # Calculate trend
            timestamps = [datetime.fromisoformat(r[0]) for r in results]
            values = [r[1] for r in results]
            
            # Simple linear trend calculation
            n = len(values)
            sum_x = sum(range(n))
            sum_y = sum(values)
            sum_xy = sum(i * values[i] for i in range(n))
            sum_x2 = sum(i * i for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            return {
                "metric_name": metric_name,
                "data_points": n,
                "time_span_days": (timestamps[-1] - timestamps[0]).days,
                "trend_slope": slope,
                "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                "first_value": values[0],
                "last_value": values[-1],
                "min_value": min(values),
                "max_value": max(values),
                "average_value": sum(values) / len(values)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend for {metric_name}: {e}")
            return {"error": str(e)}
