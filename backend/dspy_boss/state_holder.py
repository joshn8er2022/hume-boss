
"""
State Holder - Manages system state with historical tracking (100 recent + full storage)
"""

import json
import sqlite3
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from loguru import logger
import os

from .models import IterationResult, SystemState, LearningEntry


class StateHolder:
    """Manages system state with historical tracking and persistence"""
    
    def __init__(self, db_path: str = "state_storage.db", max_recent_states: int = 100):
        self.db_path = db_path
        self.max_recent_states = max_recent_states
        
        # In-memory cache for recent states (fast access)
        self.recent_states: List[Dict[str, Any]] = []
        self.current_state: Dict[str, Any] = {}
        
        # Initialize database
        asyncio.create_task(self._init_database())
        
    async def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # States table - stores all historical states
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    iteration_id INTEGER,
                    timestamp TEXT,
                    state_data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Iterations table - stores complete iteration results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS iterations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    iteration_id INTEGER UNIQUE,
                    timestamp TEXT,
                    pre_processing TEXT,
                    boss_decision TEXT,
                    execution TEXT,
                    next_prep TEXT,
                    error_info TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Learning table - stores system learnings
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learnings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learning_type TEXT,
                    content TEXT,
                    iteration_id INTEGER,
                    timestamp TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT,
                    metric_value TEXT,
                    iteration_id INTEGER,
                    timestamp TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"✅ State database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize state database: {e}")
            
    async def store_iteration_result(self, result: IterationResult):
        """Store complete iteration result"""
        try:
            # Store in database for persistence
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO iterations 
                (iteration_id, timestamp, pre_processing, boss_decision, execution, next_prep, error_info)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.iteration_id,
                result.timestamp.isoformat(),
                json.dumps(result.pre_processing, default=str) if result.pre_processing else None,
                json.dumps(result.boss_decision, default=str) if result.boss_decision else None,
                json.dumps(result.execution, default=str) if result.execution else None,
                json.dumps(result.next_prep, default=str) if result.next_prep else None,
                json.dumps(result.error_info, default=str) if result.error_info else None
            ))
            
            conn.commit()
            conn.close()
            
            # Update in-memory cache
            state_dict = asdict(result)
            self.recent_states.append(state_dict)
            
            # Maintain max recent states
            if len(self.recent_states) > self.max_recent_states:
                self.recent_states.pop(0)
                
            logger.debug(f"💾 Stored iteration {result.iteration_id} result")
            
        except Exception as e:
            logger.error(f"❌ Failed to store iteration result: {e}")
            
    def get_recent_states(self, count: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent states (up to 100 by default)"""
        if count is None:
            return self.recent_states.copy()
        return self.recent_states[-count:] if count <= len(self.recent_states) else self.recent_states.copy()
        
    def get_recent_iteration_results(self, count: int = 5) -> Dict[str, Any]:
        """Get results from recent iterations"""
        recent = self.get_recent_states(count)
        return {
            "count": len(recent),
            "results": recent,
            "summary": self._summarize_recent_results(recent)
        }
        
    def get_historical_patterns(self) -> Dict[str, Any]:
        """Analyze historical patterns from stored data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get success patterns
            cursor.execute('''
                SELECT boss_decision, execution, error_info 
                FROM iterations 
                ORDER BY iteration_id DESC 
                LIMIT 50
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            patterns = {
                "total_iterations": len(results),
                "success_rate": len([r for r in results if r[2] is None]) / len(results) if results else 0,
                "common_decisions": self._extract_common_patterns([r[0] for r in results if r[0]]),
                "failure_patterns": self._extract_failure_patterns([r[2] for r in results if r[2]]),
                "execution_patterns": self._extract_execution_patterns([r[1] for r in results if r[1]])
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze historical patterns: {e}")
            return {"error": str(e)}
            
    def get_historical_outcomes(self) -> Dict[str, Any]:
        """Get historical outcomes for forecasting"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT boss_decision, execution, error_info
                FROM iterations
                WHERE execution IS NOT NULL
                ORDER BY iteration_id DESC
                LIMIT 100
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            outcomes = {
                "successful_decisions": [],
                "failed_decisions": [],
                "outcome_statistics": {}
            }
            
            for decision, execution, error in results:
                if error is None:
                    outcomes["successful_decisions"].append({
                        "decision": decision,
                        "execution": execution
                    })
                else:
                    outcomes["failed_decisions"].append({
                        "decision": decision,
                        "error": error
                    })
                    
            outcomes["outcome_statistics"] = {
                "total_outcomes": len(results),
                "success_count": len(outcomes["successful_decisions"]),
                "failure_count": len(outcomes["failed_decisions"]),
                "success_rate": len(outcomes["successful_decisions"]) / len(results) if results else 0
            }
            
            return outcomes
            
        except Exception as e:
            logger.error(f"❌ Failed to get historical outcomes: {e}")
            return {}

    async def store_learning(self, learning_data: Dict[str, Any]):
        """Store system learning"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO learnings (learning_type, content, iteration_id, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (
                "iteration_analysis",
                json.dumps(learning_data, default=str),
                learning_data.get("iteration_id", 0),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug("💡 Stored system learning")
            
        except Exception as e:
            logger.error(f"❌ Failed to store learning: {e}")
            
    async def store_error_state(self, error: Exception, context: Optional[Any] = None):
        """Store error state for analysis"""
        try:
            error_data = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": str(context) if context else None,
                "timestamp": datetime.now().isoformat()
            }
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO learnings (learning_type, content, timestamp)
                VALUES (?, ?, ?)
            ''', (
                "error_analysis",
                json.dumps(error_data, default=str),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug("🚨 Stored error state for learning")
            
        except Exception as e:
            logger.error(f"❌ Failed to store error state: {e}")
            
    def get_total_decisions(self) -> int:
        """Get total number of decisions made"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM iterations WHERE boss_decision IS NOT NULL')
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except:
            return 0
            
    def get_success_rate(self) -> float:
        """Get overall success rate"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM iterations')
            total = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM iterations WHERE error_info IS NULL')
            successful = cursor.fetchone()[0]
            conn.close()
            return successful / total if total > 0 else 0.0
        except:
            return 0.0
            
    def update_current_state(self, state_data: Dict[str, Any]):
        """Update current state"""
        self.current_state = {
            **state_data,
            "timestamp": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
    def get_current_state(self) -> Dict[str, Any]:
        """Get current state"""
        return self.current_state.copy()
        
    async def get_state_by_iteration(self, iteration_id: int) -> Optional[Dict[str, Any]]:
        """Get state by iteration ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM iterations WHERE iteration_id = ?', (iteration_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    "iteration_id": result[1],
                    "timestamp": result[2],
                    "pre_processing": json.loads(result[3]) if result[3] else None,
                    "boss_decision": json.loads(result[4]) if result[4] else None,
                    "execution": json.loads(result[5]) if result[5] else None,
                    "next_prep": json.loads(result[6]) if result[6] else None,
                    "error_info": json.loads(result[7]) if result[7] else None
                }
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get state by iteration: {e}")
            return None
            
    def _summarize_recent_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize recent results"""
        if not results:
            return {"status": "no_data"}
            
        successful = len([r for r in results if r.get("error_info") is None])
        return {
            "total_iterations": len(results),
            "successful_iterations": successful,
            "failed_iterations": len(results) - successful,
            "recent_success_rate": successful / len(results),
            "latest_timestamp": results[-1].get("timestamp") if results else None
        }
        
    def _extract_common_patterns(self, decisions: List[str]) -> Dict[str, Any]:
        """Extract common decision patterns"""
        # Simple pattern analysis - in production this would be more sophisticated
        decision_counts = {}
        for decision in decisions:
            if decision:
                try:
                    decision_data = json.loads(decision)
                    decision_type = decision_data.get("decision", "unknown")
                    decision_counts[decision_type] = decision_counts.get(decision_type, 0) + 1
                except:
                    continue
                    
        return {
            "most_common": max(decision_counts.items(), key=lambda x: x[1]) if decision_counts else None,
            "decision_distribution": decision_counts
        }
        
    def _extract_failure_patterns(self, errors: List[str]) -> Dict[str, Any]:
        """Extract failure patterns"""
        error_types = {}
        for error in errors:
            if error:
                try:
                    error_data = json.loads(error)
                    error_type = error_data.get("error_type", "unknown")
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                except:
                    continue
                    
        return {
            "most_common_error": max(error_types.items(), key=lambda x: x[1]) if error_types else None,
            "error_distribution": error_types
        }
        
    def _extract_execution_patterns(self, executions: List[str]) -> Dict[str, Any]:
        """Extract execution patterns"""
        task_success_rates = {}
        for execution in executions:
            if execution:
                try:
                    exec_data = json.loads(execution)
                    completed = exec_data.get("tasks_completed", 0)
                    failed = exec_data.get("tasks_failed", 0)
                    total = completed + failed
                    if total > 0:
                        success_rate = completed / total
                        task_success_rates[f"execution_{len(task_success_rates)}"] = success_rate
                except:
                    continue
                    
        avg_success = sum(task_success_rates.values()) / len(task_success_rates) if task_success_rates else 0
        
        return {
            "average_execution_success_rate": avg_success,
            "total_executions_analyzed": len(task_success_rates)
        }
