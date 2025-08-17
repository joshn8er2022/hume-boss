
"""
State forecasting engine using DSPY signatures and historical patterns
Predicts future states based on current context and historical data
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field
from loguru import logger

import dspy

from .state_holder import AutonomousState
from .historical_manager import HistoricalStateManager, StatePattern, StateQuery
from ..signatures.state_forecasting import StateForecastingSignature, HistoricalAnalysisSignature, ForecastingContext, StateForecast, HistoricalPattern


class ForecastModel(BaseModel):
    """Model representing a forecast prediction"""
    
    forecast_id: str = Field(..., description="Unique forecast identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    forecast_horizon: str = Field(..., description="Time horizon (short/medium/long-term)")
    
    # Input context
    base_state: Dict[str, Any] = Field(..., description="Base state used for forecasting")
    historical_context: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Forecast results
    primary_forecast: StateForecast = Field(..., description="Primary forecast scenario")
    alternative_scenarios: List[StateForecast] = Field(default_factory=list)
    
    # Confidence and validation
    overall_confidence: float = Field(..., description="Overall confidence in forecast")
    validation_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Strategic insights
    strategic_recommendations: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    opportunity_factors: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class StateForecaster:
    """
    Advanced state forecasting engine that uses DSPY signatures and historical analysis
    to predict future system states and provide strategic insights
    """
    
    def __init__(self, historical_manager: HistoricalStateManager):
        self.historical_manager = historical_manager
        self.forecasting_signature = StateForecastingSignature()
        self.historical_analysis_signature = HistoricalAnalysisSignature()
        
        # Forecast cache
        self.forecast_cache: Dict[str, ForecastModel] = {}
        self.cache_ttl = timedelta(minutes=30)
        
        # Pattern cache
        self.pattern_cache: List[StatePattern] = []
        self.pattern_cache_time: Optional[datetime] = None
        
        logger.info("StateForecaster initialized")
    
    async def forecast_next_states(
        self, 
        current_state: AutonomousState,
        horizon: str = "short-term",
        scenarios: int = 3
    ) -> ForecastModel:
        """
        Generate forecasts for future states based on current state and historical patterns
        
        Args:
            current_state: Current system state
            horizon: Forecast horizon - "short-term", "medium-term", or "long-term"
            scenarios: Number of alternative scenarios to generate
        
        Returns:
            ForecastModel with primary forecast and alternative scenarios
        """
        try:
            logger.info(f"Generating {horizon} forecast with {scenarios} scenarios")
            
            # Check cache first
            cache_key = f"{current_state.state_id}_{horizon}_{scenarios}"
            if cache_key in self.forecast_cache:
                cached_forecast = self.forecast_cache[cache_key]
                if datetime.utcnow() - cached_forecast.created_at < self.cache_ttl:
                    logger.debug("Returning cached forecast")
                    return cached_forecast
            
            # Prepare historical context
            historical_context = await self._prepare_historical_context(current_state, horizon)
            
            # Create forecasting context
            forecasting_context = ForecastingContext(
                current_state=current_state.dict(),
                historical_states=historical_context["historical_states"],
                trend_indicators=historical_context["trend_indicators"],
                external_factors=historical_context["external_factors"],
                system_capacity=historical_context["system_capacity"]
            )
            
            # Generate scenario parameters
            scenario_parameters = await self._generate_scenario_parameters(current_state, scenarios)
            
            # Use DSPY signature to generate forecast
            with dspy.context(lm=dspy.OpenAI(model="gpt-4")):  # Use appropriate LLM
                forecast_result = self.forecasting_signature(
                    forecasting_context=forecasting_context,
                    forecast_horizon=horizon,
                    scenario_parameters=scenario_parameters
                )
            
            # Create forecast model
            forecast_model = ForecastModel(
                forecast_id=f"forecast_{current_state.state_id}_{int(datetime.utcnow().timestamp())}",
                forecast_horizon=horizon,
                base_state=current_state.dict(),
                historical_context=historical_context["historical_states"][-10:],  # Last 10 for reference
                primary_forecast=forecast_result.primary_forecast,
                alternative_scenarios=forecast_result.alternative_scenarios,
                overall_confidence=forecast_result.primary_forecast.confidence_level,
                strategic_recommendations=forecast_result.strategic_recommendations
            )
            
            # Add validation metrics
            forecast_model.validation_metrics = await self._validate_forecast(forecast_model, current_state)
            
            # Cache the forecast
            self.forecast_cache[cache_key] = forecast_model
            
            logger.info(f"Generated forecast with confidence {forecast_model.overall_confidence:.2f}")
            return forecast_model
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            # Return a simple fallback forecast
            return await self._create_fallback_forecast(current_state, horizon)
    
    async def analyze_historical_patterns(
        self, 
        current_state: AutonomousState,
        analysis_focus: List[str] = None,
        timeframe_days: int = 14
    ) -> List[HistoricalPattern]:
        """
        Analyze historical patterns relevant to current state
        
        Args:
            current_state: Current system state for context
            analysis_focus: Specific areas to focus analysis on
            timeframe_days: Number of days to analyze
        
        Returns:
            List of identified historical patterns
        """
        try:
            # Get historical data
            query = StateQuery(
                last_n_days=timeframe_days,
                limit=500,
                order_by="timestamp",
                ascending=True
            )
            
            historical_states = await self.historical_manager.query_states(query)
            if len(historical_states) < 10:
                logger.warning("Insufficient historical data for pattern analysis")
                return []
            
            # Prepare data for DSPY analysis
            historical_data = [state.dict() for state in historical_states]
            
            if not analysis_focus:
                analysis_focus = ["performance", "agent_behavior", "task_completion", "error_patterns"]
            
            # Use DSPY signature for historical analysis
            with dspy.context(lm=dspy.OpenAI(model="gpt-4")):
                analysis_result = self.historical_analysis_signature(
                    historical_data=historical_data,
                    analysis_focus=analysis_focus,
                    comparison_timeframe=f"{timeframe_days} days"
                )
            
            # Convert to HistoricalPattern objects
            patterns = analysis_result.analysis_results.identified_patterns
            
            logger.info(f"Identified {len(patterns)} historical patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing historical patterns: {e}")
            return []
    
    async def predict_performance_trends(
        self, 
        current_state: AutonomousState,
        metrics: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Predict future performance trends for specific metrics
        
        Args:
            current_state: Current system state
            metrics: List of metrics to analyze (defaults to key metrics)
        
        Returns:
            Dictionary of trend predictions for each metric
        """
        try:
            if not metrics:
                metrics = ["success_rate", "task_completion_time", "error_rate", "agent_utilization"]
            
            trend_predictions = {}
            
            for metric in metrics:
                # Get historical trend data
                trend_data = await self.historical_manager.get_trend_analysis(metric, days=14)
                
                if "error" not in trend_data:
                    # Simple trend prediction based on historical slope
                    current_value = current_state.success_indicators.get(metric) or current_state.metrics.get(metric, 0)
                    slope = trend_data.get("trend_slope", 0)
                    
                    # Predict next few values
                    predictions = []
                    for i in range(1, 6):  # Next 5 time periods
                        predicted_value = current_value + (slope * i)
                        predictions.append(max(0, min(1, predicted_value)) if metric in ["success_rate", "agent_utilization"] else max(0, predicted_value))
                    
                    trend_predictions[metric] = {
                        "current_value": current_value,
                        "trend_direction": trend_data["trend_direction"],
                        "predictions": predictions,
                        "confidence": min(0.9, abs(slope) * 10),  # Simple confidence calculation
                        "recommendation": self._generate_metric_recommendation(metric, trend_data, predictions)
                    }
                else:
                    trend_predictions[metric] = {
                        "error": trend_data["error"],
                        "recommendation": "Insufficient data for trend prediction"
                    }
            
            logger.info(f"Generated performance trend predictions for {len(trend_predictions)} metrics")
            return trend_predictions
            
        except Exception as e:
            logger.error(f"Error predicting performance trends: {e}")
            return {}
    
    async def forecast_resource_requirements(
        self, 
        current_state: AutonomousState,
        forecast_model: ForecastModel
    ) -> Dict[str, Any]:
        """
        Forecast resource requirements based on predicted future states
        
        Args:
            current_state: Current system state
            forecast_model: Forecast model to base requirements on
        
        Returns:
            Dictionary of predicted resource requirements
        """
        try:
            primary_forecast = forecast_model.primary_forecast
            
            # Current resource usage
            current_agents = len(current_state.active_agents)
            current_tasks = len(current_state.active_tasks)
            
            # Predicted resource needs based on forecasted state
            forecasted_state = primary_forecast.forecasted_state
            
            # Extract predictions from forecasted state
            predicted_agent_count = forecasted_state.get("predicted_agent_count", current_agents)
            predicted_task_load = forecasted_state.get("predicted_task_load", current_tasks)
            predicted_processing_power = forecasted_state.get("processing_requirements", "normal")
            
            # Calculate resource requirements
            resource_requirements = {
                "agents": {
                    "current": current_agents,
                    "predicted": predicted_agent_count,
                    "change": predicted_agent_count - current_agents,
                    "recommendation": "scale_up" if predicted_agent_count > current_agents * 1.2 else 
                                   "scale_down" if predicted_agent_count < current_agents * 0.8 else "maintain"
                },
                "compute": {
                    "current_load": current_tasks,
                    "predicted_load": predicted_task_load,
                    "processing_power": predicted_processing_power,
                    "recommendation": self._get_compute_recommendation(current_tasks, predicted_task_load)
                },
                "memory": {
                    "state_storage": f"{len(current_state.tasks)} tasks in memory",
                    "historical_data": "14 days of state history",
                    "recommendation": "monitor" if predicted_task_load < current_tasks * 2 else "increase_capacity"
                },
                "network": {
                    "mcp_connections": len(current_state.mcp_servers),
                    "predicted_load": "normal",
                    "recommendation": "monitor"
                }
            }
            
            # Add risk factors
            resource_requirements["risks"] = primary_forecast.potential_risks
            resource_requirements["opportunities"] = primary_forecast.opportunity_areas
            
            logger.info("Generated resource requirement forecast")
            return resource_requirements
            
        except Exception as e:
            logger.error(f"Error forecasting resource requirements: {e}")
            return {"error": str(e)}
    
    async def _prepare_historical_context(self, current_state: AutonomousState, horizon: str) -> Dict[str, Any]:
        """Prepare historical context for forecasting"""
        
        # Determine lookback period based on horizon
        lookback_days = {
            "short-term": 3,
            "medium-term": 7, 
            "long-term": 14
        }.get(horizon, 7)
        
        # Query historical states
        query = StateQuery(
            last_n_days=lookback_days,
            limit=200,
            order_by="timestamp",
            ascending=True
        )
        
        historical_states = await self.historical_manager.query_states(query)
        
        # Calculate trend indicators
        trend_indicators = {}
        if len(historical_states) > 1:
            # Success rate trend
            success_rates = [s.success_indicators.get("success_rate", 0.5) for s in historical_states[-10:]]
            if success_rates:
                trend_indicators["success_rate_trend"] = (success_rates[-1] - success_rates[0]) / len(success_rates)
            
            # Agent count trend
            agent_counts = [len(s.active_agents) for s in historical_states[-10:]]
            if agent_counts:
                trend_indicators["agent_count_trend"] = (agent_counts[-1] - agent_counts[0]) / len(agent_counts)
            
            # Error trend
            error_counts = [s.error_count for s in historical_states[-10:]]
            if error_counts:
                trend_indicators["error_trend"] = (error_counts[-1] - error_counts[0]) / len(error_counts)
        
        # External factors (simplified)
        external_factors = [
            "system_load_normal",
            "mcp_servers_stable"
        ]
        
        if current_state.error_count > 2:
            external_factors.append("elevated_error_conditions")
        
        # System capacity
        system_capacity = {
            "max_agents": 10,  # Configuration-based
            "max_concurrent_tasks": 50,
            "available_mcp_servers": len(current_state.mcp_servers),
            "current_utilization": len(current_state.active_tasks) / 50  # Simplified calculation
        }
        
        return {
            "historical_states": [s.dict() for s in historical_states],
            "trend_indicators": trend_indicators,
            "external_factors": external_factors,
            "system_capacity": system_capacity
        }
    
    async def _generate_scenario_parameters(self, current_state: AutonomousState, scenarios: int) -> Dict[str, Any]:
        """Generate parameters for different scenarios"""
        
        base_params = {
            "current_agent_count": len(current_state.active_agents),
            "current_task_count": len(current_state.active_tasks),
            "current_error_count": current_state.error_count,
            "system_phase": current_state.system_phase
        }
        
        # Generate scenario variations
        scenario_variations = []
        
        # Optimistic scenario
        scenario_variations.append({
            **base_params,
            "scenario_type": "optimistic",
            "error_reduction_factor": 0.5,
            "performance_boost": 1.2
        })
        
        # Realistic scenario  
        scenario_variations.append({
            **base_params,
            "scenario_type": "realistic",
            "error_reduction_factor": 0.8,
            "performance_boost": 1.0
        })
        
        # Pessimistic scenario
        if scenarios > 2:
            scenario_variations.append({
                **base_params,
                "scenario_type": "pessimistic", 
                "error_increase_factor": 1.5,
                "performance_degradation": 0.8
            })
        
        return {
            "base_parameters": base_params,
            "scenario_variations": scenario_variations[:scenarios]
        }
    
    async def _validate_forecast(self, forecast_model: ForecastModel, current_state: AutonomousState) -> Dict[str, float]:
        """Validate forecast using various metrics"""
        
        validation_metrics = {}
        
        # Consistency check
        primary_forecast = forecast_model.primary_forecast
        consistency_score = 1.0
        
        # Check if predicted changes are reasonable
        forecasted_state = primary_forecast.forecasted_state
        if "agent_count" in forecasted_state:
            current_agents = len(current_state.active_agents)
            predicted_agents = forecasted_state["agent_count"]
            if abs(predicted_agents - current_agents) > 5:  # Large change
                consistency_score *= 0.8
        
        validation_metrics["consistency_score"] = consistency_score
        
        # Confidence alignment
        validation_metrics["confidence_alignment"] = primary_forecast.confidence_level
        
        # Historical accuracy (would need past forecasts to calculate properly)
        validation_metrics["historical_accuracy"] = 0.75  # Placeholder
        
        return validation_metrics
    
    async def _create_fallback_forecast(self, current_state: AutonomousState, horizon: str) -> ForecastModel:
        """Create a simple fallback forecast when main forecasting fails"""
        
        # Simple state continuation
        fallback_forecast = StateForecast(
            forecasted_state=current_state.dict(),
            confidence_level=0.3,
            key_changes=["Minimal changes expected"],
            potential_risks=["Limited forecasting data"],
            timeline=f"Next iteration ({horizon})"
        )
        
        return ForecastModel(
            forecast_id=f"fallback_{current_state.state_id}",
            forecast_horizon=horizon,
            base_state=current_state.dict(),
            primary_forecast=fallback_forecast,
            overall_confidence=0.3,
            strategic_recommendations=["Monitor system closely", "Gather more historical data"]
        )
    
    def _generate_metric_recommendation(self, metric: str, trend_data: Dict, predictions: List[float]) -> str:
        """Generate recommendation based on metric trends"""
        
        trend_direction = trend_data.get("trend_direction", "stable")
        
        if metric == "success_rate":
            if trend_direction == "decreasing":
                return "Focus on error reduction and process improvement"
            elif trend_direction == "increasing":
                return "Continue current strategies, monitor for sustainability"
            else:
                return "Maintain current performance levels"
        
        elif metric == "error_rate":
            if trend_direction == "increasing":
                return "Immediate attention required to reduce error rate"
            else:
                return "Monitor error patterns and maintain preventive measures"
        
        elif metric == "agent_utilization":
            if trend_direction == "increasing":
                return "Consider scaling up agent capacity"
            elif trend_direction == "decreasing":
                return "Optimize task distribution or scale down agents"
            else:
                return "Current utilization levels appear optimal"
        
        return "Monitor trends and adjust as needed"
    
    def _get_compute_recommendation(self, current_load: int, predicted_load: int) -> str:
        """Get compute resource recommendation"""
        
        if predicted_load > current_load * 1.5:
            return "scale_up_significant"
        elif predicted_load > current_load * 1.2:
            return "scale_up_moderate"
        elif predicted_load < current_load * 0.7:
            return "scale_down"
        else:
            return "maintain_current"
    
    async def cleanup_old_forecasts(self, max_age_hours: int = 24):
        """Clean up old forecasts from cache"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        old_forecasts = [
            key for key, forecast in self.forecast_cache.items()
            if forecast.created_at < cutoff_time
        ]
        
        for key in old_forecasts:
            del self.forecast_cache[key]
        
        if old_forecasts:
            logger.info(f"Cleaned up {len(old_forecasts)} old forecasts from cache")
