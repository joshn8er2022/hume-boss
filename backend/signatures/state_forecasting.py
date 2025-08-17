
"""
DSPY Signatures for state forecasting and historical analysis
"""

import dspy
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ForecastingContext(BaseModel):
    """Context for state forecasting"""
    current_state: Dict[str, Any]
    historical_states: List[Dict[str, Any]]
    trend_indicators: Dict[str, float]
    external_factors: List[str]
    system_capacity: Dict[str, Any]


class StateForecast(BaseModel):
    """Forecasted future state"""
    forecasted_state: Dict[str, Any] = Field(..., description="Predicted future system state")
    confidence_level: float = Field(..., description="Confidence in forecast (0.0-1.0)")
    key_changes: List[str] = Field(..., description="Key changes expected")
    potential_risks: List[str] = Field(..., description="Potential risks to monitor")
    opportunity_areas: List[str] = Field(default_factory=list)
    timeline: str = Field(..., description="Expected timeline for this state")
    preparation_actions: List[str] = Field(default_factory=list)


class StateForecastingSignature(dspy.Signature):
    """
    DSPY signature for forecasting future system states.
    Analyzes patterns and predicts what future states will look like.
    """
    
    forecasting_context: ForecastingContext = dspy.InputField(description="Context for state forecasting")
    forecast_horizon: str = dspy.InputField(description="Time horizon for forecast (short/medium/long-term)")
    scenario_parameters: Dict[str, Any] = dspy.InputField(description="Parameters for scenario analysis")
    
    primary_forecast: StateForecast = dspy.OutputField(description="Primary forecasted state scenario")
    alternative_scenarios: List[StateForecast] = dspy.OutputField(description="Alternative possible scenarios")
    strategic_recommendations: List[str] = dspy.OutputField(description="Strategic recommendations based on forecasts")


class HistoricalPattern(BaseModel):
    """Identified historical pattern"""
    pattern_type: str = Field(..., description="Type of pattern identified")
    pattern_description: str = Field(..., description="Description of the pattern")
    frequency: str = Field(..., description="How often this pattern occurs")
    correlation_strength: float = Field(..., description="Strength of correlation (0.0-1.0)")
    triggers: List[str] = Field(..., description="What triggers this pattern")
    outcomes: List[str] = Field(..., description="Typical outcomes of this pattern")
    actionable_insights: List[str] = Field(default_factory=list)


class HistoricalAnalysisOutput(BaseModel):
    """Output of historical analysis"""
    identified_patterns: List[HistoricalPattern] = Field(..., description="Patterns found in historical data")
    trend_analysis: Dict[str, Any] = Field(..., description="Analysis of trends over time")
    success_factors: List[str] = Field(..., description="Factors that led to success")
    failure_indicators: List[str] = Field(..., description="Early indicators of failure")
    optimization_opportunities: List[str] = Field(default_factory=list)
    lessons_learned: List[str] = Field(default_factory=list)


class HistoricalAnalysisSignature(dspy.Signature):
    """
    DSPY signature for analyzing historical patterns and extracting insights.
    Provides context for decision making by understanding past performance.
    """
    
    historical_data: List[Dict[str, Any]] = dspy.InputField(description="Historical state and performance data")
    analysis_focus: List[str] = dspy.InputField(description="Specific areas to focus analysis on")
    comparison_timeframe: str = dspy.InputField(description="Timeframe for comparison analysis")
    
    analysis_results: HistoricalAnalysisOutput = dspy.OutputField(description="Comprehensive historical analysis")
    predictive_insights: List[str] = dspy.OutputField(description="Insights that can inform future decisions")
    pattern_reliability: float = dspy.OutputField(description="Overall reliability of identified patterns")
