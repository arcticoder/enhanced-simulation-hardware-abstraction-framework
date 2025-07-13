#!/usr/bin/env python3
"""
Crew Economic Optimizer - Enhanced Simulation Hardware Abstraction Framework

Revolutionary economic modeling framework for optimal crew size and role distribution
in interstellar LQG FTL missions with comprehensive cost-benefit analysis.

Author: Enhanced Simulation Hardware Abstraction Framework
Date: July 13, 2025
Version: 1.0.0 - Production Implementation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, uniform, gamma
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MissionType(Enum):
    """Mission types for crew optimization."""
    SCIENTIFIC_EXPLORATION = "scientific_exploration"
    TOURISM = "tourism"
    CARGO_TRANSPORT = "cargo_transport"
    COLONIZATION = "colonization"
    DIPLOMATIC = "diplomatic"
    MIXED_MISSION = "mixed_mission"

class CrewRole(Enum):
    """Crew roles for optimization."""
    COMMAND = "command"
    ENGINEERING = "engineering"
    MEDICAL = "medical"
    SCIENCE = "science"
    MAINTENANCE = "maintenance"
    SECURITY = "security"
    PASSENGERS = "passengers"
    SUPPORT = "support"

@dataclass
class EconomicParameters:
    """Economic parameters for crew optimization."""
    base_vessel_cost: float = 2.5e9  # $2.5B base vessel cost
    crew_training_cost: float = 500000  # $500K per crew member
    life_support_cost_per_person: float = 10000  # $10K per person per day
    medical_bay_cost: float = 5e6  # $5M medical bay
    passenger_revenue_per_day: float = 50000  # $50K per passenger per day
    mission_insurance_base: float = 100e6  # $100M base insurance
    fuel_cost_multiplier: float = 1.2  # 20% increase per 10 crew members
    maintenance_cost_factor: float = 0.05  # 5% of vessel cost annually
    emergency_fund_ratio: float = 0.15  # 15% emergency fund
    profit_margin_target: float = 0.20  # 20% profit margin target

@dataclass
class CrewConfiguration:
    """Crew configuration for optimization."""
    total_crew: int
    command: int
    engineering: int
    medical: int
    science: int
    maintenance: int
    security: int
    passengers: int
    support: int
    mission_type: MissionType
    mission_duration_days: int = 90
    
    def __post_init__(self):
        """Validate crew configuration."""
        if self.total_crew > 100:
            raise ValueError("Total crew cannot exceed 100 personnel")
        
        crew_sum = (self.command + self.engineering + self.medical + 
                   self.science + self.maintenance + self.security + 
                   self.passengers + self.support)
        
        if crew_sum != self.total_crew:
            raise ValueError(f"Crew roles sum ({crew_sum}) != total crew ({self.total_crew})")

@dataclass
class EconomicResults:
    """Economic analysis results."""
    total_cost: float
    total_revenue: float
    net_profit: float
    roi: float
    cost_breakdown: Dict[str, float]
    revenue_breakdown: Dict[str, float]
    safety_score: float
    efficiency_score: float
    risk_assessment: Dict[str, float]
    monte_carlo_confidence: float = 0.0

class CrewEconomicOptimizer:
    """
    Revolutionary crew economic optimization framework for interstellar missions.
    
    Implements comprehensive cost-benefit analysis with Monte Carlo simulation
    for optimal crew size and role distribution within ≤100 personnel constraint.
    """
    
    def __init__(self, economic_params: Optional[EconomicParameters] = None):
        """Initialize the crew economic optimizer."""
        self.economic_params = economic_params or EconomicParameters()
        self.optimization_history = []
        self.monte_carlo_results = {}
        
        # Role requirements by mission type
        self.mission_requirements = {
            MissionType.SCIENTIFIC_EXPLORATION: {
                "min_science": 0.25,  # 25% science crew minimum
                "min_medical": 0.08,  # 8% medical minimum
                "min_engineering": 0.20,  # 20% engineering minimum
                "passenger_ratio": 0.10  # Up to 10% passengers
            },
            MissionType.TOURISM: {
                "min_science": 0.05,  # 5% science crew minimum
                "min_medical": 0.12,  # 12% medical for passenger safety
                "min_engineering": 0.15,  # 15% engineering minimum
                "passenger_ratio": 0.60  # Up to 60% passengers
            },
            MissionType.CARGO_TRANSPORT: {
                "min_science": 0.02,  # 2% science crew minimum
                "min_medical": 0.06,  # 6% medical minimum
                "min_engineering": 0.25,  # 25% engineering for cargo systems
                "passenger_ratio": 0.05  # Up to 5% passengers
            },
            MissionType.COLONIZATION: {
                "min_science": 0.15,  # 15% science crew minimum
                "min_medical": 0.15,  # 15% medical for colonist health
                "min_engineering": 0.30,  # 30% engineering for infrastructure
                "passenger_ratio": 0.25  # Up to 25% colonists
            },
            MissionType.DIPLOMATIC: {
                "min_science": 0.08,  # 8% science crew minimum
                "min_medical": 0.08,  # 8% medical minimum
                "min_engineering": 0.15,  # 15% engineering minimum
                "passenger_ratio": 0.30  # Up to 30% diplomatic staff
            },
            MissionType.MIXED_MISSION: {
                "min_science": 0.12,  # 12% science crew minimum
                "min_medical": 0.10,  # 10% medical minimum
                "min_engineering": 0.18,  # 18% engineering minimum
                "passenger_ratio": 0.35  # Up to 35% passengers
            }
        }
        
        logger.info("Crew Economic Optimizer initialized with production parameters")
    
    def calculate_total_cost(self, config: CrewConfiguration) -> Dict[str, float]:
        """Calculate total mission cost breakdown."""
        costs = {}
        
        # Base vessel cost (amortized over mission)
        costs["vessel_amortization"] = (self.economic_params.base_vessel_cost * 
                                       config.mission_duration_days / 365.25 / 10)  # 10-year amortization
        
        # Crew training costs
        non_passenger_crew = config.total_crew - config.passengers
        costs["crew_training"] = non_passenger_crew * self.economic_params.crew_training_cost
        
        # Life support costs
        costs["life_support"] = (config.total_crew * 
                               self.economic_params.life_support_cost_per_person * 
                               config.mission_duration_days)
        
        # Medical bay costs (scaled by crew size)
        medical_scaling = max(1.0, config.total_crew / 50)  # Scale medical costs
        costs["medical_systems"] = self.economic_params.medical_bay_cost * medical_scaling
        
        # Fuel costs (increased with crew size)
        fuel_multiplier = 1 + (config.total_crew / 100) * 0.5  # 50% increase at max crew
        costs["fuel_systems"] = 50e6 * fuel_multiplier  # Base $50M fuel cost
        
        # Insurance costs (risk-adjusted)
        risk_factor = self.calculate_mission_risk(config)
        costs["insurance"] = self.economic_params.mission_insurance_base * (1 + risk_factor)
        
        # Maintenance costs
        costs["maintenance"] = (self.economic_params.base_vessel_cost * 
                              self.economic_params.maintenance_cost_factor * 
                              config.mission_duration_days / 365.25)
        
        # Emergency fund
        total_operational = sum([costs["life_support"], costs["fuel_systems"], 
                               costs["maintenance"]])
        costs["emergency_fund"] = total_operational * self.economic_params.emergency_fund_ratio
        
        # Crew compensation (mission-duration based)
        costs["crew_compensation"] = self.calculate_crew_compensation(config)
        
        return costs
    
    def calculate_total_revenue(self, config: CrewConfiguration) -> Dict[str, float]:
        """Calculate total mission revenue breakdown."""
        revenues = {}
        
        # Passenger revenue
        revenues["passenger_revenue"] = (config.passengers * 
                                       self.economic_params.passenger_revenue_per_day * 
                                       config.mission_duration_days)
        
        # Scientific data revenue (for scientific missions)
        if config.mission_type == MissionType.SCIENTIFIC_EXPLORATION:
            science_factor = config.science / config.total_crew
            revenues["scientific_data"] = 25e6 * science_factor  # Up to $25M for data
        
        # Cargo transport revenue
        if config.mission_type == MissionType.CARGO_TRANSPORT:
            cargo_capacity = max(0, 100 - config.total_crew) * 2  # 2 tons per unused crew slot
            revenues["cargo_revenue"] = cargo_capacity * 500000  # $500K per ton
        
        # Government contracts (diplomatic/colonization)
        if config.mission_type in [MissionType.DIPLOMATIC, MissionType.COLONIZATION]:
            revenues["government_contract"] = 150e6  # $150M government contract
        
        # Technology licensing (innovation revenue)
        engineering_factor = config.engineering / config.total_crew
        revenues["technology_licensing"] = 10e6 * engineering_factor  # Up to $10M licensing
        
        return revenues
    
    def calculate_crew_compensation(self, config: CrewConfiguration) -> float:
        """Calculate crew compensation based on roles and mission duration."""
        compensation_rates = {  # Daily rates
            "command": 2000,
            "engineering": 1500,
            "medical": 1800,
            "science": 1400,
            "maintenance": 1200,
            "security": 1300,
            "support": 1000
        }
        
        total_compensation = 0
        for role in ["command", "engineering", "medical", "science", 
                    "maintenance", "security", "support"]:
            crew_count = getattr(config, role)
            daily_rate = compensation_rates[role]
            total_compensation += crew_count * daily_rate * config.mission_duration_days
        
        # Hazard pay (10% for interstellar missions)
        hazard_multiplier = 1.1
        return total_compensation * hazard_multiplier
    
    def calculate_mission_risk(self, config: CrewConfiguration) -> float:
        """Calculate mission risk factor for insurance/cost adjustments."""
        risk_factors = {
            "crew_size_risk": max(0, (config.total_crew - 50) / 50 * 0.2),  # Risk increases with size
            "passenger_risk": config.passengers / config.total_crew * 0.3,  # Passenger safety risk
            "medical_adequacy": max(0, (0.08 - config.medical / config.total_crew) * 2),  # Medical shortage risk
            "engineering_adequacy": max(0, (0.15 - config.engineering / config.total_crew) * 1.5),  # Engineering shortage risk
        }
        
        total_risk = sum(risk_factors.values())
        return min(total_risk, 1.0)  # Cap at 100% risk premium
    
    def calculate_safety_score(self, config: CrewConfiguration) -> float:
        """Calculate safety score (0-100) for crew configuration."""
        safety_components = {}
        
        # Medical coverage
        medical_ratio = config.medical / config.total_crew
        safety_components["medical"] = min(100, (medical_ratio / 0.10) * 30)  # 30 points max
        
        # Engineering redundancy
        engineering_ratio = config.engineering / config.total_crew
        safety_components["engineering"] = min(100, (engineering_ratio / 0.20) * 25)  # 25 points max
        
        # Command structure
        command_ratio = config.command / config.total_crew
        optimal_command = 0.08  # 8% optimal command ratio
        safety_components["command"] = min(100, (1 - abs(command_ratio - optimal_command) / optimal_command) * 20)
        
        # Security adequacy
        security_ratio = config.security / config.total_crew
        min_security = 0.05  # 5% minimum security
        safety_components["security"] = min(100, (security_ratio / min_security) * 15)  # 15 points max
        
        # Passenger safety (lower is safer for operations)
        passenger_ratio = config.passengers / config.total_crew
        safety_components["passenger_safety"] = max(0, (1 - passenger_ratio) * 10)  # 10 points max
        
        return sum(safety_components.values())
    
    def calculate_efficiency_score(self, config: CrewConfiguration) -> float:
        """Calculate operational efficiency score (0-100)."""
        efficiency_components = {}
        
        # Role distribution efficiency
        total_operational = (config.command + config.engineering + config.medical + 
                           config.science + config.maintenance + config.security + config.support)
        operational_ratio = total_operational / config.total_crew
        efficiency_components["operational_ratio"] = operational_ratio * 40  # 40 points max
        
        # Mission-specific efficiency
        mission_reqs = self.mission_requirements[config.mission_type]
        
        # Science adequacy for mission
        science_ratio = config.science / config.total_crew
        min_science = mission_reqs["min_science"]
        if science_ratio >= min_science:
            efficiency_components["science_adequacy"] = min(20, (science_ratio / min_science) * 20)
        else:
            efficiency_components["science_adequacy"] = (science_ratio / min_science) * 20
        
        # Engineering adequacy
        engineering_ratio = config.engineering / config.total_crew
        min_engineering = mission_reqs["min_engineering"]
        if engineering_ratio >= min_engineering:
            efficiency_components["engineering_adequacy"] = min(20, (engineering_ratio / min_engineering) * 20)
        else:
            efficiency_components["engineering_adequacy"] = (engineering_ratio / min_engineering) * 20
        
        # Passenger optimization (mission-dependent)
        passenger_ratio = config.passengers / config.total_crew
        optimal_passenger = mission_reqs["passenger_ratio"]
        passenger_efficiency = 1 - abs(passenger_ratio - optimal_passenger) / optimal_passenger
        efficiency_components["passenger_optimization"] = passenger_efficiency * 20  # 20 points max
        
        return sum(efficiency_components.values())
    
    def evaluate_crew_configuration(self, config: CrewConfiguration) -> EconomicResults:
        """Comprehensive evaluation of crew configuration."""
        # Calculate costs and revenues
        cost_breakdown = self.calculate_total_cost(config)
        revenue_breakdown = self.calculate_total_revenue(config)
        
        total_cost = sum(cost_breakdown.values())
        total_revenue = sum(revenue_breakdown.values())
        net_profit = total_revenue - total_cost
        roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0
        
        # Calculate scores
        safety_score = self.calculate_safety_score(config)
        efficiency_score = self.calculate_efficiency_score(config)
        
        # Risk assessment
        mission_risk = self.calculate_mission_risk(config)
        risk_assessment = {
            "mission_risk": mission_risk,
            "financial_risk": max(0, -roi / 100),  # Negative ROI as financial risk
            "operational_risk": (100 - efficiency_score) / 100,
            "safety_risk": (100 - safety_score) / 100
        }
        
        return EconomicResults(
            total_cost=total_cost,
            total_revenue=total_revenue,
            net_profit=net_profit,
            roi=roi,
            cost_breakdown=cost_breakdown,
            revenue_breakdown=revenue_breakdown,
            safety_score=safety_score,
            efficiency_score=efficiency_score,
            risk_assessment=risk_assessment
        )
    
    def monte_carlo_analysis(self, config: CrewConfiguration, 
                           n_simulations: int = 10000) -> Dict[str, any]:
        """
        Monte Carlo simulation for economic uncertainty analysis.
        
        Args:
            config: Base crew configuration
            n_simulations: Number of simulation iterations
            
        Returns:
            Dictionary with simulation results and confidence intervals
        """
        logger.info(f"Running Monte Carlo analysis with {n_simulations} simulations")
        
        # Store results
        roi_results = []
        profit_results = []
        cost_results = []
        revenue_results = []
        
        # Parameter uncertainty ranges (±20% typical)
        uncertainty_factor = 0.20
        
        for i in range(n_simulations):
            # Create modified economic parameters with uncertainty
            modified_params = EconomicParameters()
            
            # Apply random variations to key parameters
            modified_params.passenger_revenue_per_day *= (1 + np.random.normal(0, uncertainty_factor))
            modified_params.life_support_cost_per_person *= (1 + np.random.normal(0, uncertainty_factor))
            modified_params.crew_training_cost *= (1 + np.random.normal(0, uncertainty_factor))
            modified_params.mission_insurance_base *= (1 + np.random.normal(0, uncertainty_factor))
            
            # Temporarily replace parameters
            original_params = self.economic_params
            self.economic_params = modified_params
            
            # Evaluate configuration
            result = self.evaluate_crew_configuration(config)
            
            # Store results
            roi_results.append(result.roi)
            profit_results.append(result.net_profit)
            cost_results.append(result.total_cost)
            revenue_results.append(result.total_revenue)
            
            # Restore original parameters
            self.economic_params = original_params
        
        # Calculate confidence intervals
        roi_array = np.array(roi_results)
        profit_array = np.array(profit_results)
        
        monte_carlo_results = {
            "n_simulations": n_simulations,
            "roi": {
                "mean": np.mean(roi_array),
                "std": np.std(roi_array),
                "ci_95_lower": np.percentile(roi_array, 2.5),
                "ci_95_upper": np.percentile(roi_array, 97.5),
                "probability_positive": np.mean(roi_array > 0)
            },
            "profit": {
                "mean": np.mean(profit_array),
                "std": np.std(profit_array),
                "ci_95_lower": np.percentile(profit_array, 2.5),
                "ci_95_upper": np.percentile(profit_array, 97.5),
                "probability_positive": np.mean(profit_array > 0)
            },
            "cost_statistics": {
                "mean": np.mean(cost_results),
                "std": np.std(cost_results)
            },
            "revenue_statistics": {
                "mean": np.mean(revenue_results),
                "std": np.std(revenue_results)
            }
        }
        
        self.monte_carlo_results = monte_carlo_results
        logger.info(f"Monte Carlo analysis complete. Mean ROI: {monte_carlo_results['roi']['mean']:.2f}%")
        
        return monte_carlo_results
    
    def optimize_crew_size(self, mission_type: MissionType, 
                          mission_duration: int = 90,
                          optimization_objective: str = "roi") -> Tuple[CrewConfiguration, EconomicResults]:
        """
        Optimize crew size and distribution for maximum ROI or other objectives.
        
        Args:
            mission_type: Type of mission for optimization
            mission_duration: Mission duration in days
            optimization_objective: "roi", "profit", "safety", or "efficiency"
            
        Returns:
            Tuple of optimal configuration and economic results
        """
        logger.info(f"Optimizing crew for {mission_type.value} mission, objective: {optimization_objective}")
        
        mission_reqs = self.mission_requirements[mission_type]
        
        def objective_function(x):
            """Objective function for optimization."""
            try:
                # Convert optimization variables to crew configuration
                total_crew = int(x[0])
                command = max(1, int(x[1] * total_crew))  # At least 1 command
                engineering = max(1, int(x[2] * total_crew))  # At least 1 engineering
                medical = max(1, int(x[3] * total_crew))  # At least 1 medical
                science = int(x[4] * total_crew)
                maintenance = int(x[5] * total_crew)
                security = int(x[6] * total_crew)
                support = int(x[7] * total_crew)
                
                # Calculate passengers (remaining crew)
                operational_crew = command + engineering + medical + science + maintenance + security + support
                passengers = max(0, total_crew - operational_crew)
                
                # Ensure we don't exceed total crew
                if operational_crew > total_crew:
                    # Scale down proportionally
                    scale_factor = (total_crew - 1) / operational_crew  # Keep at least 1 for command
                    command = max(1, int(command * scale_factor))
                    engineering = max(1, int(engineering * scale_factor))
                    medical = max(1, int(medical * scale_factor))
                    science = int(science * scale_factor)
                    maintenance = int(maintenance * scale_factor)
                    security = int(security * scale_factor)
                    support = int(support * scale_factor)
                    passengers = total_crew - (command + engineering + medical + science + maintenance + security + support)
                
                config = CrewConfiguration(
                    total_crew=total_crew,
                    command=command,
                    engineering=engineering,
                    medical=medical,
                    science=science,
                    maintenance=maintenance,
                    security=security,
                    passengers=passengers,
                    support=support,
                    mission_type=mission_type,
                    mission_duration_days=mission_duration
                )
                
                result = self.evaluate_crew_configuration(config)
                
                # Return negative value for minimization
                if optimization_objective == "roi":
                    return -result.roi
                elif optimization_objective == "profit":
                    return -result.net_profit
                elif optimization_objective == "safety":
                    return -result.safety_score
                elif optimization_objective == "efficiency":
                    return -result.efficiency_score
                else:
                    # Combined objective (weighted average)
                    combined_score = (result.roi * 0.4 + result.safety_score * 0.3 + 
                                    result.efficiency_score * 0.3)
                    return -combined_score
                    
            except Exception as e:
                logger.warning(f"Optimization iteration failed: {e}")
                return 1e6  # Large penalty for invalid configurations
        
        # Define bounds for optimization variables
        bounds = [
            (10, 100),  # Total crew size
            (0.02, 0.15),  # Command ratio (2-15%)
            (mission_reqs["min_engineering"], 0.35),  # Engineering ratio
            (mission_reqs["min_medical"], 0.20),  # Medical ratio
            (mission_reqs["min_science"], 0.35),  # Science ratio
            (0.05, 0.20),  # Maintenance ratio
            (0.02, 0.15),  # Security ratio
            (0.02, 0.15),  # Support ratio
        ]
        
        # Use differential evolution for global optimization
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=300,
            popsize=20,
            seed=42,
            atol=1e-6,
            tol=1e-6
        )
        
        if result.success:
            # Extract optimal configuration
            x_opt = result.x
            total_crew = int(x_opt[0])
            command = max(1, int(x_opt[1] * total_crew))
            engineering = max(1, int(x_opt[2] * total_crew))
            medical = max(1, int(x_opt[3] * total_crew))
            science = int(x_opt[4] * total_crew)
            maintenance = int(x_opt[5] * total_crew)
            security = int(x_opt[6] * total_crew)
            support = int(x_opt[7] * total_crew)
            
            operational_crew = command + engineering + medical + science + maintenance + security + support
            passengers = max(0, total_crew - operational_crew)
            
            optimal_config = CrewConfiguration(
                total_crew=total_crew,
                command=command,
                engineering=engineering,
                medical=medical,
                science=science,
                maintenance=maintenance,
                security=security,
                passengers=passengers,
                support=support,
                mission_type=mission_type,
                mission_duration_days=mission_duration
            )
            
            optimal_results = self.evaluate_crew_configuration(optimal_config)
            
            logger.info(f"Optimization successful! Optimal crew size: {total_crew}, ROI: {optimal_results.roi:.2f}%")
            
            return optimal_config, optimal_results
        else:
            logger.error("Optimization failed to converge")
            raise RuntimeError("Crew optimization failed to converge")
    
    def pareto_frontier_analysis(self, mission_type: MissionType, 
                                mission_duration: int = 90,
                                n_points: int = 50) -> pd.DataFrame:
        """
        Generate Pareto frontier for multi-objective optimization.
        
        Analyzes trade-offs between ROI, safety, and efficiency.
        """
        logger.info(f"Generating Pareto frontier with {n_points} points")
        
        pareto_results = []
        
        # Generate combinations of objectives weights
        for i in range(n_points):
            # Random weights for ROI, safety, and efficiency
            weights = np.random.dirichlet([1, 1, 1])  # Sum to 1
            roi_weight, safety_weight, efficiency_weight = weights
            
            try:
                # Define weighted objective function
                def weighted_objective(x):
                    try:
                        total_crew = int(x[0])
                        command = max(1, int(x[1] * total_crew))
                        engineering = max(1, int(x[2] * total_crew))
                        medical = max(1, int(x[3] * total_crew))
                        science = int(x[4] * total_crew)
                        maintenance = int(x[5] * total_crew)
                        security = int(x[6] * total_crew)
                        support = int(x[7] * total_crew)
                        
                        operational_crew = command + engineering + medical + science + maintenance + security + support
                        passengers = max(0, total_crew - operational_crew)
                        
                        if operational_crew > total_crew:
                            scale_factor = (total_crew - 1) / operational_crew
                            command = max(1, int(command * scale_factor))
                            engineering = max(1, int(engineering * scale_factor))
                            medical = max(1, int(medical * scale_factor))
                            science = int(science * scale_factor)
                            maintenance = int(maintenance * scale_factor)
                            security = int(security * scale_factor)
                            support = int(support * scale_factor)
                            passengers = total_crew - (command + engineering + medical + science + maintenance + security + support)
                        
                        config = CrewConfiguration(
                            total_crew=total_crew,
                            command=command,
                            engineering=engineering,
                            medical=medical,
                            science=science,
                            maintenance=maintenance,
                            security=security,
                            passengers=passengers,
                            support=support,
                            mission_type=mission_type,
                            mission_duration_days=mission_duration
                        )
                        
                        result = self.evaluate_crew_configuration(config)
                        
                        # Normalize scores to 0-1 range for weighting
                        normalized_roi = max(0, result.roi) / 100  # Assuming max ROI around 100%
                        normalized_safety = result.safety_score / 100
                        normalized_efficiency = result.efficiency_score / 100
                        
                        weighted_score = (roi_weight * normalized_roi + 
                                        safety_weight * normalized_safety + 
                                        efficiency_weight * normalized_efficiency)
                        
                        return -weighted_score  # Negative for minimization
                        
                    except Exception:
                        return 1e6
                
                # Optimization bounds
                mission_reqs = self.mission_requirements[mission_type]
                bounds = [
                    (10, 100),  # Total crew
                    (0.02, 0.15),  # Command ratio
                    (mission_reqs["min_engineering"], 0.35),  # Engineering ratio
                    (mission_reqs["min_medical"], 0.20),  # Medical ratio
                    (mission_reqs["min_science"], 0.35),  # Science ratio
                    (0.05, 0.20),  # Maintenance ratio
                    (0.02, 0.15),  # Security ratio
                    (0.02, 0.15),  # Support ratio
                ]
                
                result = differential_evolution(
                    weighted_objective,
                    bounds,
                    maxiter=100,
                    popsize=15,
                    seed=42 + i
                )
                
                if result.success:
                    x_opt = result.x
                    total_crew = int(x_opt[0])
                    command = max(1, int(x_opt[1] * total_crew))
                    engineering = max(1, int(x_opt[2] * total_crew))
                    medical = max(1, int(x_opt[3] * total_crew))
                    science = int(x_opt[4] * total_crew)
                    maintenance = int(x_opt[5] * total_crew)
                    security = int(x_opt[6] * total_crew)
                    support = int(x_opt[7] * total_crew)
                    
                    operational_crew = command + engineering + medical + science + maintenance + security + support
                    passengers = max(0, total_crew - operational_crew)
                    
                    config = CrewConfiguration(
                        total_crew=total_crew,
                        command=command,
                        engineering=engineering,
                        medical=medical,
                        science=science,
                        maintenance=maintenance,
                        security=security,
                        passengers=passengers,
                        support=support,
                        mission_type=mission_type,
                        mission_duration_days=mission_duration
                    )
                    
                    economic_result = self.evaluate_crew_configuration(config)
                    
                    pareto_results.append({
                        "total_crew": total_crew,
                        "command": command,
                        "engineering": engineering,
                        "medical": medical,
                        "science": science,
                        "maintenance": maintenance,
                        "security": security,
                        "passengers": passengers,
                        "support": support,
                        "roi": economic_result.roi,
                        "safety_score": economic_result.safety_score,
                        "efficiency_score": economic_result.efficiency_score,
                        "total_cost": economic_result.total_cost,
                        "total_revenue": economic_result.total_revenue,
                        "net_profit": economic_result.net_profit,
                        "roi_weight": roi_weight,
                        "safety_weight": safety_weight,
                        "efficiency_weight": efficiency_weight
                    })
                    
            except Exception as e:
                logger.warning(f"Pareto point {i} failed: {e}")
                continue
        
        pareto_df = pd.DataFrame(pareto_results)
        logger.info(f"Pareto frontier analysis complete with {len(pareto_df)} valid points")
        
        return pareto_df
    
    def generate_optimization_report(self, config: CrewConfiguration, 
                                   results: EconomicResults,
                                   monte_carlo_results: Optional[Dict] = None,
                                   save_path: Optional[str] = None) -> Dict[str, any]:
        """Generate comprehensive optimization report."""
        
        report = {
            "optimization_summary": {
                "mission_type": config.mission_type.value,
                "mission_duration_days": config.mission_duration_days,
                "total_crew": config.total_crew,
                "optimization_date": pd.Timestamp.now().isoformat()
            },
            "crew_configuration": {
                "command": config.command,
                "engineering": config.engineering,
                "medical": config.medical,
                "science": config.science,
                "maintenance": config.maintenance,
                "security": config.security,
                "passengers": config.passengers,
                "support": config.support
            },
            "economic_results": {
                "total_cost_usd": results.total_cost,
                "total_revenue_usd": results.total_revenue,
                "net_profit_usd": results.net_profit,
                "roi_percent": results.roi,
                "cost_breakdown": results.cost_breakdown,
                "revenue_breakdown": results.revenue_breakdown
            },
            "performance_metrics": {
                "safety_score": results.safety_score,
                "efficiency_score": results.efficiency_score,
                "risk_assessment": results.risk_assessment
            }
        }
        
        if monte_carlo_results:
            report["monte_carlo_analysis"] = monte_carlo_results
            
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Optimization report saved to {save_path}")
            
        return report


def demonstrate_crew_optimization():
    """Demonstration of the crew economic optimization framework."""
    print("\n" + "="*80)
    print("CREW ECONOMIC OPTIMIZER - DEMONSTRATION")
    print("Enhanced Simulation Hardware Abstraction Framework")
    print("="*80)
    
    # Initialize optimizer
    optimizer = CrewEconomicOptimizer()
    
    # Test different mission types
    mission_types = [
        MissionType.SCIENTIFIC_EXPLORATION,
        MissionType.TOURISM,
        MissionType.CARGO_TRANSPORT
    ]
    
    optimization_results = {}
    
    for mission_type in mission_types:
        print(f"\n🚀 Optimizing crew for {mission_type.value.replace('_', ' ').title()} Mission")
        print("-" * 60)
        
        try:
            # Optimize crew configuration
            optimal_config, optimal_results = optimizer.optimize_crew_size(
                mission_type=mission_type,
                mission_duration=90,
                optimization_objective="roi"
            )
            
            # Run Monte Carlo analysis
            print("Running Monte Carlo analysis...")
            mc_results = optimizer.monte_carlo_analysis(optimal_config, n_simulations=1000)
            
            # Display results
            print(f"\n✅ OPTIMAL CONFIGURATION:")
            print(f"   Total Crew: {optimal_config.total_crew}")
            print(f"   Command: {optimal_config.command}, Engineering: {optimal_config.engineering}")
            print(f"   Medical: {optimal_config.medical}, Science: {optimal_config.science}")
            print(f"   Maintenance: {optimal_config.maintenance}, Security: {optimal_config.security}")
            print(f"   Passengers: {optimal_config.passengers}, Support: {optimal_config.support}")
            
            print(f"\n📊 ECONOMIC RESULTS:")
            print(f"   ROI: {optimal_results.roi:.2f}%")
            print(f"   Net Profit: ${optimal_results.net_profit/1e6:.2f}M")
            print(f"   Total Cost: ${optimal_results.total_cost/1e6:.2f}M")
            print(f"   Total Revenue: ${optimal_results.total_revenue/1e6:.2f}M")
            
            print(f"\n🎯 PERFORMANCE SCORES:")
            print(f"   Safety Score: {optimal_results.safety_score:.1f}/100")
            print(f"   Efficiency Score: {optimal_results.efficiency_score:.1f}/100")
            
            print(f"\n📈 MONTE CARLO CONFIDENCE (95% CI):")
            print(f"   ROI Range: {mc_results['roi']['ci_95_lower']:.2f}% to {mc_results['roi']['ci_95_upper']:.2f}%")
            print(f"   Probability of Positive ROI: {mc_results['roi']['probability_positive']*100:.1f}%")
            
            optimization_results[mission_type.value] = {
                "config": optimal_config,
                "results": optimal_results,
                "monte_carlo": mc_results
            }
            
        except Exception as e:
            print(f"❌ Optimization failed for {mission_type.value}: {e}")
    
    print(f"\n🎉 CREW OPTIMIZATION DEMONSTRATION COMPLETE")
    print(f"Successfully optimized {len(optimization_results)} mission types")
    print("="*80)
    
    return optimization_results


if __name__ == "__main__":
    # Run demonstration
    results = demonstrate_crew_optimization()
