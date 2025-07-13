#!/usr/bin/env python3
"""
Mission Profile Integrator - Enhanced Simulation Hardware Abstraction Framework

Mission-specific crew optimization with dynamic adaptation and profile matching.

Author: Enhanced Simulation Hardware Abstraction Framework
Date: July 13, 2025
Version: 1.0.0 - Production Implementation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

from crew_economic_optimizer import MissionType, CrewConfiguration, CrewEconomicOptimizer
from crew_role_optimizer import CrewRoleOptimizer, RoleRequirements, RoleOptimizationResults

logger = logging.getLogger(__name__)

@dataclass
class MissionPhase:
    """Mission phase definition."""
    phase_id: str
    duration_days: int
    crew_requirements: Dict[str, int]
    critical_systems: List[str]
    risk_level: float
    workload_multiplier: float

@dataclass
class MissionProfile:
    """Complete mission profile."""
    mission_id: str
    mission_type: MissionType
    total_duration: int
    phases: List[MissionPhase]
    destination: str
    passenger_capacity: int
    cargo_capacity: float
    environmental_conditions: Dict[str, float]
    success_criteria: Dict[str, float]

@dataclass
class AdaptationStrategy:
    """Crew adaptation strategy."""
    trigger_conditions: Dict[str, float]
    adaptation_actions: List[str]
    resource_requirements: Dict[str, float]
    implementation_time: int
    success_probability: float

class MissionProfileIntegrator:
    """
    Mission-specific crew optimization with dynamic profile adaptation.
    """
    
    def __init__(self, economic_optimizer: CrewEconomicOptimizer, 
                 role_optimizer: CrewRoleOptimizer):
        """Initialize mission profile integrator."""
        self.economic_optimizer = economic_optimizer
        self.role_optimizer = role_optimizer
        self.mission_templates = self._initialize_mission_templates()
        
    def _initialize_mission_templates(self) -> Dict[MissionType, MissionProfile]:
        """Initialize standard mission templates."""
        templates = {}
        
        # Scientific Exploration Template
        templates[MissionType.SCIENTIFIC_EXPLORATION] = MissionProfile(
            mission_id="sci_exploration_template",
            mission_type=MissionType.SCIENTIFIC_EXPLORATION,
            total_duration=90,
            phases=[
                MissionPhase("departure", 15, {"engineering": 0.3, "command": 0.1}, 
                           ["propulsion", "navigation"], 0.6, 1.2),
                MissionPhase("transit", 30, {"maintenance": 0.2, "medical": 0.1}, 
                           ["life_support", "radiation_shielding"], 0.3, 0.8),
                MissionPhase("exploration", 30, {"science": 0.4, "engineering": 0.2}, 
                           ["sensors", "communications"], 0.8, 1.5),
                MissionPhase("return", 15, {"engineering": 0.3, "command": 0.1}, 
                           ["propulsion", "navigation"], 0.5, 1.1)
            ],
            destination="Proxima Centauri",
            passenger_capacity=20,
            cargo_capacity=50.0,
            environmental_conditions={"radiation": 0.7, "gravity": 0.1},
            success_criteria={"data_collection": 0.9, "crew_safety": 0.95}
        )
        
        # Tourism Template
        templates[MissionType.TOURISM] = MissionProfile(
            mission_id="tourism_template",
            mission_type=MissionType.TOURISM,
            total_duration=90,
            phases=[
                MissionPhase("departure", 10, {"support": 0.4, "medical": 0.15}, 
                           ["comfort_systems", "entertainment"], 0.4, 1.3),
                MissionPhase("transit", 35, {"support": 0.5, "medical": 0.12}, 
                           ["life_support", "comfort"], 0.2, 1.0),
                MissionPhase("destination", 30, {"support": 0.6, "security": 0.1}, 
                           ["safety_systems", "excursion_equipment"], 0.6, 1.4),
                MissionPhase("return", 15, {"engineering": 0.2, "support": 0.3}, 
                           ["propulsion", "comfort"], 0.3, 1.1)
            ],
            destination="Proxima Centauri Orbit",
            passenger_capacity=60,
            cargo_capacity=20.0,
            environmental_conditions={"comfort": 0.9, "safety": 0.95},
            success_criteria={"passenger_satisfaction": 0.92, "safety": 0.98}
        )
        
        return templates
    
    def optimize_for_mission_profile(self, profile: MissionProfile, 
                                   constraints: Optional[Dict] = None) -> Dict[str, any]:
        """Optimize crew for specific mission profile."""
        
        logger.info(f"Optimizing crew for mission: {profile.mission_id}")
        
        # Phase-specific optimization
        phase_results = {}
        for phase in profile.phases:
            phase_config = self._optimize_phase_crew(phase, profile)
            phase_results[phase.phase_id] = phase_config
        
        # Overall optimization
        overall_config = self._integrate_phase_requirements(phase_results, profile)
        
        # Economic analysis
        economic_results = self.economic_optimizer.evaluate_crew_configuration(overall_config)
        
        # Role optimization
        role_requirements = self.role_optimizer.generate_role_requirements(
            profile.mission_type, overall_config.total_crew
        )
        role_results = self.role_optimizer.optimize_role_assignments(
            overall_config, role_requirements
        )
        
        # Adaptation strategies
        adaptation_strategies = self._generate_adaptation_strategies(profile, overall_config)
        
        return {
            "mission_profile": profile,
            "optimal_configuration": overall_config,
            "economic_results": economic_results,
            "role_results": role_results,
            "phase_analysis": phase_results,
            "adaptation_strategies": adaptation_strategies
        }
    
    def _optimize_phase_crew(self, phase: MissionPhase, profile: MissionProfile) -> CrewConfiguration:
        """Optimize crew for specific mission phase."""
        
        base_crew = max(10, int(profile.passenger_capacity * 0.3))  # 30% crew minimum
        
        # Adjust based on phase requirements
        role_adjustments = {}
        for role, multiplier in phase.crew_requirements.items():
            if role == "engineering":
                role_adjustments["engineering"] = int(base_crew * multiplier * 0.3)
            elif role == "medical":
                role_adjustments["medical"] = int(base_crew * multiplier * 0.15)
            elif role == "science":
                role_adjustments["science"] = int(base_crew * multiplier * 0.25)
            elif role == "support":
                role_adjustments["support"] = int(base_crew * multiplier * 0.2)
            # Add other roles as needed
        
        # Create configuration with proper role balancing
        total_crew = min(100, base_crew + profile.passenger_capacity)
        
        # Calculate operational crew (excluding passengers)
        operational_crew = total_crew - profile.passenger_capacity
        
        command = max(1, int(operational_crew * 0.08))
        engineering = role_adjustments.get("engineering", int(operational_crew * 0.25))
        medical = role_adjustments.get("medical", int(operational_crew * 0.12))
        science = role_adjustments.get("science", int(operational_crew * 0.20))
        maintenance = max(1, int(operational_crew * 0.15))
        security = max(1, int(operational_crew * 0.08))
        support = role_adjustments.get("support", int(operational_crew * 0.12))
        
        # Ensure roles sum to operational crew
        role_sum = command + engineering + medical + science + maintenance + security + support
        if role_sum != operational_crew:
            # Adjust support to balance
            support = operational_crew - (command + engineering + medical + science + maintenance + security)
            support = max(0, support)
        
        config = CrewConfiguration(
            total_crew=total_crew,
            command=command,
            engineering=engineering,
            medical=medical,
            science=science,
            maintenance=maintenance,
            security=security,
            passengers=profile.passenger_capacity,
            support=support,
            mission_type=profile.mission_type,
            mission_duration_days=phase.duration_days
        )
        
        return config
    
    def _integrate_phase_requirements(self, phase_results: Dict, 
                                    profile: MissionProfile) -> CrewConfiguration:
        """Integrate phase requirements into overall crew configuration."""
        
        # Take maximum requirements across all phases
        max_requirements = {
            "command": 0, "engineering": 0, "medical": 0, "science": 0,
            "maintenance": 0, "security": 0, "support": 0
        }
        
        for phase_id, config in phase_results.items():
            max_requirements["command"] = max(max_requirements["command"], config.command)
            max_requirements["engineering"] = max(max_requirements["engineering"], config.engineering)
            max_requirements["medical"] = max(max_requirements["medical"], config.medical)
            max_requirements["science"] = max(max_requirements["science"], config.science)
            max_requirements["maintenance"] = max(max_requirements["maintenance"], config.maintenance)
            max_requirements["security"] = max(max_requirements["security"], config.security)
            max_requirements["support"] = max(max_requirements["support"], config.support)
        
        total_operational = sum(max_requirements.values())
        total_crew = min(100, total_operational + profile.passenger_capacity)
        
        return CrewConfiguration(
            total_crew=total_crew,
            command=max_requirements["command"],
            engineering=max_requirements["engineering"],
            medical=max_requirements["medical"],
            science=max_requirements["science"],
            maintenance=max_requirements["maintenance"],
            security=max_requirements["security"],
            passengers=profile.passenger_capacity,
            support=max_requirements["support"],
            mission_type=profile.mission_type,
            mission_duration_days=profile.total_duration
        )
    
    def _generate_adaptation_strategies(self, profile: MissionProfile, 
                                      config: CrewConfiguration) -> List[AdaptationStrategy]:
        """Generate adaptation strategies for mission contingencies."""
        
        strategies = []
        
        # Medical emergency strategy
        strategies.append(AdaptationStrategy(
            trigger_conditions={"medical_emergency": 1.0, "crew_injury": 0.8},
            adaptation_actions=[
                "Cross-train additional medical personnel",
                "Redistribute medical responsibilities",
                "Activate emergency medical protocols"
            ],
            resource_requirements={"training_time": 48, "medical_supplies": 1.5},
            implementation_time=24,
            success_probability=0.85
        ))
        
        # Engineering failure strategy
        strategies.append(AdaptationStrategy(
            trigger_conditions={"system_failure": 0.7, "maintenance_overload": 0.6},
            adaptation_actions=[
                "Activate cross-trained engineering backup",
                "Implement emergency repair protocols",
                "Redistribute engineering workload"
            ],
            resource_requirements={"spare_parts": 1.3, "engineering_hours": 2.0},
            implementation_time=12,
            success_probability=0.90
        ))
        
        return strategies


def demonstrate_mission_integration():
    """Demonstration of mission profile integration."""
    print("\n" + "="*60)
    print("MISSION PROFILE INTEGRATION - DEMONSTRATION")
    print("="*60)
    
    # Initialize components
    economic_optimizer = CrewEconomicOptimizer()
    role_optimizer = CrewRoleOptimizer(economic_optimizer)
    mission_integrator = MissionProfileIntegrator(economic_optimizer, role_optimizer)
    
    # Test with scientific exploration
    sci_profile = mission_integrator.mission_templates[MissionType.SCIENTIFIC_EXPLORATION]
    
    print(f"\n🚀 Optimizing for Scientific Exploration Mission")
    print(f"   Duration: {sci_profile.total_duration} days")
    print(f"   Phases: {len(sci_profile.phases)}")
    print(f"   Passenger Capacity: {sci_profile.passenger_capacity}")
    
    results = mission_integrator.optimize_for_mission_profile(sci_profile)
    
    config = results["optimal_configuration"]
    economic = results["economic_results"]
    
    print(f"\n✅ OPTIMIZATION RESULTS:")
    print(f"   Total Crew: {config.total_crew}")
    print(f"   ROI: {economic.roi:.2f}%")
    print(f"   Safety Score: {economic.safety_score:.1f}/100")
    print(f"   Adaptation Strategies: {len(results['adaptation_strategies'])}")
    
    print(f"\n🎉 MISSION INTEGRATION COMPLETE")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = demonstrate_mission_integration()
