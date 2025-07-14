#!/usr/bin/env python3
"""
Multi-Crew Vessel Architecture Integration Framework
Comprehensive framework for FTL vessel design supporting ≤100 personnel crews

Addresses UQ-VESSEL-001: Multi-Crew Vessel Architecture Integration Framework
Repository: enhanced-simulation-hardware-abstraction-framework
Priority: HIGH (Severity 2) - Direct dependency for crew vessel development
"""

import numpy as np
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VesselMode(Enum):
    """Vessel operational modes for convertible geometry"""
    PLANETARY_LANDING = "planetary_landing"
    IMPULSE_CRUISE = "impulse_cruise" 
    WARP_BUBBLE = "warp_bubble"
    EMERGENCY_CONFIGURATION = "emergency"

class CrewRole(Enum):
    """Crew role classifications for optimization"""
    COMMAND = "command"
    NAVIGATION = "navigation"
    ENGINEERING = "engineering"
    MEDICAL = "medical"
    SCIENCE = "science"
    SECURITY = "security"
    SUPPORT = "support"

@dataclass
class LifeSupportRequirements:
    """Life support system requirements for crew vessel"""
    crew_size: int
    mission_duration_days: int
    oxygen_consumption_per_person_day: float = 0.84  # kg/day
    water_consumption_per_person_day: float = 2.5   # kg/day
    food_consumption_per_person_day: float = 1.83   # kg/day
    waste_generation_per_person_day: float = 1.5    # kg/day
    power_consumption_per_person: float = 100       # W continuous
    
    def calculate_total_requirements(self) -> Dict[str, float]:
        """Calculate total mission requirements"""
        total_days = self.mission_duration_days * self.crew_size
        return {
            'total_oxygen_kg': total_days * self.oxygen_consumption_per_person_day,
            'total_water_kg': total_days * self.water_consumption_per_person_day,
            'total_food_kg': total_days * self.food_consumption_per_person_day,
            'total_waste_kg': total_days * self.waste_generation_per_person_day,
            'total_power_kw': self.crew_size * self.power_consumption_per_person / 1000,
            'total_mass_kg': total_days * (
                self.oxygen_consumption_per_person_day + 
                self.water_consumption_per_person_day + 
                self.food_consumption_per_person_day
            )
        }

@dataclass
class VesselGeometry:
    """Vessel geometry configuration for different operational modes"""
    mode: VesselMode
    length_m: float
    width_m: float
    height_m: float
    volume_m3: float
    mass_kg: float
    crew_volume_per_person_m3: float = 50  # Minimum habitable volume per crew member
    
    def validate_crew_accommodation(self, crew_size: int) -> bool:
        """Validate vessel can accommodate crew size"""
        required_crew_volume = crew_size * self.crew_volume_per_person_m3
        available_crew_volume = self.volume_m3 * 0.6  # 60% for crew, 40% for systems
        return available_crew_volume >= required_crew_volume

@dataclass
class SafetyProtocols:
    """Medical-grade safety protocols for crew operations"""
    graviton_safety_active: bool = True
    medical_tractor_array_active: bool = True
    emergency_response_time_ms: float = 25.0  # Maximum emergency response
    biological_safety_margin: float = 1e12    # WHO compliance margin
    positive_energy_constraint: bool = True    # T_μν ≥ 0 enforcement
    
    def validate_safety_requirements(self) -> Dict[str, bool]:
        """Validate all safety requirements are met"""
        return {
            'graviton_safety': self.graviton_safety_active,
            'medical_systems': self.medical_tractor_array_active,
            'emergency_response': self.emergency_response_time_ms <= 25.0,
            'biological_safety': self.biological_safety_margin >= 1e10,
            'energy_constraint': self.positive_energy_constraint
        }

class CrewOptimizer:
    """Crew complement optimization for mission efficiency"""
    
    def __init__(self):
        self.role_requirements = {
            CrewRole.COMMAND: {'min': 2, 'max': 5, 'priority': 1.0},
            CrewRole.NAVIGATION: {'min': 2, 'max': 4, 'priority': 0.9},
            CrewRole.ENGINEERING: {'min': 3, 'max': 15, 'priority': 0.95},
            CrewRole.MEDICAL: {'min': 2, 'max': 8, 'priority': 0.9},
            CrewRole.SCIENCE: {'min': 1, 'max': 20, 'priority': 0.7},
            CrewRole.SECURITY: {'min': 2, 'max': 10, 'priority': 0.6},
            CrewRole.SUPPORT: {'min': 2, 'max': 15, 'priority': 0.5}
        }
    
    def optimize_crew_distribution(self, total_crew: int, mission_type: str) -> Dict[CrewRole, int]:
        """Optimize crew role distribution for mission requirements"""
        logger.info(f"Optimizing crew distribution for {total_crew} personnel, mission: {mission_type}")
        
        # Mission-specific role priorities
        mission_modifiers = {
            'exploration': {CrewRole.SCIENCE: 1.5, CrewRole.ENGINEERING: 1.2},
            'transport': {CrewRole.COMMAND: 1.1, CrewRole.SUPPORT: 1.3},
            'emergency': {CrewRole.MEDICAL: 1.4, CrewRole.ENGINEERING: 1.3},
            'diplomatic': {CrewRole.COMMAND: 1.3, CrewRole.SECURITY: 1.2},
            'research': {CrewRole.SCIENCE: 1.6, CrewRole.MEDICAL: 1.1},
            'defense': {CrewRole.SECURITY: 1.5, CrewRole.ENGINEERING: 1.2}
        }
        
        # Apply mission modifiers
        adjusted_priorities = {}
        for role in CrewRole:
            base_priority = self.role_requirements[role]['priority']
            modifier = mission_modifiers.get(mission_type, {}).get(role, 1.0)
            adjusted_priorities[role] = base_priority * modifier
        
        # Allocate minimum required crew
        allocation = {}
        remaining_crew = total_crew
        
        for role in CrewRole:
            min_required = self.role_requirements[role]['min']
            allocation[role] = min_required
            remaining_crew -= min_required
        
        # Distribute remaining crew based on priorities
        while remaining_crew > 0:
            best_role = max(
                [role for role in CrewRole if allocation[role] < self.role_requirements[role]['max']],
                key=lambda r: adjusted_priorities[r],
                default=None
            )
            if best_role is None:
                break
            allocation[best_role] += 1
            remaining_crew -= 1
        
        logger.info(f"Optimized crew allocation: {allocation}")
        return allocation

class VesselConfigurationManager:
    """Manages convertible vessel geometry for different operational modes"""
    
    def __init__(self):
        self.configurations = {
            VesselMode.PLANETARY_LANDING: VesselGeometry(
                mode=VesselMode.PLANETARY_LANDING,
                length_m=120, width_m=80, height_m=25,
                volume_m3=240000, mass_kg=1500000
            ),
            VesselMode.IMPULSE_CRUISE: VesselGeometry(
                mode=VesselMode.IMPULSE_CRUISE,
                length_m=150, width_m=60, height_m=30,
                volume_m3=270000, mass_kg=1400000
            ),
            VesselMode.WARP_BUBBLE: VesselGeometry(
                mode=VesselMode.WARP_BUBBLE,
                length_m=200, width_m=40, height_m=40,
                volume_m3=320000, mass_kg=1300000
            ),
            VesselMode.EMERGENCY_CONFIGURATION: VesselGeometry(
                mode=VesselMode.EMERGENCY_CONFIGURATION,
                length_m=100, width_m=100, height_m=20,
                volume_m3=200000, mass_kg=1600000
            )
        }
    
    def get_optimal_configuration(self, crew_size: int, mission_phase: str) -> VesselGeometry:
        """Get optimal vessel configuration for current conditions"""
        logger.info(f"Selecting optimal configuration for crew size {crew_size}, phase: {mission_phase}")
        
        # Phase-specific configuration preferences
        phase_preferences = {
            'launch': VesselMode.PLANETARY_LANDING,
            'interplanetary': VesselMode.IMPULSE_CRUISE,
            'interstellar': VesselMode.WARP_BUBBLE,
            'landing': VesselMode.PLANETARY_LANDING,
            'emergency': VesselMode.EMERGENCY_CONFIGURATION
        }
        
        preferred_mode = phase_preferences.get(mission_phase, VesselMode.IMPULSE_CRUISE)
        configuration = self.configurations[preferred_mode]
        
        # Validate crew accommodation
        if not configuration.validate_crew_accommodation(crew_size):
            logger.warning(f"Configuration {preferred_mode} insufficient for {crew_size} crew")
            # Find alternative configuration
            for mode, config in self.configurations.items():
                if config.validate_crew_accommodation(crew_size):
                    logger.info(f"Alternative configuration selected: {mode}")
                    return config
            
            # If no configuration sufficient, use largest
            logger.warning("No configuration sufficient, using warp bubble (largest)")
            return self.configurations[VesselMode.WARP_BUBBLE]
        
        return configuration
    
    def calculate_reconfiguration_time(self, from_mode: VesselMode, to_mode: VesselMode) -> float:
        """Calculate time required for vessel reconfiguration (minutes)"""
        complexity_matrix = {
            (VesselMode.PLANETARY_LANDING, VesselMode.IMPULSE_CRUISE): 15,
            (VesselMode.IMPULSE_CRUISE, VesselMode.WARP_BUBBLE): 20,
            (VesselMode.WARP_BUBBLE, VesselMode.IMPULSE_CRUISE): 18,
            (VesselMode.IMPULSE_CRUISE, VesselMode.PLANETARY_LANDING): 12,
        }
        
        # Emergency reconfiguration is always fast
        if to_mode == VesselMode.EMERGENCY_CONFIGURATION:
            return 3.0
        
        return complexity_matrix.get((from_mode, to_mode), 25.0)

class MultiCrewVesselArchitecture:
    """Main vessel architecture integration framework"""
    
    def __init__(self, max_crew: int = 100, mission_duration_days: int = 30):
        self.max_crew = max_crew
        self.mission_duration_days = mission_duration_days
        self.crew_optimizer = CrewOptimizer()
        self.config_manager = VesselConfigurationManager()
        self.safety_protocols = SafetyProtocols()
        
        logger.info(f"Multi-Crew Vessel Architecture initialized for {max_crew} crew, {mission_duration_days} days")
    
    def design_vessel_for_mission(self, crew_size: int, mission_type: str, 
                                 mission_phases: List[str]) -> Dict:
        """Complete vessel design optimization for specific mission"""
        logger.info(f"Designing vessel for mission: {mission_type}, crew: {crew_size}")
        
        # Validate inputs
        if crew_size > self.max_crew:
            raise ValueError(f"Crew size {crew_size} exceeds maximum {self.max_crew}")
        
        # Optimize crew distribution
        crew_allocation = self.crew_optimizer.optimize_crew_distribution(crew_size, mission_type)
        
        # Calculate life support requirements
        life_support = LifeSupportRequirements(crew_size, self.mission_duration_days)
        life_support_totals = life_support.calculate_total_requirements()
        
        # Design configurations for each mission phase
        phase_configurations = {}
        reconfiguration_timeline = []
        
        for i, phase in enumerate(mission_phases):
            config = self.config_manager.get_optimal_configuration(crew_size, phase)
            phase_configurations[phase] = config
            
            if i > 0:
                prev_config = phase_configurations[mission_phases[i-1]]
                reconfig_time = self.config_manager.calculate_reconfiguration_time(
                    prev_config.mode, config.mode
                )
                reconfiguration_timeline.append({
                    'phase_transition': f"{mission_phases[i-1]} -> {phase}",
                    'reconfiguration_time_minutes': reconfig_time,
                    'from_mode': prev_config.mode.value,
                    'to_mode': config.mode.value
                })
        
        # Validate safety requirements
        safety_validation = self.safety_protocols.validate_safety_requirements()
        
        # Calculate operational efficiency metrics
        efficiency_metrics = self._calculate_operational_efficiency(
            crew_allocation, life_support_totals, phase_configurations
        )
        
        # Compile comprehensive design
        vessel_design = {
            'mission_parameters': {
                'mission_type': mission_type,
                'crew_size': crew_size,
                'mission_duration_days': self.mission_duration_days,
                'mission_phases': mission_phases
            },
            'crew_allocation': {role.value: count for role, count in crew_allocation.items()},
            'life_support_requirements': life_support_totals,
            'vessel_configurations': {
                phase: asdict(config) for phase, config in phase_configurations.items()
            },
            'reconfiguration_timeline': reconfiguration_timeline,
            'safety_validation': safety_validation,
            'operational_efficiency': efficiency_metrics,
            'design_validation': self._validate_complete_design(
                crew_size, life_support_totals, phase_configurations, safety_validation
            )
        }
        
        logger.info(f"Vessel design completed with efficiency score: {efficiency_metrics['overall_efficiency']:.3f}")
        return vessel_design
    
    def _calculate_operational_efficiency(self, crew_allocation: Dict, life_support: Dict, 
                                        configurations: Dict) -> Dict:
        """Calculate operational efficiency metrics"""
        # Crew efficiency (based on role optimization)
        total_crew = sum(crew_allocation.values())
        role_efficiency = sum(
            count / self.crew_optimizer.role_requirements[role]['max'] 
            for role, count in crew_allocation.items()
        ) / len(crew_allocation)
        
        # Mass efficiency (volume utilization)
        avg_volume = np.mean([config.volume_m3 for config in configurations.values()])
        volume_efficiency = min(1.0, (total_crew * 50) / (avg_volume * 0.6))  # 60% usable volume
        
        # Life support efficiency
        life_support_efficiency = 1.0 - (life_support['total_mass_kg'] / 100000) / total_crew
        life_support_efficiency = max(0.1, min(1.0, life_support_efficiency))
        
        # Overall efficiency (weighted average)
        overall_efficiency = (
            0.4 * role_efficiency + 
            0.3 * volume_efficiency + 
            0.3 * life_support_efficiency
        )
        
        return {
            'role_optimization_efficiency': role_efficiency,
            'volume_utilization_efficiency': volume_efficiency,
            'life_support_efficiency': life_support_efficiency,
            'overall_efficiency': overall_efficiency,
            'crew_density_per_m3': total_crew / avg_volume
        }
    
    def _validate_complete_design(self, crew_size: int, life_support: Dict, 
                                 configurations: Dict, safety: Dict) -> Dict:
        """Validate complete vessel design meets all requirements"""
        validations = {
            'crew_accommodation': all(
                config.validate_crew_accommodation(crew_size) 
                for config in configurations.values()
            ),
            'life_support_feasible': life_support['total_mass_kg'] < 500000,  # 500 ton limit
            'safety_compliant': all(safety.values()),
            'mission_duration_valid': self.mission_duration_days <= 30,
            'crew_size_valid': crew_size <= self.max_crew
        }
        
        overall_valid = all(validations.values())
        
        return {
            'individual_validations': validations,
            'overall_design_valid': overall_valid,
            'validation_score': sum(validations.values()) / len(validations),
            'critical_issues': [k for k, v in validations.items() if not v]
        }

def main():
    """Demonstrate multi-crew vessel architecture framework"""
    logger.info("=== Multi-Crew Vessel Architecture Integration Framework ===")
    
    # Initialize framework
    vessel_architect = MultiCrewVesselArchitecture(max_crew=100, mission_duration_days=30)
    
    # Test mission scenarios
    test_scenarios = [
        {
            'crew_size': 25,
            'mission_type': 'exploration',
            'mission_phases': ['launch', 'interstellar', 'exploration', 'return']
        },
        {
            'crew_size': 50,
            'mission_type': 'diplomatic',
            'mission_phases': ['launch', 'interstellar', 'diplomatic', 'return']
        },
        {
            'crew_size': 75,
            'mission_type': 'research',
            'mission_phases': ['launch', 'interstellar', 'research', 'extended_research', 'return']
        },
        {
            'crew_size': 100,
            'mission_type': 'transport',
            'mission_phases': ['launch', 'interstellar', 'transport', 'return']
        }
    ]
    
    results = {}
    
    for i, scenario in enumerate(test_scenarios):
        logger.info(f"\n--- Testing Scenario {i+1}: {scenario['mission_type']} ---")
        
        try:
            design = vessel_architect.design_vessel_for_mission(
                crew_size=scenario['crew_size'],
                mission_type=scenario['mission_type'],
                mission_phases=scenario['mission_phases']
            )
            
            results[f"scenario_{i+1}"] = design
            
            # Print key results
            efficiency = design['operational_efficiency']['overall_efficiency']
            validation = design['design_validation']['overall_design_valid']
            
            logger.info(f"Design completed: Efficiency {efficiency:.3f}, Valid: {validation}")
            
            if not validation:
                logger.warning(f"Critical issues: {design['design_validation']['critical_issues']}")
            
        except Exception as e:
            logger.error(f"Scenario {i+1} failed: {e}")
            results[f"scenario_{i+1}"] = {'error': str(e)}
    
    # Save results
    with open('multi_crew_vessel_architecture_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n=== Framework Testing Complete ===")
    logger.info(f"Results saved to: multi_crew_vessel_architecture_results.json")
    
    # Calculate overall framework performance
    valid_scenarios = [r for r in results.values() if 'error' not in r]
    if valid_scenarios:
        avg_efficiency = np.mean([
            r['operational_efficiency']['overall_efficiency'] 
            for r in valid_scenarios
        ])
        validation_rate = sum([
            r['design_validation']['overall_design_valid'] 
            for r in valid_scenarios
        ]) / len(valid_scenarios)
        
        logger.info(f"Framework Performance:")
        logger.info(f"  Average Efficiency: {avg_efficiency:.3f}")
        logger.info(f"  Validation Rate: {validation_rate:.1%}")
        logger.info(f"  Scenarios Processed: {len(valid_scenarios)}/{len(test_scenarios)}")
    
    return results

if __name__ == "__main__":
    main()
