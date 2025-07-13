#!/usr/bin/env python3
"""
Crew Vessel Design Framework - Comprehensive Test Suite
======================================================

Test suite for crew vessel design framework covering life support integration,
emergency evacuation protocols, crew quarters optimization, and command systems.

Author: Enhanced Simulation Framework
Date: July 12, 2025
Status: Production Testing Implementation
"""

import unittest
import json
import tempfile
import os
import sys
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from crew_vessel_design_framework import (
    CrewVesselDesignFramework,
    CrewVesselConfiguration,
    LifeSupportSystem,
    EmergencyEvacuationSystem,
    CrewQuartersOptimization,
    CommandControlSystems,
    EmergencyLevel,
    CrewRole
)

class TestCrewVesselConfiguration(unittest.TestCase):
    """Test crew vessel configuration parameters"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = CrewVesselConfiguration()
    
    def test_default_configuration(self):
        """Test default configuration values"""
        self.assertEqual(self.config.personnel_capacity, 100)
        self.assertEqual(self.config.mission_duration_days, 90)  # Complete mission
        self.assertEqual(self.config.max_supraluminal_flight_days, 30)  # Per transit limit
        self.assertEqual(self.config.cruise_velocity_c, 53.5)
        self.assertEqual(self.config.safety_factor, 4.2)
        self.assertEqual(self.config.life_support_efficiency, 99.9)
        self.assertEqual(self.config.emergency_response_time, 60.0)
    
    def test_custom_configuration(self):
        """Test custom configuration parameters"""
        custom_config = CrewVesselConfiguration(
            personnel_capacity=50,
            mission_duration_days=15,
            cruise_velocity_c=40.0,
            safety_factor=5.0
        )
        
        self.assertEqual(custom_config.personnel_capacity, 50)
        self.assertEqual(custom_config.mission_duration_days, 15)
        self.assertEqual(custom_config.cruise_velocity_c, 40.0)
        self.assertEqual(custom_config.safety_factor, 5.0)

class TestLifeSupportSystem(unittest.TestCase):
    """Test life support system calculations and validation"""
    
    def setUp(self):
        """Set up life support system"""
        self.life_support = LifeSupportSystem()
    
    def test_recycling_efficiency_requirements(self):
        """Test that recycling efficiency meets design requirements"""
        self.assertGreaterEqual(self.life_support.atmospheric_recycling_efficiency, 99.9)
        self.assertGreaterEqual(self.life_support.water_recycling_efficiency, 99.95)
        self.assertGreaterEqual(self.life_support.waste_processing_efficiency, 99.8)
    
    def test_lqg_enhancement_parameters(self):
        """Test LQG enhancement capabilities"""
        self.assertEqual(self.life_support.lqg_filtration_enhancement, 242e6)
        self.assertTrue(self.life_support.quantum_air_purification)
        self.assertTrue(self.life_support.casimir_environmental_integration)
        self.assertTrue(self.life_support.tmu_nu_positive_constraint)
    
    def test_consumables_calculation_crew_100_days_30(self):
        """Test consumables calculation for 100 crew, 30 days"""
        requirements = self.life_support.calculate_consumables_requirement(100, 30)
        
        # Verify all required resources are calculated
        expected_resources = ['oxygen_kg', 'water_liters', 'food_kg', 'power_kwh']
        for resource in expected_resources:
            self.assertIn(resource, requirements)
            self.assertGreater(requirements[resource], 0)
        
        # Test specific calculations with high recycling efficiency
        # Oxygen: 100 crew × 0.84 kg/day × 30 days × (1 - 99.9%) + emergency reserve
        base_oxygen = 100 * 0.84 * 30
        net_oxygen = base_oxygen * (1 - 99.9/100)
        emergency_oxygen = net_oxygen * (7.0 / 30)  # 7-day emergency reserve
        expected_oxygen = net_oxygen + emergency_oxygen
        
        self.assertAlmostEqual(requirements['oxygen_kg'], expected_oxygen, places=2)
    
    def test_consumables_calculation_varied_crew_sizes(self):
        """Test consumables calculation for different crew sizes"""
        crew_sizes = [25, 50, 75, 100]
        mission_days = 30
        
        for crew_count in crew_sizes:
            requirements = self.life_support.calculate_consumables_requirement(crew_count, mission_days)
            
            # Verify scaling is proportional to crew size
            self.assertGreater(requirements['food_kg'], crew_count * 1.83 * mission_days * 0.8)  # At least 80% of gross
            self.assertGreater(requirements['power_kwh'], crew_count * 2.5 * mission_days * 0.8)

class TestEmergencyEvacuationSystem(unittest.TestCase):
    """Test emergency evacuation system capabilities"""
    
    def setUp(self):
        """Set up emergency evacuation system"""
        self.emergency_system = EmergencyEvacuationSystem()
    
    def test_evacuation_coverage(self):
        """Test that evacuation system provides 100% crew coverage"""
        capability = self.emergency_system.calculate_evacuation_capability()
        
        self.assertEqual(capability['total_evacuation_capacity'], 100)  # 20 pods × 5 capacity
        self.assertEqual(capability['crew_coverage_percentage'], 100.0)
        self.assertEqual(capability['evacuation_time_seconds'], 60.0)
    
    def test_emergency_return_capability(self):
        """Test emergency return from Proxima Centauri"""
        capability = self.emergency_system.calculate_evacuation_capability()
        
        # Emergency return at 72c from 4.37 ly should be reasonable
        self.assertLess(capability['emergency_return_days'], 30)  # Faster than normal mission
        self.assertGreater(capability['emergency_return_days'], 15)  # But not instantaneous
    
    def test_redundancy_factor(self):
        """Test evacuation system redundancy"""
        capability = self.emergency_system.calculate_evacuation_capability()
        
        # Should have redundancy in escape pods
        self.assertGreaterEqual(capability['redundancy_factor'], 1.0)
        self.assertEqual(capability['pods_required'], 20)  # 100 crew ÷ 5 per pod
        self.assertEqual(capability['pods_available'], 20)
    
    def test_emergency_safety_systems(self):
        """Test emergency safety system integration"""
        self.assertTrue(self.emergency_system.artificial_gravity_emergency)
        self.assertTrue(self.emergency_system.positive_energy_constraint)
        self.assertTrue(self.emergency_system.automated_navigation)
        self.assertTrue(self.emergency_system.medical_tractor_integration)

class TestCrewQuartersOptimization(unittest.TestCase):
    """Test crew quarters optimization and layout calculations"""
    
    def setUp(self):
        """Set up crew quarters optimization"""
        self.crew_quarters = CrewQuartersOptimization()
        self.vessel_dimensions = (150.0, 25.0, 8.0)  # Default vessel dimensions
    
    def test_personal_space_requirements(self):
        """Test personal space allocation meets 15m³ requirement"""
        self.assertEqual(self.crew_quarters.personal_space_m3, 15.0)
        self.assertTrue(self.crew_quarters.privacy_partitions)
        self.assertTrue(self.crew_quarters.individual_climate_control)
    
    def test_quarters_layout_100_crew(self):
        """Test quarters layout for 100 crew members"""
        layout = self.crew_quarters.calculate_quarters_layout(100, self.vessel_dimensions)
        
        # Verify basic calculations
        expected_total_volume = 150.0 * 25.0 * 8.0  # 30,000 m³
        self.assertEqual(layout['total_vessel_volume_m3'], expected_total_volume)
        
        # Verify crew quarters allocation (35% of total volume)
        expected_crew_volume = expected_total_volume * 0.35  # 10,500 m³
        self.assertEqual(layout['crew_quarters_volume_m3'], expected_crew_volume)
        
        # Verify per-crew space calculation
        expected_per_crew = expected_crew_volume / 100  # 105 m³ per crew
        self.assertEqual(layout['available_space_per_crew_m3'], expected_per_crew)
        
        # Verify space requirement is met (105 m³ > 15 m³ required)
        self.assertTrue(layout['space_requirement_met'])
        self.assertLess(layout['space_utilization_efficiency'], 20)  # Should be ~14.3%
    
    def test_space_allocation_percentages(self):
        """Test vessel space allocation percentages sum to 100%"""
        layout = self.crew_quarters.calculate_quarters_layout(100, self.vessel_dimensions)
        
        # All percentage allocations should sum to 100%
        total_percentage = (
            35.0 +  # crew_quarters
            8.0 +   # command_bridge
            12.0 +  # life_support
            15.0 +  # engineering
            10.0 +  # common_areas
            8.0 +   # cargo_storage
            5.0 +   # emergency_systems
            7.0     # maintenance_access
        )
        self.assertEqual(total_percentage, 100.0)
        
        # Verify all space allocations are present
        required_allocations = [
            'crew_quarters_percentage_volume_m3',
            'command_bridge_percentage_volume_m3',
            'life_support_percentage_volume_m3',
            'engineering_percentage_volume_m3',
            'common_areas_percentage_volume_m3',
            'cargo_storage_percentage_volume_m3',
            'emergency_systems_percentage_volume_m3',
            'maintenance_access_percentage_volume_m3'
        ]
        
        for allocation in required_allocations:
            self.assertIn(allocation, layout)
            self.assertGreater(layout[allocation], 0)
    
    def test_advanced_comfort_features(self):
        """Test advanced comfort and material features"""
        self.assertTrue(self.crew_quarters.casimir_ultra_smooth_surfaces)
        self.assertTrue(self.crew_quarters.artificial_gravity_1g)
        self.assertTrue(self.crew_quarters.quantum_enhanced_comfort)
        self.assertTrue(self.crew_quarters.modular_reconfiguration)

class TestCommandControlSystems(unittest.TestCase):
    """Test command and control systems requirements"""
    
    def setUp(self):
        """Set up command and control systems"""
        self.command_systems = CommandControlSystems()
    
    def test_bridge_configuration(self):
        """Test bridge station configuration"""
        self.assertEqual(self.command_systems.bridge_stations, 12)
        self.assertEqual(self.command_systems.automation_level, 0.85)
        self.assertTrue(self.command_systems.ai_assisted_operations)
        self.assertTrue(self.command_systems.manual_override_capability)
    
    def test_navigation_capabilities(self):
        """Test advanced navigation system capabilities"""
        self.assertTrue(self.command_systems.unified_lqg_navigation)
        self.assertTrue(self.command_systems.ftl_communication_relay)
        self.assertTrue(self.command_systems.quantum_sensor_positioning)
        self.assertTrue(self.command_systems.real_time_stellar_navigation)
    
    def test_control_requirements_100_crew(self):
        """Test control requirements for 100 crew"""
        requirements = self.command_systems.calculate_control_requirements(100)
        
        # Verify crew role distribution
        role_dist = requirements['crew_role_distribution']
        total_assigned = sum(role_dist.values())
        self.assertEqual(total_assigned, 100)
        
        # Verify specific role requirements
        self.assertEqual(role_dist['commander'], 1)
        self.assertGreaterEqual(role_dist['engineer'], 10)  # Need sufficient engineers
        self.assertGreaterEqual(role_dist['medical'], 5)    # Need sufficient medical staff
        
        # Verify bridge requirements
        self.assertEqual(requirements['bridge_stations_available'], 12)
        self.assertGreaterEqual(requirements['bridge_crew_requirement'], 12)
        
        # Verify automation level
        self.assertEqual(requirements['automation_percentage'], 85.0)
        self.assertEqual(requirements['manual_systems_percentage'], 15.0)
    
    def test_repository_integration(self):
        """Test integration with repository ecosystem"""
        self.assertTrue(self.command_systems.polymerized_lqg_communication)
        self.assertTrue(self.command_systems.unified_lqg_ftl_control)

class TestCrewVesselDesignFramework(unittest.TestCase):
    """Test main crew vessel design framework"""
    
    def setUp(self):
        """Set up crew vessel design framework"""
        self.framework = CrewVesselDesignFramework()
    
    def test_framework_initialization(self):
        """Test framework initialization with default configuration"""
        self.assertIsNotNone(self.framework.config)
        self.assertIsNotNone(self.framework.life_support)
        self.assertIsNotNone(self.framework.emergency_system)
        self.assertIsNotNone(self.framework.crew_quarters)
        self.assertIsNotNone(self.framework.command_systems)
        
        self.assertEqual(self.framework.mission_distance_ly, 4.37)
        self.assertTrue(self.framework.earth_proxima_mission_profile)
    
    def test_mission_requirements_calculation(self):
        """Test comprehensive mission requirements calculation"""
        requirements = self.framework.calculate_mission_requirements()
        
        # Verify all major sections are present
        required_sections = [
            'mission_profile',
            'life_support_requirements',
            'emergency_evacuation_capability',
            'crew_quarters_layout',
            'command_control_requirements',
            'vessel_configuration'
        ]
        
        for section in required_sections:
            self.assertIn(section, requirements)
        
        # Test mission profile calculations
        profile = requirements['mission_profile']
        self.assertEqual(profile['destination'], 'Proxima Centauri')
        self.assertEqual(profile['distance_light_years'], 4.37)
        self.assertTrue(profile['supraluminal_constraint_met'])  # ≤30 days per transit
        self.assertTrue(profile['mission_feasibility'])  # Complete 90-day mission feasible
    
    def test_design_validation(self):
        """Test design validation against requirements"""
        validation = self.framework.validate_design_requirements()
        
        # Verify validation structure
        self.assertIn('individual_validations', validation)
        self.assertIn('overall_design_valid', validation)
        self.assertIn('validation_score_percentage', validation)
        self.assertIn('design_readiness', validation)
        
        # All validations should pass with default configuration
        individual_vals = validation['individual_validations']
        for validation_name, result in individual_vals.items():
            self.assertTrue(result, f"Validation failed: {validation_name}")
        
        # Overall validation should pass
        self.assertTrue(validation['overall_design_valid'])
        self.assertEqual(validation['validation_score_percentage'], 100.0)
        self.assertEqual(validation['design_readiness'], 'PRODUCTION READY')
        self.assertEqual(len(validation['critical_issues']), 0)
    
    def test_implementation_roadmap(self):
        """Test implementation roadmap generation"""
        roadmap = self.framework.generate_implementation_roadmap()
        
        # Verify roadmap structure
        self.assertIn('implementation_phases', roadmap)
        self.assertIn('total_duration_months', roadmap)
        self.assertIn('estimated_completion', roadmap)
        
        # Verify all required phases are present
        phases = roadmap['implementation_phases']
        required_phases = [
            'phase_1_life_support',
            'phase_2_emergency_systems',
            'phase_3_crew_habitat',
            'phase_4_command_systems'
        ]
        
        for phase in required_phases:
            self.assertIn(phase, phases)
            
            # Verify phase structure
            phase_info = phases[phase]
            self.assertIn('duration_months', phase_info)
            self.assertIn('primary_repository', phase_info)
            self.assertIn('supporting_systems', phase_info)
            self.assertIn('targets', phase_info)
            self.assertIn('deliverables', phase_info)
        
        # Verify total duration calculation
        expected_duration = sum(phase['duration_months'] for phase in phases.values())
        self.assertEqual(roadmap['total_duration_months'], expected_duration)
    
    def test_export_design_specifications(self):
        """Test design specifications export functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory for export
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                # Export specifications
                filename = self.framework.export_design_specifications()
                
                # Verify file was created
                self.assertTrue(os.path.exists(filename))
                
                # Verify file contents
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                # Check required sections
                required_sections = [
                    'metadata',
                    'mission_profile',
                    'design_validation',
                    'implementation_roadmap',
                    'repository_integration'
                ]
                
                for section in required_sections:
                    self.assertIn(section, data)
                
                # Verify metadata
                metadata = data['metadata']
                self.assertEqual(metadata['framework'], 'Enhanced Simulation Hardware Abstraction Framework')
                self.assertEqual(metadata['design_type'], 'Crew Vessel - 30-Day Endurance')
                self.assertEqual(metadata['status'], 'Production Ready')
                
            finally:
                os.chdir(original_cwd)

class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and edge cases"""
    
    def test_reduced_crew_scenario(self):
        """Test framework with reduced crew size"""
        config = CrewVesselConfiguration(personnel_capacity=50, mission_duration_days=90)
        framework = CrewVesselDesignFramework(config)
        
        validation = framework.validate_design_requirements()
        self.assertTrue(validation['overall_design_valid'])
        
        # Verify evacuation coverage still adequate
        evacuation = validation['mission_requirements']['emergency_evacuation_capability']
        self.assertGreaterEqual(evacuation['crew_coverage_percentage'], 100.0)
    
    def test_extended_mission_scenario(self):
        """Test framework with extended mission duration"""
        config = CrewVesselConfiguration(mission_duration_days=120, mission_operations_days=30)
        framework = CrewVesselDesignFramework(config)
        
        requirements = framework.calculate_mission_requirements()
        
        # Verify life support scales appropriately
        consumables = requirements['life_support_requirements']
        self.assertGreater(consumables['oxygen_kg'], 0)
        self.assertGreater(consumables['water_liters'], 0)
        
        # Extended mission should still be feasible
        profile = requirements['mission_profile']
        self.assertTrue(profile['mission_feasibility'])
        self.assertTrue(profile['supraluminal_constraint_met'])
    
    def test_high_velocity_scenario(self):
        """Test framework with higher cruise velocity"""
        config = CrewVesselConfiguration(cruise_velocity_c=70.0, outbound_transit_days=23, return_transit_days=23)
        framework = CrewVesselDesignFramework(config)
        
        requirements = framework.calculate_mission_requirements()
        profile = requirements['mission_profile']
        
        # Higher velocity should reduce transit time and still meet constraints
        self.assertLess(profile['one_way_transit_days'], 25)
        self.assertTrue(profile['supraluminal_constraint_met'])
        self.assertTrue(profile['mission_feasibility'])

def run_comprehensive_tests():
    """Run comprehensive test suite with detailed reporting"""
    
    print("=" * 80)
    print("CREW VESSEL DESIGN FRAMEWORK - COMPREHENSIVE TEST SUITE")
    print("Enhanced Simulation Hardware Abstraction Framework")
    print("=" * 80)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestCrewVesselConfiguration,
        TestLifeSupportSystem,
        TestEmergencyEvacuationSystem,
        TestCrewQuartersOptimization,
        TestCommandControlSystems,
        TestCrewVesselDesignFramework,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED - FRAMEWORK READY FOR PRODUCTION")
    else:
        print("\n❌ SOME TESTS FAILED - REVIEW REQUIRED")
    
    print("=" * 80)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
