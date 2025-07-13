#!/usr/bin/env python
"""
Test runner for unmanned probe design framework
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import unittest
from unmanned_probe_design_framework import UnmannedProbeDesignFramework, UnmannedProbeRequirements

class TestUnmannedProbeDesign(unittest.TestCase):
    def setUp(self):
        """Set up test framework"""
        # Create requirements for testing
        requirements = UnmannedProbeRequirements()
        requirements.max_velocity_c = 480.0  # 480c target
        requirements.mass_reduction_target = 0.99  # 99% mass reduction
        requirements.mission_duration_years = 1000  # Long-term missions
        requirements.autonomous_reliability = 0.9998  # 99.98% reliability
        
        self.framework = UnmannedProbeDesignFramework(requirements)
    
    def test_physics_validation(self):
        """Test zero exotic energy validation"""
        validation = self.framework.validate_physics_framework()
        
        print("\n=== Physics Framework Validation ===")
        print(f"Framework: {validation['physics_validation']['framework_basis']}")
        print(f"Exotic matter required: {validation['physics_validation']['exotic_matter_required']}")
        print(f"Energy type: {validation['physics_validation']['energy_type']}")
        print(f"Forbidden physics used: {validation['compliance_check']['forbidden_physics_used']}")
        
        # Assertions
        self.assertFalse(validation['physics_validation']['exotic_matter_required'])
        self.assertFalse(validation['physics_validation']['exotic_energy_required'])
        self.assertFalse(validation['compliance_check']['forbidden_physics_used'])
        self.assertTrue(validation['compliance_check']['physics_framework_valid'])
        
    def test_velocity_calculation(self):
        """Test 480c velocity achievement"""
        mass_reduction = 0.99  # 99% mass reduction
        velocity_analysis = self.framework.calculate_velocity_enhancement(mass_reduction)
        
        print("\n=== Velocity Enhancement Analysis ===")
        print(f"Physics framework: {velocity_analysis['physics_framework']}")
        print(f"Enhanced velocity: {velocity_analysis['enhanced_velocity_c']:.1f}c")
        print(f"LQG coupling efficiency: {velocity_analysis['lqg_coupling_efficiency']:.1f}x")
        print(f"Exotic matter required: {velocity_analysis['exotic_matter_required']}")
        
        # Verify 480c achievement with zero exotic matter (with floating point tolerance)
        self.assertAlmostEqual(velocity_analysis['enhanced_velocity_c'], 480.0, places=0)
        self.assertFalse(velocity_analysis['exotic_matter_required'])
        self.assertEqual(velocity_analysis['physics_framework'], 'Zero Exotic Energy LQG-based FTL')

if __name__ == '__main__':
    print("Running Unmanned Probe Design Framework Tests...")
    unittest.main(verbosity=2)
