"""
Test Framework for FTL-Capable Hull Design Implementation
========================================================

Comprehensive testing of naval architecture and advanced materials integration
Validates 48c operations, convertible geometry, and material specifications

Test Coverage:
- Naval architecture framework validation
- Advanced materials integration testing
- Vessel design specification verification
- Performance metrics validation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import unittest
from naval_architecture_framework import (
    NavalArchitectureFramework, VesselCategory, OperationalMode
)
from advanced_materials_integration import (
    AdvancedMaterialsFramework, AdvancedMaterialType
)

class TestFTLHullDesign(unittest.TestCase):
    """Test suite for FTL-capable hull design implementation"""
    
    def setUp(self):
        """Initialize test frameworks"""
        self.naval_framework = NavalArchitectureFramework()
        self.materials_framework = AdvancedMaterialsFramework()
        
    def test_naval_architecture_initialization(self):
        """Test naval architecture framework initialization"""
        # Verify vessel designs are loaded
        self.assertIn(VesselCategory.UNMANNED_PROBE, self.naval_framework.vessel_designs)
        self.assertIn(VesselCategory.CREW_VESSEL, self.naval_framework.vessel_designs)
        
        # Verify geometry systems are initialized
        self.assertIn("retractable_panels", self.naval_framework.geometry_systems)
        self.assertIn("dynamic_ballasting", self.naval_framework.geometry_systems)
        
    def test_advanced_materials_database(self):
        """Test advanced materials database"""
        materials = self.materials_framework.materials_database
        
        # Verify all required materials are present
        self.assertIn("plate_nanolattice", materials)
        self.assertIn("optimized_carbon", materials)
        self.assertIn("graphene_metamaterial", materials)
        
        # Verify material specifications meet FTL requirements
        for material in materials.values():
            self.assertGreaterEqual(material.ultimate_tensile_strength, 50.0)  # ≥50 GPa
            self.assertGreaterEqual(material.young_modulus, 1.0)  # ≥1 TPa
            self.assertGreaterEqual(material.vickers_hardness, 20.0)  # ≥20 GPa
            
    def test_48c_velocity_validation(self):
        """Test 48c velocity operations validation"""
        # Test all materials for 48c operations
        for material_name in self.materials_framework.materials_database.keys():
            validation = self.materials_framework.validate_48c_operations(material_name)
            
            # Verify successful validation
            self.assertTrue(validation.get("validated_for_48c", False))
            self.assertGreaterEqual(validation.get("safety_factor", 0), 3.0)
            
    def test_vessel_performance_analysis(self):
        """Test vessel performance analysis"""
        # Test crew vessel in all operational modes
        for mode in [OperationalMode.PLANETARY_LANDING, 
                    OperationalMode.IMPULSE_CRUISE, OperationalMode.WARP_BUBBLE]:
            
            analysis = self.naval_framework.analyze_vessel_performance(
                VesselCategory.CREW_VESSEL, mode
            )
            
            # Verify analysis completeness
            self.assertIn("overall_performance_score", analysis)
            self.assertIn("naval_architecture", analysis)
            self.assertIn("safety_analysis", analysis)
            
            # Verify performance meets requirements
            self.assertGreaterEqual(analysis["overall_performance_score"], 0.80)
            
    def test_convertible_geometry_performance(self):
        """Test convertible geometry system performance"""
        crew_vessel = self.naval_framework.vessel_designs[VesselCategory.CREW_VESSEL]
        geometry = crew_vessel.convertible_geometry
        
        # Verify transition time meets requirements (≤5 minutes)
        self.assertLessEqual(geometry.transition_time, 300.0)
        
        # Verify efficiency targets (≥90% all modes)
        for mode in geometry.mode_efficiency:
            self.assertGreaterEqual(geometry.mode_efficiency[mode], 0.90)
            
    def test_hull_material_specifications(self):
        """Test hull material specifications"""
        # Test crew vessel materials
        crew_spec = self.materials_framework.generate_hull_material_specification("crew_vessel")
        self.assertIn("material_assignment", crew_spec)
        self.assertIn("performance_validation", crew_spec)
        
        # Test probe materials  
        probe_spec = self.materials_framework.generate_hull_material_specification("unmanned_probe")
        self.assertIn("material_assignment", probe_spec)
        self.assertIn("performance_validation", probe_spec)
        
        # Verify 48c validation for both vessel types
        self.assertIn("48c_operations", crew_spec["performance_validation"])
        self.assertIn("48c_operations", probe_spec["performance_validation"])
        
    def test_safety_margins(self):
        """Test safety margin compliance"""
        for vessel_category in [VesselCategory.CREW_VESSEL, VesselCategory.UNMANNED_PROBE]:
            vessel = self.naval_framework.vessel_designs[vessel_category]
            
            # Verify structural safety factors ≥3.0
            self.assertGreaterEqual(vessel.safety_factors["structural"], 3.0)
            
            # Verify crew comfort limits (if applicable)
            if vessel.crew_capacity > 0:
                self.assertLessEqual(vessel.crew_comfort_limits["transition_acceleration"], 0.1)
                
    def test_manufacturing_feasibility(self):
        """Test manufacturing feasibility"""
        protocols = self.materials_framework.manufacturing_protocols
        
        # Verify all manufacturing technologies have feasibility scores
        for tech in protocols.values():
            self.assertIn("vessel_scale_feasibility", tech)
            self.assertGreater(tech["vessel_scale_feasibility"], 0.7)  # Minimum feasibility
            
    def test_design_optimization(self):
        """Test vessel design optimization"""
        target_performance = {
            "efficiency_target": 0.90,
            "safety_target": 3.0,
            "transition_target": 300.0
        }
        
        optimization = self.naval_framework.optimize_vessel_design(
            VesselCategory.CREW_VESSEL, target_performance
        )
        
        # Verify optimization results
        self.assertIn("optimization_result", optimization)
        self.assertIn("predicted_performance", optimization)
        
        # Verify golden ratio enhancement
        enhancement = optimization["optimization_result"]["performance_enhancement"]
        self.assertGreater(enhancement["efficiency_boost"], 1.0)
        
    def test_comprehensive_design_report(self):
        """Test comprehensive design report generation"""
        report = self.naval_framework.generate_vessel_design_report(VesselCategory.CREW_VESSEL)
        
        # Verify report completeness
        self.assertIn("mode_performance_analyses", report)
        self.assertIn("optimization_analysis", report)
        self.assertIn("performance_summary", report)
        self.assertIn("design_recommendations", report)
        
        # Verify performance summary metrics
        summary = report["performance_summary"]
        self.assertIn("average_efficiency", summary)
        self.assertIn("meets_efficiency_target", summary)
        
def run_comprehensive_validation():
    """Run comprehensive validation of FTL hull design implementation"""
    print("FTL-Capable Hull Design Validation")
    print("=" * 50)
    
    # Initialize frameworks
    naval_framework = NavalArchitectureFramework()
    materials_framework = AdvancedMaterialsFramework()
    
    print("\n1. Advanced Materials Validation:")
    print("-" * 30)
    
    # Validate all materials for 48c operations
    all_materials_valid = True
    for material_name in materials_framework.materials_database.keys():
        validation = materials_framework.validate_48c_operations(material_name)
        safety_factor = validation.get("safety_factor", 0)
        is_valid = validation.get("validated_for_48c", False)
        
        status = "✅ VALID" if is_valid else "❌ FAILED"
        print(f"  {material_name}: {safety_factor:.1f}x safety factor - {status}")
        
        if not is_valid:
            all_materials_valid = False
            
    print(f"\nMaterials Validation: {'✅ ALL PASS' if all_materials_valid else '❌ FAILURES DETECTED'}")
    
    print("\n2. Vessel Performance Analysis:")
    print("-" * 30)
    
    # Analyze crew vessel performance in all modes
    modes_tested = 0
    modes_passed = 0
    
    for mode in [OperationalMode.PLANETARY_LANDING, OperationalMode.IMPULSE_CRUISE, 
                OperationalMode.WARP_BUBBLE]:
        modes_tested += 1
        
        analysis = naval_framework.analyze_vessel_performance(VesselCategory.CREW_VESSEL, mode)
        score = analysis.get("overall_performance_score", 0)
        efficiency = analysis.get("geometry_performance", {}).get("mode_efficiency", 0)
        
        if score >= 0.80 and efficiency >= 0.90:
            modes_passed += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            
        print(f"  {mode.value}: {score:.3f} score, {efficiency:.1%} efficiency - {status}")
        
    print(f"\nPerformance Analysis: {modes_passed}/{modes_tested} modes passing")
    
    print("\n3. Implementation Completeness:")
    print("-" * 30)
    
    # Check implementation completeness
    vessel_designs = len(naval_framework.vessel_designs)
    geometry_systems = len(naval_framework.geometry_systems)
    material_specs = len(materials_framework.materials_database)
    
    print(f"  Vessel Designs: {vessel_designs}/2 implemented")
    print(f"  Geometry Systems: {geometry_systems}/3 initialized")
    print(f"  Material Specifications: {material_specs}/3 validated")
    
    # Overall validation score
    materials_score = 1.0 if all_materials_valid else 0.5
    performance_score = modes_passed / modes_tested
    completeness_score = min(vessel_designs/2, geometry_systems/3, material_specs/3)
    
    overall_score = (materials_score + performance_score + completeness_score) / 3
    
    print(f"\n4. Overall Validation Results:")
    print("-" * 30)
    print(f"  Materials Validation: {materials_score:.1%}")
    print(f"  Performance Analysis: {performance_score:.1%}")
    print(f"  Implementation Completeness: {completeness_score:.1%}")
    print(f"  Overall Score: {overall_score:.1%}")
    
    if overall_score >= 0.90:
        print(f"\n🎉 VALIDATION SUCCESS: FTL Hull Design Implementation Complete!")
        print(f"   Ready for 48c operations with comprehensive material validation")
    elif overall_score >= 0.80:
        print(f"\n✅ VALIDATION PASSED: Implementation functional with minor improvements needed")
    else:
        print(f"\n❌ VALIDATION FAILED: Critical issues require resolution")
        
    return overall_score

if __name__ == "__main__":
    # Run unit tests
    print("Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "=" * 60)
    
    # Run comprehensive validation
    validation_score = run_comprehensive_validation()
    
    print(f"\nFinal Validation Score: {validation_score:.1%}")
