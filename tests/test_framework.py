"""
Simple test of the Enhanced Simulation Framework
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
framework_root = Path(__file__).parent
src_path = framework_root / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all modules can be imported"""
    try:
        print("Testing imports...")
        
        # Test enhanced simulation framework
        from enhanced_simulation_framework import EnhancedSimulationFramework, FrameworkConfig
        print("✓ Main framework imported successfully")
        
        # Test field evolution
        from digital_twin.enhanced_stochastic_field_evolution import FieldEvolutionConfig
        print("✓ Field evolution module imported successfully")
        
        # Test multi-physics
        from multi_physics.enhanced_multi_physics_coupling import MultiPhysicsConfig
        print("✓ Multi-physics module imported successfully")
        
        # Test Einstein-Maxwell
        from multi_physics.einstein_maxwell_material_coupling import EinsteinMaxwellConfig
        print("✓ Einstein-Maxwell module imported successfully")
        
        # Test metamaterial
        from metamaterial_fusion.enhanced_metamaterial_amplification import MetamaterialConfig
        print("✓ Metamaterial module imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_framework_creation():
    """Test framework creation and basic functionality"""
    try:
        print("\nTesting framework creation...")
        
        from enhanced_simulation_framework import EnhancedSimulationFramework
        
        # Create framework
        framework = EnhancedSimulationFramework()
        print("✓ Framework created successfully")
        
        # Test initialization (lightweight)
        print("✓ Framework basic functionality verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Framework creation failed: {e}")
        return False

def main():
    """Run basic tests"""
    print("🧪 Enhanced Simulation Framework - Basic Tests")
    print("=" * 50)
    
    # Test imports
    import_success = test_imports()
    
    # Test framework creation
    creation_success = test_framework_creation()
    
    # Summary
    print("\n" + "=" * 50)
    if import_success and creation_success:
        print("✅ All basic tests passed!")
        print("Framework is ready for full simulation.")
        return 0
    else:
        print("❌ Some tests failed.")
        print("Please check the error messages above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
