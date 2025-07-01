#!/usr/bin/env python3
"""
Simple test to verify framework imports
"""
import sys
from pathlib import Path

# Add src to path
framework_root = Path(__file__).parent
sys.path.insert(0, str(framework_root / "src"))

def test_imports():
    print("Testing framework imports...")
    
    try:
        print("1. Testing Enhanced Stochastic Field Evolution...")
        from digital_twin.enhanced_stochastic_field_evolution import (
            EnhancedStochasticFieldEvolution, FieldEvolutionConfig
        )
        print("   ✓ Field evolution module imported successfully")
        
        print("2. Testing Multi-Physics Coupling...")
        from multi_physics.enhanced_multi_physics_coupling import (
            EnhancedMultiPhysicsCoupling, MultiPhysicsConfig
        )
        print("   ✓ Multi-physics module imported successfully")
        
        print("3. Testing Metamaterial Amplification...")
        from metamaterial_fusion.enhanced_metamaterial_amplification import (
            EnhancedMetamaterialAmplification, MetamaterialConfig
        )
        print("   ✓ Metamaterial module imported successfully")
        
        print("4. Testing Einstein-Maxwell Coupling...")
        from multi_physics.einstein_maxwell_material_coupling import (
            EinsteinMaxwellMaterialCoupling, EinsteinMaxwellConfig
        )
        print("   ✓ Einstein-Maxwell module imported successfully")
        
        print("5. Testing Main Framework...")
        from enhanced_simulation_framework import (
            EnhancedSimulationFramework, FrameworkConfig
        )
        print("   ✓ Main framework imported successfully")
        
        print("\n🎉 ALL MODULES IMPORTED SUCCESSFULLY! 🎉")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)
