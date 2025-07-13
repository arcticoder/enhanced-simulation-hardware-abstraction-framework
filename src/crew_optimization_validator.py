#!/usr/bin/env python3
"""
Crew Optimization Validation Framework - Enhanced Simulation Hardware Abstraction Framework

Comprehensive validation and testing framework for crew complement optimization.

Author: Enhanced Simulation Hardware Abstraction Framework
Date: July 13, 2025
Version: 1.0.0 - Production Implementation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import logging
import time
from pathlib import Path

from crew_economic_optimizer import CrewEconomicOptimizer, MissionType, CrewConfiguration
from crew_role_optimizer import CrewRoleOptimizer
from mission_profile_integrator import MissionProfileIntegrator

logger = logging.getLogger(__name__)

@dataclass
class ValidationResults:
    """Validation test results."""
    test_name: str
    passed: bool
    score: float
    details: Dict[str, any]
    execution_time: float
    recommendations: List[str]

@dataclass
class BenchmarkResults:
    """Benchmark comparison results."""
    optimization_time: float
    solution_quality: float
    convergence_rate: float
    robustness_score: float
    scalability_metrics: Dict[str, float]

class CrewOptimizationValidator:
    """
    Comprehensive validation framework for crew optimization systems.
    """
    
    def __init__(self):
        """Initialize validation framework."""
        self.economic_optimizer = CrewEconomicOptimizer()
        self.role_optimizer = CrewRoleOptimizer(self.economic_optimizer)
        self.mission_integrator = MissionProfileIntegrator(
            self.economic_optimizer, self.role_optimizer
        )
        self.validation_history = []
        
    def run_comprehensive_validation(self) -> Dict[str, ValidationResults]:
        """Run comprehensive validation suite."""
        
        logger.info("Starting comprehensive crew optimization validation")
        
        validation_tests = {
            "economic_model_validation": self._validate_economic_model,
            "role_optimization_validation": self._validate_role_optimization,
            "constraint_satisfaction": self._validate_constraints,
            "scalability_testing": self._validate_scalability,
            "robustness_testing": self._validate_robustness,
            "integration_testing": self._validate_integration
        }
        
        results = {}
        total_start_time = time.time()
        
        for test_name, test_function in validation_tests.items():
            print(f"\n🧪 Running {test_name.replace('_', ' ').title()}...")
            
            start_time = time.time()
            try:
                result = test_function()
                execution_time = time.time() - start_time
                
                validation_result = ValidationResults(
                    test_name=test_name,
                    passed=result["passed"],
                    score=result["score"],
                    details=result["details"],
                    execution_time=execution_time,
                    recommendations=result.get("recommendations", [])
                )
                
                results[test_name] = validation_result
                
                status = "✅ PASSED" if result["passed"] else "❌ FAILED"
                print(f"   {status} - Score: {result['score']:.3f} - Time: {execution_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Validation test {test_name} failed: {e}")
                results[test_name] = ValidationResults(
                    test_name=test_name,
                    passed=False,
                    score=0.0,
                    details={"error": str(e)},
                    execution_time=time.time() - start_time,
                    recommendations=[f"Fix error: {str(e)}"]
                )
        
        total_time = time.time() - total_start_time
        
        # Generate summary
        passed_tests = sum(1 for r in results.values() if r.passed)
        total_tests = len(results)
        average_score = np.mean([r.score for r in results.values()])
        
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Average Score: {average_score:.3f}")
        print(f"   Total Time: {total_time:.2f}s")
        
        return results
    
    def _validate_economic_model(self) -> Dict[str, any]:
        """Validate economic optimization model."""
        
        test_results = {"passed": True, "score": 0.0, "details": {}}
        
        # Test 1: ROI calculation accuracy
        test_config = CrewConfiguration(
            total_crew=50, command=4, engineering=12, medical=5, science=8,
            maintenance=6, security=3, passengers=10, support=2,
            mission_type=MissionType.SCIENTIFIC_EXPLORATION
        )
        
        economic_result = self.economic_optimizer.evaluate_crew_configuration(test_config)
        
        # Validate ROI is reasonable (-50% to +200%)
        roi_valid = -50 <= economic_result.roi <= 200
        test_results["details"]["roi_range_valid"] = roi_valid
        
        # Validate cost components sum correctly
        total_calculated = sum(economic_result.cost_breakdown.values())
        cost_sum_valid = abs(total_calculated - economic_result.total_cost) < 1000
        test_results["details"]["cost_sum_valid"] = cost_sum_valid
        
        # Test 2: Optimization convergence
        try:
            optimal_config, optimal_results = self.economic_optimizer.optimize_crew_size(
                MissionType.TOURISM, mission_duration=90
            )
            optimization_valid = optimal_results.roi > -10  # Should achieve reasonable ROI
            test_results["details"]["optimization_convergence"] = optimization_valid
        except Exception as e:
            optimization_valid = False
            test_results["details"]["optimization_error"] = str(e)
        
        # Calculate score
        checks = [roi_valid, cost_sum_valid, optimization_valid]
        test_results["score"] = sum(checks) / len(checks)
        test_results["passed"] = test_results["score"] >= 0.8
        
        return test_results
    
    def _validate_role_optimization(self) -> Dict[str, any]:
        """Validate role optimization functionality."""
        
        test_results = {"passed": True, "score": 0.0, "details": {}}
        
        test_config = CrewConfiguration(
            total_crew=30, command=3, engineering=8, medical=3, science=6,
            maintenance=4, security=2, passengers=2, support=2,
            mission_type=MissionType.SCIENTIFIC_EXPLORATION
        )
        
        # Generate role requirements
        role_requirements = self.role_optimizer.generate_role_requirements(
            test_config.mission_type, test_config.total_crew
        )
        
        # Test role optimization
        role_results = self.role_optimizer.optimize_role_assignments(
            test_config, role_requirements
        )
        
        # Validate efficiency scores
        efficiency_valid = role_results.efficiency_metrics["overall_efficiency"] > 0.5
        test_results["details"]["efficiency_valid"] = efficiency_valid
        
        # Validate redundancy coverage
        avg_redundancy = np.mean(list(role_results.redundancy_analysis.values()))
        redundancy_valid = avg_redundancy > 0.4
        test_results["details"]["redundancy_valid"] = redundancy_valid
        
        # Validate risk assessment
        overall_risk = role_results.risk_assessment["overall_risk"]
        risk_valid = 0 <= overall_risk <= 1
        test_results["details"]["risk_valid"] = risk_valid
        
        # Calculate score
        checks = [efficiency_valid, redundancy_valid, risk_valid]
        test_results["score"] = sum(checks) / len(checks)
        test_results["passed"] = test_results["score"] >= 0.8
        
        return test_results
    
    def _validate_constraints(self) -> Dict[str, any]:
        """Validate constraint satisfaction."""
        
        test_results = {"passed": True, "score": 0.0, "details": {}}
        
        # Test crew size constraints
        for mission_type in [MissionType.TOURISM, MissionType.SCIENTIFIC_EXPLORATION]:
            try:
                optimal_config, _ = self.economic_optimizer.optimize_crew_size(mission_type)
                
                # Check total crew constraint
                crew_size_valid = 10 <= optimal_config.total_crew <= 100
                test_results["details"][f"{mission_type.value}_crew_size"] = crew_size_valid
                
                # Check role distribution
                total_roles = (optimal_config.command + optimal_config.engineering +
                             optimal_config.medical + optimal_config.science +
                             optimal_config.maintenance + optimal_config.security +
                             optimal_config.passengers + optimal_config.support)
                
                role_sum_valid = total_roles == optimal_config.total_crew
                test_results["details"][f"{mission_type.value}_role_sum"] = role_sum_valid
                
            except Exception as e:
                test_results["details"][f"{mission_type.value}_error"] = str(e)
        
        # Calculate score based on constraint satisfaction
        valid_checks = [v for k, v in test_results["details"].items() 
                       if isinstance(v, bool) and v]
        total_checks = [v for k, v in test_results["details"].items() 
                       if isinstance(v, bool)]
        
        if total_checks:
            test_results["score"] = len(valid_checks) / len(total_checks)
        else:
            test_results["score"] = 0.0
        
        test_results["passed"] = test_results["score"] >= 0.9
        
        return test_results
    
    def _validate_scalability(self) -> Dict[str, any]:
        """Validate system scalability."""
        
        test_results = {"passed": True, "score": 0.0, "details": {}}
        
        crew_sizes = [20, 50, 80, 100]
        execution_times = []
        
        for crew_size in crew_sizes:
            start_time = time.time()
            
            test_config = CrewConfiguration(
                total_crew=crew_size,
                command=max(2, int(crew_size * 0.08)),
                engineering=int(crew_size * 0.25),
                medical=max(2, int(crew_size * 0.10)),
                science=int(crew_size * 0.15),
                maintenance=int(crew_size * 0.12),
                security=max(1, int(crew_size * 0.05)),
                passengers=int(crew_size * 0.20),
                support=int(crew_size * 0.05),
                mission_type=MissionType.TOURISM
            )
            
            # Test economic evaluation
            economic_result = self.economic_optimizer.evaluate_crew_configuration(test_config)
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            test_results["details"][f"crew_{crew_size}_time"] = execution_time
        
        # Check scalability (execution time should scale reasonably)
        max_time = max(execution_times)
        scalability_valid = max_time < 5.0  # Should complete in under 5 seconds
        
        test_results["details"]["max_execution_time"] = max_time
        test_results["details"]["scalability_valid"] = scalability_valid
        
        test_results["score"] = 1.0 if scalability_valid else 0.5
        test_results["passed"] = scalability_valid
        
        return test_results
    
    def _validate_robustness(self) -> Dict[str, any]:
        """Validate system robustness to parameter variations."""
        
        test_results = {"passed": True, "score": 0.0, "details": {}}
        
        base_config = CrewConfiguration(
            total_crew=40, command=3, engineering=10, medical=4, science=6,
            maintenance=5, security=2, passengers=8, support=2,
            mission_type=MissionType.TOURISM
        )
        
        # Test with parameter variations
        variations = [0.8, 0.9, 1.0, 1.1, 1.2]  # ±20% variation
        baseline_roi = None
        roi_stability = []
        
        for variation in variations:
            # Modify economic parameters
            modified_params = self.economic_optimizer.economic_params
            original_passenger_revenue = modified_params.passenger_revenue_per_day
            modified_params.passenger_revenue_per_day = original_passenger_revenue * variation
            
            try:
                result = self.economic_optimizer.evaluate_crew_configuration(base_config)
                
                if baseline_roi is None:
                    baseline_roi = result.roi
                
                roi_deviation = abs(result.roi - baseline_roi) / abs(baseline_roi) if baseline_roi != 0 else 0
                roi_stability.append(roi_deviation)
                
            except Exception as e:
                test_results["details"]["robustness_error"] = str(e)
            
            # Restore original parameter
            modified_params.passenger_revenue_per_day = original_passenger_revenue
        
        # Calculate robustness score
        if roi_stability:
            avg_deviation = np.mean(roi_stability)
            robustness_score = max(0, 1 - avg_deviation)  # Lower deviation = higher score
            test_results["details"]["average_roi_deviation"] = avg_deviation
            test_results["details"]["robustness_score"] = robustness_score
        else:
            robustness_score = 0
        
        test_results["score"] = robustness_score
        test_results["passed"] = robustness_score > 0.7
        
        return test_results
    
    def _validate_integration(self) -> Dict[str, any]:
        """Validate integration between all components."""
        
        test_results = {"passed": True, "score": 0.0, "details": {}}
        
        # Test full integration pipeline
        try:
            # Get mission template
            sci_profile = self.mission_integrator.mission_templates[MissionType.SCIENTIFIC_EXPLORATION]
            
            # Run full optimization
            integration_results = self.mission_integrator.optimize_for_mission_profile(sci_profile)
            
            # Validate results structure
            required_keys = ["optimal_configuration", "economic_results", "role_results", 
                           "phase_analysis", "adaptation_strategies"]
            
            structure_valid = all(key in integration_results for key in required_keys)
            test_results["details"]["structure_valid"] = structure_valid
            
            # Validate configuration consistency
            config = integration_results["optimal_configuration"]
            config_valid = (config.total_crew <= 100 and 
                          config.mission_type == MissionType.SCIENTIFIC_EXPLORATION)
            test_results["details"]["config_valid"] = config_valid
            
            # Validate economic results
            economic = integration_results["economic_results"]
            economic_valid = hasattr(economic, "roi") and hasattr(economic, "total_cost")
            test_results["details"]["economic_valid"] = economic_valid
            
            # Calculate integration score
            checks = [structure_valid, config_valid, economic_valid]
            test_results["score"] = sum(checks) / len(checks)
            test_results["passed"] = test_results["score"] >= 0.8
            
        except Exception as e:
            test_results["details"]["integration_error"] = str(e)
            test_results["score"] = 0.0
            test_results["passed"] = False
        
        return test_results
    
    def generate_validation_report(self, results: Dict[str, ValidationResults], 
                                 save_path: Optional[str] = None) -> Dict[str, any]:
        """Generate comprehensive validation report."""
        
        report = {
            "validation_summary": {
                "total_tests": len(results),
                "passed_tests": sum(1 for r in results.values() if r.passed),
                "average_score": np.mean([r.score for r in results.values()]),
                "total_execution_time": sum(r.execution_time for r in results.values()),
                "validation_date": pd.Timestamp.now().isoformat()
            },
            "test_results": {},
            "recommendations": [],
            "overall_assessment": ""
        }
        
        # Compile test results
        for test_name, result in results.items():
            report["test_results"][test_name] = {
                "passed": result.passed,
                "score": result.score,
                "execution_time": result.execution_time,
                "details": result.details,
                "recommendations": result.recommendations
            }
            
            # Collect recommendations
            report["recommendations"].extend(result.recommendations)
        
        # Overall assessment
        overall_score = report["validation_summary"]["average_score"]
        if overall_score >= 0.9:
            report["overall_assessment"] = "EXCELLENT - System ready for production"
        elif overall_score >= 0.8:
            report["overall_assessment"] = "GOOD - Minor improvements recommended"
        elif overall_score >= 0.7:
            report["overall_assessment"] = "ACCEPTABLE - Some issues need attention"
        else:
            report["overall_assessment"] = "NEEDS IMPROVEMENT - Major issues detected"
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Validation report saved to {save_path}")
        
        return report


def demonstrate_validation():
    """Demonstration of the validation framework."""
    print("\n" + "="*60)
    print("CREW OPTIMIZATION VALIDATION - DEMONSTRATION")
    print("="*60)
    
    # Initialize validator
    validator = CrewOptimizationValidator()
    
    # Run comprehensive validation
    print("\n🧪 Running Comprehensive Validation Suite...")
    validation_results = validator.run_comprehensive_validation()
    
    # Generate report
    report = validator.generate_validation_report(validation_results)
    
    print(f"\n📋 VALIDATION REPORT:")
    print(f"   Overall Assessment: {report['overall_assessment']}")
    print(f"   Tests Passed: {report['validation_summary']['passed_tests']}/{report['validation_summary']['total_tests']}")
    print(f"   Average Score: {report['validation_summary']['average_score']:.3f}")
    print(f"   Total Time: {report['validation_summary']['total_execution_time']:.2f}s")
    
    if report["recommendations"]:
        print(f"\n💡 TOP RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"][:3], 1):
            print(f"   {i}. {rec}")
    
    print(f"\n🎉 VALIDATION FRAMEWORK DEMONSTRATION COMPLETE")
    print("="*60)
    
    return validation_results, report


if __name__ == "__main__":
    results, report = demonstrate_validation()
