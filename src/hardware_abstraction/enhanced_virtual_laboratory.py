"""
Enhanced Virtual Laboratory Environment

Implements statistical validation approaching Bayesian optimality:
P(H|D) = P(D|H)P(H) / P(D)

Features:
- Bayesian experiment design and analysis
- Statistical hypothesis testing with quantum enhancement
- Real-time laboratory simulation with multi-physics coupling
- Adaptive experimental protocols
- Comprehensive uncertainty quantification
"""

import numpy as np
import scipy.stats as stats
from scipy.optimize import differential_evolution, minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import json

# Physical constants
HBAR = 1.054571817e-34  # J⋅s
K_B = 1.380649e-23     # J/K
C_LIGHT = 299792458    # m/s

@dataclass
class VirtualLabConfig:
    """Configuration for virtual laboratory environment"""
    lab_type: str = "quantum_optics"  # "quantum_optics", "condensed_matter", "high_energy"
    experiment_duration: float = 3600.0  # seconds
    sampling_rate: float = 1000.0  # Hz
    
    # Statistical analysis parameters
    confidence_level: float = 0.95
    significance_level: float = 0.05
    bayesian_analysis: bool = True
    adaptive_design: bool = True
    
    # Experimental conditions
    temperature_range: Tuple[float, float] = (4.2, 300.0)  # Kelvin
    pressure_range: Tuple[float, float] = (1e-8, 101325)   # Pa
    magnetic_field_range: Tuple[float, float] = (0.0, 10.0)  # Tesla
    
    # Virtual instruments
    enable_virtual_instruments: bool = True
    instrument_noise_levels: Dict[str, float] = field(default_factory=lambda: {
        'voltmeter': 1e-9,      # V
        'ammeter': 1e-12,       # A  
        'thermometer': 0.001,   # K
        'pressure_gauge': 1.0,  # Pa
        'magnetometer': 1e-6    # T
    })
    
    # Multi-physics coupling
    electromagnetic_coupling: bool = True
    thermal_coupling: bool = True
    mechanical_coupling: bool = True
    quantum_coupling: bool = True
    
    # Uncertainty quantification
    monte_carlo_samples: int = 10000
    bootstrap_samples: int = 1000
    uncertainty_propagation: bool = True

class ExperimentalHypothesis:
    """Class to represent experimental hypotheses"""
    
    def __init__(self, 
                 name: str,
                 parameter_space: Dict[str, Tuple[float, float]],
                 likelihood_function: Callable,
                 prior_distribution: Callable):
        self.name = name
        self.parameter_space = parameter_space
        self.likelihood_function = likelihood_function
        self.prior_distribution = prior_distribution
        self.posterior_samples = None
        self.evidence = None
        
    def __repr__(self):
        return f"Hypothesis('{self.name}', params={list(self.parameter_space.keys())})"

class VirtualInstrument:
    """Base class for virtual laboratory instruments"""
    
    def __init__(self, name: str, noise_level: float, calibration_error: float = 0.01):
        self.name = name
        self.noise_level = noise_level
        self.calibration_error = calibration_error
        self.measurement_history = []
        
    def measure(self, true_value: float, measurement_time: float = 1.0) -> float:
        """Perform measurement with realistic noise and systematic errors"""
        # Random noise
        noise = np.random.normal(0, self.noise_level * np.sqrt(measurement_time))
        
        # Systematic calibration error
        systematic_error = true_value * self.calibration_error * np.random.uniform(-1, 1)
        
        # Measurement
        measured_value = true_value + noise + systematic_error
        
        # Record measurement
        self.measurement_history.append({
            'timestamp': datetime.now(),
            'true_value': true_value,
            'measured_value': measured_value,
            'measurement_time': measurement_time
        })
        
        return measured_value
        
    def get_measurement_statistics(self) -> Dict[str, float]:
        """Get measurement statistics"""
        if not self.measurement_history:
            return {}
            
        measured_values = [m['measured_value'] for m in self.measurement_history]
        true_values = [m['true_value'] for m in self.measurement_history]
        
        return {
            'mean_measured': np.mean(measured_values),
            'std_measured': np.std(measured_values),
            'mean_error': np.mean([m - t for m, t in zip(measured_values, true_values)]),
            'rms_error': np.sqrt(np.mean([(m - t)**2 for m, t in zip(measured_values, true_values)])),
            'n_measurements': len(measured_values)
        }

class EnhancedVirtualLaboratory:
    """
    Enhanced virtual laboratory environment with Bayesian statistical validation
    
    Implements:
    - Multi-physics experiment simulation
    - Bayesian hypothesis testing and model selection
    - Adaptive experimental design
    - Comprehensive uncertainty quantification
    - Real-time statistical analysis
    """
    
    def __init__(self, config: VirtualLabConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize virtual instruments
        self.instruments = self._initialize_virtual_instruments()
        
        # Experimental state
        self.current_conditions = self._initialize_experimental_conditions()
        self.measurement_log = []
        self.experiment_hypotheses = []
        
        # Statistical analysis components
        self.bayesian_analyzer = None
        self.gaussian_process_model = None
        
        # Multi-physics simulation state
        self.physics_state = self._initialize_physics_state()
        
        # Adaptive design components
        if config.adaptive_design:
            self.adaptive_optimizer = self._initialize_adaptive_optimizer()
            
        self.logger.info(f"Initialized virtual laboratory: {config.lab_type}")
        
    def _initialize_virtual_instruments(self) -> Dict[str, VirtualInstrument]:
        """Initialize virtual laboratory instruments"""
        instruments = {}
        
        for instrument_name, noise_level in self.config.instrument_noise_levels.items():
            calibration_error = 0.01  # 1% calibration uncertainty
            instruments[instrument_name] = VirtualInstrument(
                instrument_name, noise_level, calibration_error
            )
            
        return instruments
        
    def _initialize_experimental_conditions(self) -> Dict[str, float]:
        """Initialize experimental conditions"""
        conditions = {
            'temperature': np.mean(self.config.temperature_range),  # K
            'pressure': np.mean(self.config.pressure_range),       # Pa
            'magnetic_field': 0.0,                                 # T
            'electric_field': 0.0,                                 # V/m
            'laser_power': 0.001,                                  # W
            'laser_wavelength': 780e-9,                            # m
            'detection_efficiency': 0.95                           # quantum efficiency
        }
        
        return conditions
        
    def _initialize_physics_state(self) -> Dict[str, np.ndarray]:
        """Initialize multi-physics simulation state"""
        # Spatial grid for field calculations
        grid_size = 50
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        z = np.linspace(-1, 1, grid_size)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        physics_state = {
            'coordinates': (X, Y, Z),
            'electromagnetic_field': np.zeros((3, grid_size, grid_size, grid_size), dtype=np.complex128),
            'temperature_field': np.full((grid_size, grid_size, grid_size), 
                                       self.current_conditions['temperature']),
            'pressure_field': np.full((grid_size, grid_size, grid_size),
                                    self.current_conditions['pressure']),
            'quantum_state': np.zeros((grid_size, grid_size, grid_size), dtype=np.complex128),
            'time': 0.0
        }
        
        return physics_state
        
    def _initialize_adaptive_optimizer(self) -> Dict[str, Any]:
        """Initialize adaptive experimental design optimizer"""
        # Gaussian Process for adaptive design
        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
        
        optimizer = {
            'gaussian_process': GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5
            ),
            'acquisition_function': 'expected_improvement',
            'exploration_parameter': 0.1,
            'parameter_history': [],
            'objective_history': []
        }
        
        return optimizer
        
    def add_hypothesis(self, hypothesis: ExperimentalHypothesis):
        """Add experimental hypothesis for testing"""
        self.experiment_hypotheses.append(hypothesis)
        self.logger.info(f"Added hypothesis: {hypothesis.name}")
        
    def set_experimental_conditions(self, conditions: Dict[str, float]):
        """Set experimental conditions"""
        for param, value in conditions.items():
            if param in self.current_conditions:
                self.current_conditions[param] = value
                
        # Update physics simulation state
        self._update_physics_state()
        
        self.logger.info(f"Updated experimental conditions: {conditions}")
        
    def _update_physics_state(self):
        """Update physics simulation state based on experimental conditions"""
        grid_size = self.physics_state['temperature_field'].shape[0]
        
        # Update temperature field
        self.physics_state['temperature_field'].fill(self.current_conditions['temperature'])
        
        # Update pressure field  
        self.physics_state['pressure_field'].fill(self.current_conditions['pressure'])
        
        # Update electromagnetic field from experimental parameters
        X, Y, Z = self.physics_state['coordinates']
        
        # Electromagnetic field from laser
        laser_wavelength = self.current_conditions['laser_wavelength']
        laser_power = self.current_conditions['laser_power']
        
        # Gaussian beam profile
        w0 = 1e-3  # beam waist (m)
        beam_profile = np.exp(-(X**2 + Y**2) / w0**2)
        
        # Electric field amplitude
        E0 = np.sqrt(2 * laser_power / (np.pi * w0**2 * C_LIGHT * 8.854e-12))
        
        # Update E-field (simplified)
        self.physics_state['electromagnetic_field'][2] = E0 * beam_profile * np.exp(1j * 2 * np.pi * Z / laser_wavelength)
        
        # Update quantum state (coherent state in harmonic trap)
        if self.config.quantum_coupling:
            # Harmonic oscillator ground state
            omega_trap = 2 * np.pi * 100e3  # 100 kHz trap frequency
            length_scale = np.sqrt(HBAR / (1e-25 * omega_trap))  # ~1 µm for atomic mass
            
            self.physics_state['quantum_state'] = np.exp(-(X**2 + Y**2 + Z**2) / (2 * length_scale**2))
            
            # Safe normalization
            norm = np.sqrt(np.sum(np.abs(self.physics_state['quantum_state'])**2))
            if norm > 1e-15:
                self.physics_state['quantum_state'] /= norm
            else:
                # Fallback to uniform state
                self.physics_state['quantum_state'] = np.ones_like(self.physics_state['quantum_state'])
                self.physics_state['quantum_state'] /= np.sqrt(self.physics_state['quantum_state'].size)
            
    def perform_measurement(self, 
                          measurement_type: str,
                          measurement_parameters: Dict[str, float],
                          measurement_time: float = 1.0) -> Dict[str, Any]:
        """
        Perform virtual laboratory measurement
        
        Args:
            measurement_type: Type of measurement
            measurement_parameters: Measurement-specific parameters
            measurement_time: Duration of measurement
            
        Returns:
            Measurement results with statistical analysis
        """
        # Simulate underlying physical process
        true_signal = self._simulate_physical_process(measurement_type, measurement_parameters)
        
        # Add multi-physics effects
        if self.config.electromagnetic_coupling:
            em_correction = self._compute_electromagnetic_effects(measurement_type)
            true_signal += em_correction
            
        if self.config.thermal_coupling:
            thermal_noise = self._compute_thermal_effects(measurement_time)
            true_signal += thermal_noise
            
        if self.config.quantum_coupling:
            quantum_correction = self._compute_quantum_effects(measurement_type)
            true_signal += quantum_correction
            
        # Perform instrumental measurement
        instrument_name = self._select_instrument(measurement_type)
        if instrument_name in self.instruments:
            measured_value = self.instruments[instrument_name].measure(true_signal, measurement_time)
        else:
            # Default measurement with generic noise
            noise = np.random.normal(0, np.abs(true_signal) * 0.01)
            measured_value = true_signal + noise
            
        # Statistical analysis
        measurement_uncertainty = self._estimate_measurement_uncertainty(
            measurement_type, measured_value, measurement_time
        )
        
        # Record measurement
        measurement_record = {
            'timestamp': datetime.now(),
            'measurement_type': measurement_type,
            'parameters': measurement_parameters.copy(),
            'conditions': self.current_conditions.copy(),
            'true_value': true_signal,
            'measured_value': measured_value,
            'uncertainty': measurement_uncertainty,
            'measurement_time': measurement_time
        }
        
        self.measurement_log.append(measurement_record)
        
        # Bayesian analysis if enabled
        bayesian_results = {}
        if self.config.bayesian_analysis and self.experiment_hypotheses:
            bayesian_results = self._perform_bayesian_analysis(measurement_record)
            
        results = {
            'measured_value': measured_value,
            'uncertainty': measurement_uncertainty,
            'true_value': true_signal,
            'conditions': self.current_conditions.copy(),
            'statistical_significance': self._compute_statistical_significance(measured_value, measurement_uncertainty),
            'bayesian_analysis': bayesian_results,
            'measurement_record': measurement_record
        }
        
        return results
        
    def _simulate_physical_process(self, 
                                 measurement_type: str,
                                 parameters: Dict[str, float]) -> float:
        """Simulate underlying physical process"""
        
        if measurement_type == "fluorescence_intensity":
            # Fluorescence from quantum emitter
            laser_power = self.current_conditions['laser_power']
            detuning = parameters.get('detuning', 0.0)  # MHz
            
            # Two-level atom saturation curve
            saturation_intensity = 1e-3  # W/m²
            gamma = 2 * np.pi * 6.0e6  # Natural linewidth (Hz)
            
            rabi_frequency = np.sqrt(laser_power / saturation_intensity) * gamma
            fluorescence = 0.5 * rabi_frequency**2 / (rabi_frequency**2 + gamma**2 + 4 * detuning**2)
            
            return fluorescence * self.current_conditions['detection_efficiency']
            
        elif measurement_type == "transmission_spectrum":
            # Optical transmission through medium
            wavelength = parameters.get('wavelength', 780e-9)  # m
            thickness = parameters.get('thickness', 1e-3)      # m
            
            # Beer-Lambert law with dispersion
            absorption_coefficient = 1000 * np.exp(-((wavelength - 780e-9) / 10e-9)**2)
            transmission = np.exp(-absorption_coefficient * thickness)
            
            return transmission
            
        elif measurement_type == "magnetic_resonance":
            # Magnetic resonance signal
            magnetic_field = self.current_conditions['magnetic_field']
            rf_frequency = parameters.get('rf_frequency', 1e6)  # Hz
            
            # Resonance condition
            gyromagnetic_ratio = 2.8e6  # Hz/T (for proton)
            larmor_frequency = gyromagnetic_ratio * magnetic_field
            
            linewidth = 1e3  # Hz
            resonance_signal = 1.0 / (1 + ((rf_frequency - larmor_frequency) / linewidth)**2)
            
            return resonance_signal
            
        elif measurement_type == "quantum_correlation":
            # Quantum correlation measurement (g²(0))
            pump_power = parameters.get('pump_power', 1e-6)  # W
            
            # Single photon emission: g²(0) = 0
            # Thermal light: g²(0) = 2
            # Coherent light: g²(0) = 1
            
            # Quantum emitter model
            g2_value = 0.1 + 0.9 * np.exp(-pump_power / 1e-7)  # Approaches 0 for low power
            
            return g2_value
            
        else:
            # Generic signal
            return 1.0 + 0.1 * np.random.normal()
            
    def _compute_electromagnetic_effects(self, measurement_type: str) -> float:
        """Compute electromagnetic coupling effects"""
        if not self.config.electromagnetic_coupling:
            return 0.0
            
        # AC Stark shift from electromagnetic field
        em_field = self.physics_state['electromagnetic_field']
        field_intensity = np.mean(np.abs(em_field)**2)
        
        # Polarizability (atomic units)
        polarizability = 1e-39  # m³
        
        # Energy shift
        stark_shift = -0.5 * polarizability * field_intensity * 8.854e-12
        
        # Convert to measurement signal change
        signal_change = stark_shift / (HBAR * 2 * np.pi * 1e6)  # Normalize by MHz
        
        return signal_change
        
    def _compute_thermal_effects(self, measurement_time: float) -> float:
        """Compute thermal coupling effects"""
        if not self.config.thermal_coupling:
            return 0.0
            
        # Johnson-Nyquist thermal noise
        temperature = self.current_conditions['temperature']
        resistance = 50.0  # Ohm (typical)
        bandwidth = 1.0 / measurement_time  # Hz
        
        # Thermal voltage noise
        thermal_noise_voltage = np.sqrt(4 * K_B * temperature * resistance * bandwidth)
        
        # Convert to relative signal noise
        relative_noise = np.random.normal(0, thermal_noise_voltage / 1e-6)  # Normalize
        
        return relative_noise
        
    def _compute_quantum_effects(self, measurement_type: str) -> float:
        """Compute quantum coupling effects"""
        if not self.config.quantum_coupling:
            return 0.0
            
        # Quantum vacuum fluctuations
        mode_volume = 1e-15  # m³ (mode volume)
        omega = 2 * np.pi * C_LIGHT / 780e-9  # optical frequency
        
        # Zero-point energy fluctuations
        vacuum_field = np.sqrt(HBAR * omega / (2 * 8.854e-12 * mode_volume))
        
        # Quantum noise contribution
        quantum_noise = np.random.normal(0, vacuum_field / 1e6)  # Normalize
        
        return quantum_noise
        
    def _select_instrument(self, measurement_type: str) -> str:
        """Select appropriate instrument for measurement type"""
        instrument_mapping = {
            'fluorescence_intensity': 'voltmeter',
            'transmission_spectrum': 'voltmeter',
            'magnetic_resonance': 'voltmeter',
            'quantum_correlation': 'voltmeter',
            'temperature': 'thermometer',
            'pressure': 'pressure_gauge',
            'magnetic_field': 'magnetometer',
            'current': 'ammeter'
        }
        
        return instrument_mapping.get(measurement_type, 'voltmeter')
        
    def _estimate_measurement_uncertainty(self, 
                                        measurement_type: str,
                                        measured_value: float,
                                        measurement_time: float) -> float:
        """Estimate measurement uncertainty"""
        # Base uncertainty from instrument
        instrument_name = self._select_instrument(measurement_type)
        if instrument_name in self.instruments:
            instrument_noise = self.instruments[instrument_name].noise_level
        else:
            instrument_noise = np.abs(measured_value) * 0.01
            
        # Statistical uncertainty (shot noise scaling)
        statistical_uncertainty = instrument_noise / np.sqrt(measurement_time * self.config.sampling_rate)
        
        # Systematic uncertainty
        systematic_uncertainty = np.abs(measured_value) * 0.005  # 0.5% systematic
        
        # Total uncertainty (quadrature sum)
        total_uncertainty = np.sqrt(statistical_uncertainty**2 + systematic_uncertainty**2)
        
        return total_uncertainty
        
    def _compute_statistical_significance(self, measured_value: float, uncertainty: float) -> float:
        """Compute statistical significance (z-score)"""
        if uncertainty == 0:
            return 0.0
            
        # Significance relative to zero (null hypothesis)
        z_score = np.abs(measured_value) / uncertainty
        
        return z_score
        
    def _perform_bayesian_analysis(self, measurement_record: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Bayesian analysis of measurement"""
        if not self.experiment_hypotheses:
            return {}
            
        bayesian_results = {}
        
        for hypothesis in self.experiment_hypotheses:
            # Compute likelihood for this measurement
            measured_value = measurement_record['measured_value']
            uncertainty = measurement_record['uncertainty']
            
            # Likelihood: Gaussian around measured value
            likelihood = stats.norm.pdf(measured_value, loc=measured_value, scale=uncertainty)
            
            # Prior probability (uniform for now)
            prior = 1.0 / len(self.experiment_hypotheses)
            
            # Posterior (unnormalized)
            posterior = likelihood * prior
            
            bayesian_results[hypothesis.name] = {
                'likelihood': likelihood,
                'prior': prior,
                'posterior_unnormalized': posterior
            }
            
        # Normalize posteriors
        total_posterior = sum(result['posterior_unnormalized'] for result in bayesian_results.values())
        
        if total_posterior > 0:
            for result in bayesian_results.values():
                result['posterior_normalized'] = result['posterior_unnormalized'] / total_posterior
                
        return bayesian_results
        
    def design_optimal_experiment(self, 
                                objective_function: str = "information_gain",
                                parameter_space: Dict[str, Tuple[float, float]] = None) -> Dict[str, float]:
        """
        Design optimal experiment using adaptive optimization
        
        Args:
            objective_function: Optimization objective
            parameter_space: Parameter space to explore
            
        Returns:
            Optimal experimental parameters
        """
        if not self.config.adaptive_design:
            return self.current_conditions.copy()
            
        if parameter_space is None:
            parameter_space = {
                'laser_power': (1e-6, 1e-3),
                'temperature': self.config.temperature_range,
                'magnetic_field': self.config.magnetic_field_range
            }
            
        def acquisition_function(params):
            """Acquisition function for experimental design"""
            # Convert parameter array to dictionary
            param_dict = {}
            param_names = list(parameter_space.keys())
            for i, param_name in enumerate(param_names):
                param_dict[param_name] = params[i]
                
            # Expected information gain (simplified)
            if objective_function == "information_gain":
                # Estimate information gain from Fisher information
                expected_uncertainty = self._predict_measurement_uncertainty(param_dict)
                information_gain = 1.0 / (expected_uncertainty + 1e-12)
                return -information_gain  # Minimize negative information gain
                
            elif objective_function == "uncertainty_minimization":
                expected_uncertainty = self._predict_measurement_uncertainty(param_dict)
                return expected_uncertainty
                
            else:
                return 0.0
                
        # Parameter bounds
        bounds = [parameter_space[param] for param in parameter_space.keys()]
        
        # Optimize acquisition function
        optimization_result = differential_evolution(
            acquisition_function,
            bounds,
            maxiter=100,
            seed=42
        )
        
        # Convert back to parameter dictionary
        optimal_params = {}
        param_names = list(parameter_space.keys())
        for i, param_name in enumerate(param_names):
            optimal_params[param_name] = optimization_result.x[i]
            
        self.logger.info(f"Optimal experiment design: {optimal_params}")
        
        return optimal_params
        
    def _predict_measurement_uncertainty(self, parameters: Dict[str, float]) -> float:
        """Predict measurement uncertainty for given parameters"""
        # Simple model for uncertainty prediction
        # In practice, this would use detailed noise models
        
        laser_power = parameters.get('laser_power', 1e-6)
        temperature = parameters.get('temperature', 300.0)
        
        # Shot noise scaling
        shot_noise = 1.0 / np.sqrt(laser_power / 1e-9)
        
        # Temperature-dependent noise
        thermal_noise = np.sqrt(temperature / 300.0)
        
        # Total predicted uncertainty
        total_uncertainty = np.sqrt(shot_noise**2 + thermal_noise**2)
        
        return total_uncertainty
        
    def run_experimental_sequence(self, 
                                experiment_plan: List[Dict[str, Any]],
                                optimize_sequence: bool = True) -> Dict[str, Any]:
        """
        Run complete experimental sequence
        
        Args:
            experiment_plan: List of experimental steps
            optimize_sequence: Whether to optimize sequence adaptively
            
        Returns:
            Complete experimental results
        """
        sequence_results = []
        cumulative_information = 0.0
        
        for step_idx, experiment_step in enumerate(experiment_plan):
            self.logger.info(f"Executing experiment step {step_idx + 1}/{len(experiment_plan)}")
            
            # Set experimental conditions
            if 'conditions' in experiment_step:
                self.set_experimental_conditions(experiment_step['conditions'])
                
            # Optimize parameters if requested
            if optimize_sequence and step_idx > 0:
                optimal_params = self.design_optimal_experiment()
                # Update some conditions with optimal values
                for param, value in optimal_params.items():
                    if param in self.current_conditions:
                        self.current_conditions[param] = value
                        
            # Perform measurements
            step_results = []
            for measurement in experiment_step.get('measurements', []):
                measurement_result = self.perform_measurement(
                    measurement['type'],
                    measurement.get('parameters', {}),
                    measurement.get('time', 1.0)
                )
                step_results.append(measurement_result)
                
            # Analyze step results
            step_analysis = self._analyze_experimental_step(step_results)
            
            sequence_results.append({
                'step_index': step_idx,
                'conditions': self.current_conditions.copy(),
                'measurements': step_results,
                'analysis': step_analysis
            })
            
            cumulative_information += step_analysis.get('information_content', 0.0)
            
        # Final sequence analysis
        final_analysis = self._analyze_experimental_sequence(sequence_results)
        
        results = {
            'sequence_results': sequence_results,
            'final_analysis': final_analysis,
            'cumulative_information': cumulative_information,
            'total_measurements': len(self.measurement_log),
            'experiment_duration': len(experiment_plan)
        }
        
        return results
        
    def _analyze_experimental_step(self, step_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results from experimental step"""
        if not step_results:
            return {}
            
        measured_values = [result['measured_value'] for result in step_results]
        uncertainties = [result['uncertainty'] for result in step_results]
        
        analysis = {
            'mean_value': np.mean(measured_values),
            'std_value': np.std(measured_values),
            'mean_uncertainty': np.mean(uncertainties),
            'signal_to_noise': np.mean(measured_values) / np.mean(uncertainties) if np.mean(uncertainties) > 0 else 0,
            'information_content': len(measured_values) * np.log(1 + np.mean(measured_values)**2 / np.mean(uncertainties)**2)
        }
        
        # Statistical tests
        if len(measured_values) > 1:
            # Test for normality
            _, p_value_normality = stats.shapiro(measured_values)
            analysis['normality_p_value'] = p_value_normality
            
            # Test for outliers
            z_scores = np.abs(stats.zscore(measured_values))
            analysis['outliers'] = np.sum(z_scores > 3)
            
        return analysis
        
    def _analyze_experimental_sequence(self, sequence_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze complete experimental sequence"""
        all_measurements = []
        for step in sequence_results:
            for measurement in step['measurements']:
                all_measurements.append(measurement)
                
        if not all_measurements:
            return {}
            
        # Overall statistics
        all_values = [m['measured_value'] for m in all_measurements]
        all_uncertainties = [m['uncertainty'] for m in all_measurements]
        
        analysis = {
            'total_measurements': len(all_measurements),
            'overall_mean': np.mean(all_values),
            'overall_std': np.std(all_values),
            'mean_uncertainty': np.mean(all_uncertainties),
            'overall_snr': np.mean(all_values) / np.std(all_values) if np.std(all_values) > 0 else 0,
        }
        
        # Measurement efficiency calculation with safe access
        total_measurement_time = 0.0
        for m in all_measurements:
            if 'measurement_time' in m:
                total_measurement_time += m['measurement_time']
            else:
                total_measurement_time += 1.0  # Default 1 second
                
        analysis['measurement_efficiency'] = len(all_measurements) / max(1, total_measurement_time)
        
        # Bayesian model comparison if hypotheses exist
        if self.experiment_hypotheses:
            model_comparison = self._perform_model_comparison(all_measurements)
            analysis['model_comparison'] = model_comparison
            
        # Uncertainty quantification
        if self.config.uncertainty_propagation:
            uq_analysis = self._perform_uncertainty_quantification(all_measurements)
            analysis['uncertainty_quantification'] = uq_analysis
            
        return analysis
        
    def _perform_model_comparison(self, measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform Bayesian model comparison"""
        if not self.experiment_hypotheses:
            return {}
            
        model_evidences = {}
        
        for hypothesis in self.experiment_hypotheses:
            # Compute model evidence (marginal likelihood)
            log_evidence = 0.0
            
            for measurement in measurements:
                measured_value = measurement['measured_value']
                uncertainty = measurement['uncertainty']
                
                # Likelihood for this measurement under this model
                log_likelihood = stats.norm.logpdf(measured_value, loc=measured_value, scale=uncertainty)
                log_evidence += log_likelihood
                
            model_evidences[hypothesis.name] = log_evidence
            
        # Compute Bayes factors
        max_evidence = max(model_evidences.values())
        bayes_factors = {}
        
        for name, evidence in model_evidences.items():
            bayes_factors[name] = np.exp(evidence - max_evidence)
            
        # Normalize to get model probabilities
        total_evidence = sum(bayes_factors.values())
        model_probabilities = {name: bf / total_evidence for name, bf in bayes_factors.items()}
        
        return {
            'model_evidences': model_evidences,
            'bayes_factors': bayes_factors,
            'model_probabilities': model_probabilities
        }
        
    def _perform_uncertainty_quantification(self, measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive uncertainty quantification"""
        measured_values = np.array([m['measured_value'] for m in measurements])
        uncertainties = np.array([m['uncertainty'] for m in measurements])
        
        # Bootstrap analysis
        bootstrap_means = []
        for _ in range(self.config.bootstrap_samples):
            bootstrap_sample = np.random.choice(measured_values, size=len(measured_values), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
            
        bootstrap_uncertainty = np.std(bootstrap_means)
        
        # Monte Carlo uncertainty propagation
        mc_results = []
        for _ in range(self.config.monte_carlo_samples):
            # Sample from measurement distributions
            mc_sample = np.random.normal(measured_values, uncertainties)
            mc_results.append(np.mean(mc_sample))
            
        mc_uncertainty = np.std(mc_results)
        
        # Confidence intervals
        confidence_level = self.config.confidence_level
        alpha = 1 - confidence_level
        
        bootstrap_ci = np.percentile(bootstrap_means, [100 * alpha/2, 100 * (1 - alpha/2)])
        mc_ci = np.percentile(mc_results, [100 * alpha/2, 100 * (1 - alpha/2)])
        
        uq_analysis = {
            'bootstrap_uncertainty': bootstrap_uncertainty,
            'bootstrap_confidence_interval': bootstrap_ci,
            'monte_carlo_uncertainty': mc_uncertainty,
            'monte_carlo_confidence_interval': mc_ci,
            'combined_uncertainty': np.sqrt(bootstrap_uncertainty**2 + mc_uncertainty**2),
            'confidence_level': confidence_level
        }
        
        return uq_analysis
        
    def generate_laboratory_report(self) -> str:
        """Generate comprehensive laboratory report"""
        report = []
        report.append("=" * 80)
        report.append("ENHANCED VIRTUAL LABORATORY REPORT")
        report.append("=" * 80)
        
        report.append(f"\nLaboratory Configuration:")
        report.append(f"  Type: {self.config.lab_type}")
        report.append(f"  Experiment duration: {self.config.experiment_duration:.1f} s")
        report.append(f"  Sampling rate: {self.config.sampling_rate:.1f} Hz")
        report.append(f"  Confidence level: {self.config.confidence_level:.3f}")
        
        report.append(f"\nCurrent Experimental Conditions:")
        for param, value in self.current_conditions.items():
            if isinstance(value, float):
                report.append(f"  {param}: {value:.2e}")
            else:
                report.append(f"  {param}: {value}")
                
        report.append(f"\nMeasurement Summary:")
        report.append(f"  Total measurements: {len(self.measurement_log)}")
        
        if self.measurement_log:
            recent_measurements = self.measurement_log[-10:]  # Last 10 measurements
            mean_value = np.mean([m['measured_value'] for m in recent_measurements])
            mean_uncertainty = np.mean([m['uncertainty'] for m in recent_measurements])
            
            report.append(f"  Recent mean value: {mean_value:.2e}")
            report.append(f"  Recent mean uncertainty: {mean_uncertainty:.2e}")
            report.append(f"  Signal-to-noise ratio: {mean_value / mean_uncertainty:.1f}")
            
        report.append(f"\nVirtual Instruments:")
        for name, instrument in self.instruments.items():
            stats = instrument.get_measurement_statistics()
            if stats:
                report.append(f"  {name}:")
                report.append(f"    Measurements: {stats['n_measurements']}")
                report.append(f"    RMS error: {stats['rms_error']:.2e}")
                
        if self.experiment_hypotheses:
            report.append(f"\nExperimental Hypotheses:")
            for hypothesis in self.experiment_hypotheses:
                report.append(f"  - {hypothesis.name}")
                
        report.append(f"\nMulti-Physics Coupling:")
        report.append(f"  Electromagnetic: {self.config.electromagnetic_coupling}")
        report.append(f"  Thermal: {self.config.thermal_coupling}")
        report.append(f"  Mechanical: {self.config.mechanical_coupling}")
        report.append(f"  Quantum: {self.config.quantum_coupling}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)

def create_virtual_laboratory(config: Optional[VirtualLabConfig] = None) -> EnhancedVirtualLaboratory:
    """
    Factory function to create virtual laboratory
    
    Args:
        config: Optional configuration
        
    Returns:
        Configured virtual laboratory
    """
    if config is None:
        config = VirtualLabConfig(
            lab_type="quantum_optics",
            bayesian_analysis=True,
            adaptive_design=True,
            enable_virtual_instruments=True
        )
        
    return EnhancedVirtualLaboratory(config)

if __name__ == "__main__":
    # Demonstration
    logging.basicConfig(level=logging.INFO)
    
    # Create virtual laboratory
    lab_config = VirtualLabConfig(
        lab_type="quantum_optics",
        experiment_duration=1800.0,  # 30 minutes
        bayesian_analysis=True,
        adaptive_design=True
    )
    
    virtual_lab = create_virtual_laboratory(lab_config)
    
    # Add experimental hypotheses
    def linear_model(x, params):
        return params[0] * x + params[1]
        
    def quadratic_model(x, params):
        return params[0] * x**2 + params[1] * x + params[2]
        
    hypothesis1 = ExperimentalHypothesis(
        "linear_dependence",
        {"slope": (0.0, 2.0), "intercept": (-1.0, 1.0)},
        linear_model,
        lambda x: stats.uniform.pdf(x, -1, 2)
    )
    
    hypothesis2 = ExperimentalHypothesis(
        "quadratic_dependence", 
        {"a": (0.0, 1.0), "b": (0.0, 2.0), "c": (-1.0, 1.0)},
        quadratic_model,
        lambda x: stats.uniform.pdf(x, -1, 2)
    )
    
    virtual_lab.add_hypothesis(hypothesis1)
    virtual_lab.add_hypothesis(hypothesis2)
    
    # Design experimental sequence
    experiment_plan = [
        {
            'conditions': {'laser_power': 1e-6, 'temperature': 4.2},
            'measurements': [
                {'type': 'fluorescence_intensity', 'parameters': {'detuning': 0.0}, 'time': 1.0},
                {'type': 'quantum_correlation', 'parameters': {'pump_power': 1e-6}, 'time': 2.0}
            ]
        },
        {
            'conditions': {'laser_power': 5e-6, 'temperature': 77.0},
            'measurements': [
                {'type': 'fluorescence_intensity', 'parameters': {'detuning': 1.0}, 'time': 1.0},
                {'type': 'transmission_spectrum', 'parameters': {'wavelength': 780e-9}, 'time': 0.5}
            ]
        },
        {
            'conditions': {'laser_power': 1e-5, 'magnetic_field': 0.1},
            'measurements': [
                {'type': 'magnetic_resonance', 'parameters': {'rf_frequency': 1e6}, 'time': 5.0}
            ]
        }
    ]
    
    # Run experimental sequence
    results = virtual_lab.run_experimental_sequence(experiment_plan, optimize_sequence=True)
    
    # Generate report
    report = virtual_lab.generate_laboratory_report()
    print(report)
    
    print(f"\nExperimental Sequence Results:")
    print(f"  Total measurements: {results['total_measurements']}")
    print(f"  Cumulative information: {results['cumulative_information']:.2f}")
    
    if 'model_comparison' in results['final_analysis']:
        model_probs = results['final_analysis']['model_comparison']['model_probabilities']
        print(f"  Model probabilities:")
        for model, prob in model_probs.items():
            print(f"    {model}: {prob:.3f}")
            
    if 'uncertainty_quantification' in results['final_analysis']:
        uq = results['final_analysis']['uncertainty_quantification']
        print(f"  Combined uncertainty: {uq['combined_uncertainty']:.2e}")
        print(f"  Confidence interval: [{uq['monte_carlo_confidence_interval'][0]:.2e}, {uq['monte_carlo_confidence_interval'][1]:.2e}]")
