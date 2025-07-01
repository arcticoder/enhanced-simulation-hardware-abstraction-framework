"""
Enhanced Precision Measurement Simulation

Implements quantum-limited precision measurements approaching Heisenberg limit:
Δx Δp ≥ ℏ/2

Features:
- Quantum error correction for enhanced precision
- Shot noise limit optimization
- Quantum squeezing for sub-shot-noise sensitivity
- Correlation function analysis
- Multi-parameter estimation with Fisher information
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# Physical constants
HBAR = 1.054571817e-34  # J⋅s
K_B = 1.380649e-23     # J/K
C_LIGHT = 299792458    # m/s

@dataclass
class PrecisionMeasurementConfig:
    """Configuration for precision measurement simulation"""
    measurement_type: str = "quantum_interferometry"  # "quantum_interferometry", "atomic_clock", "gravitational_wave"
    n_measurements: int = 10000
    measurement_time: float = 1.0  # seconds
    quantum_efficiency: float = 0.99
    temperature: float = 4.2  # Kelvin (cryogenic)
    
    # Quantum enhancement parameters
    use_quantum_squeezing: bool = True
    squeezing_parameter: float = 10.0  # dB
    use_quantum_error_correction: bool = True
    error_correction_efficiency: float = 0.95
    
    # Noise parameters
    shot_noise_limited: bool = True
    technical_noise_level: float = 1e-12
    thermal_noise_temperature: float = 300.0  # K
    
    # Multi-parameter estimation
    n_parameters: int = 3
    correlation_matrix_enabled: bool = True
    fisher_information_analysis: bool = True

class EnhancedPrecisionMeasurementSimulator:
    """
    Enhanced precision measurement simulator approaching Heisenberg limit
    
    Implements quantum-enhanced measurements with error correction:
    - Quantum interferometry with squeezing
    - Multi-parameter Fisher information estimation
    - Correlation function analysis
    - Shot noise optimization
    """
    
    def __init__(self, config: PrecisionMeasurementConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize quantum state parameters
        self.n_qubits = int(np.log2(config.n_measurements)) if config.n_measurements > 1 else 1
        self.quantum_state = self._initialize_quantum_state()
        
        # Measurement operators
        self.measurement_operators = self._initialize_measurement_operators()
        
        # Noise characterization
        self.noise_spectrum = self._characterize_noise_spectrum()
        
        # Fisher information matrix for multi-parameter estimation
        if config.fisher_information_analysis:
            self.fisher_matrix = self._initialize_fisher_matrix()
            
        # Quantum error correction parameters
        if config.use_quantum_error_correction:
            self.error_correction_codes = self._initialize_error_correction()
            
        self.logger.info(f"Initialized precision measurement simulator: {config.measurement_type}")
        
    def _initialize_quantum_state(self) -> np.ndarray:
        """Initialize quantum state for measurement"""
        # Start with coherent state |α⟩
        alpha = np.sqrt(self.config.n_measurements)
        
        if self.config.use_quantum_squeezing:
            # Apply squeezing transformation
            r = self.config.squeezing_parameter * np.log(10) / 20  # Convert dB to natural units
            
            # Squeezed coherent state parameters
            state_dim = min(100, self.config.n_measurements)  # Truncate for computational efficiency
            psi = np.zeros(state_dim, dtype=np.complex128)
            
            # Squeezed vacuum with displacement
            for n in range(state_dim):
                if n % 2 == 0:  # Even Fock states
                    try:
                        factorial_n = np.math.factorial(n) if n < 170 else 1e308  # Avoid overflow
                        sqrt_factorial = np.sqrt(float(factorial_n))  # Convert to float first
                        psi[n] = (np.tanh(r)**(n/2) * alpha**(n/2) * 
                                 np.exp(-0.5 * np.abs(alpha)**2) / 
                                 np.sqrt(np.cosh(r)) / sqrt_factorial)
                    except (OverflowError, ValueError, TypeError):
                        psi[n] = 0.0  # Set to zero for problematic values
                             
            # Normalize safely
            norm = np.linalg.norm(psi)
            if norm > 1e-15:  # Avoid division by zero
                psi = psi / norm
            else:
                # Reset to a simple state if normalization fails
                psi = np.zeros_like(psi)
                if len(psi) > 0:
                    psi[0] = 1.0
            
        else:
            # Simple coherent state
            state_dim = min(50, self.config.n_measurements)
            psi = np.zeros(state_dim, dtype=np.complex128)
            
            for n in range(state_dim):
                try:
                    factorial_n = np.math.factorial(n) if n < 170 else 1e308  # Avoid overflow
                    sqrt_factorial = np.sqrt(float(factorial_n))  # Convert to float first
                    psi[n] = (alpha**n * np.exp(-0.5 * np.abs(alpha)**2) / sqrt_factorial)
                except (OverflowError, ValueError, TypeError):
                    psi[n] = 0.0  # Set to zero for problematic values
                         
            psi = psi / np.linalg.norm(psi)
            
        return psi
        
    def _initialize_measurement_operators(self) -> Dict[str, np.ndarray]:
        """Initialize quantum measurement operators"""
        operators = {}
        state_dim = len(self.quantum_state)
        
        # Position operator (quadrature X)
        X = np.zeros((state_dim, state_dim), dtype=np.complex128)
        for n in range(state_dim - 1):
            X[n, n+1] = np.sqrt(n + 1)
            X[n+1, n] = np.sqrt(n + 1)
        X = X / np.sqrt(2)
        operators['position'] = X
        
        # Momentum operator (quadrature P)
        P = np.zeros((state_dim, state_dim), dtype=np.complex128)
        for n in range(state_dim - 1):
            P[n, n+1] = 1j * np.sqrt(n + 1)
            P[n+1, n] = -1j * np.sqrt(n + 1)
        P = P / np.sqrt(2)
        operators['momentum'] = P
        
        # Number operator
        N = np.diag(np.arange(state_dim))
        operators['number'] = N
        
        # Pauli operators for qubit measurements
        sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        
        operators['pauli_x'] = sigma_x
        operators['pauli_y'] = sigma_y
        operators['pauli_z'] = sigma_z
        
        return operators
        
    def _characterize_noise_spectrum(self) -> Dict[str, np.ndarray]:
        """Characterize noise spectrum for measurement"""
        # Frequency array
        frequencies = np.logspace(-3, 6, 1000)  # 1 mHz to 1 MHz
        
        noise_spectrum = {}
        
        # Shot noise (white noise at quantum limit)
        shot_noise = np.full_like(frequencies, HBAR)
        noise_spectrum['shot_noise'] = shot_noise
        
        # Technical noise (1/f noise)
        technical_noise = self.config.technical_noise_level / frequencies
        noise_spectrum['technical_noise'] = technical_noise
        
        # Thermal noise
        thermal_noise = (4 * K_B * self.config.thermal_noise_temperature / 
                        (2 * np.pi * frequencies))
        noise_spectrum['thermal_noise'] = thermal_noise
        
        # Total noise
        total_noise = shot_noise + technical_noise + thermal_noise
        if self.config.use_quantum_squeezing:
            # Squeezing reduces noise below shot noise limit
            squeezing_factor = 10**(-self.config.squeezing_parameter / 10)
            total_noise = shot_noise * squeezing_factor + technical_noise + thermal_noise
            
        noise_spectrum['total_noise'] = total_noise
        noise_spectrum['frequencies'] = frequencies
        
        return noise_spectrum
        
    def _initialize_fisher_matrix(self) -> np.ndarray:
        """Initialize Fisher information matrix for multi-parameter estimation"""
        n_params = self.config.n_parameters
        fisher_matrix = np.zeros((n_params, n_params))
        
        # Diagonal terms (individual parameter sensitivities)
        for i in range(n_params):
            # Shot noise limited sensitivity
            fisher_matrix[i, i] = self.config.n_measurements / HBAR
            
            # Apply quantum enhancement
            if self.config.use_quantum_squeezing:
                squeezing_enhancement = 10**(self.config.squeezing_parameter / 10)
                fisher_matrix[i, i] *= squeezing_enhancement
                
        # Off-diagonal terms (parameter correlations)
        if self.config.correlation_matrix_enabled:
            for i in range(n_params):
                for j in range(i + 1, n_params):
                    # Correlation strength decreases with parameter separation
                    correlation = 0.1 * np.exp(-0.5 * (i - j)**2)
                    fisher_matrix[i, j] = correlation * np.sqrt(fisher_matrix[i, i] * fisher_matrix[j, j])
                    fisher_matrix[j, i] = fisher_matrix[i, j]
                    
        return fisher_matrix
        
    def _initialize_error_correction(self) -> Dict[str, Any]:
        """Initialize quantum error correction parameters"""
        error_correction = {}
        
        # Stabilizer codes for quantum error correction
        # Simple 3-qubit bit flip code
        stabilizer_generators = [
            np.kron(np.kron(self.measurement_operators['pauli_z'], 
                           self.measurement_operators['pauli_z']),
                   np.eye(2)),
            np.kron(np.eye(2),
                   np.kron(self.measurement_operators['pauli_z'],
                          self.measurement_operators['pauli_z']))
        ]
        
        error_correction['stabilizers'] = stabilizer_generators
        error_correction['logical_operators'] = [
            np.kron(np.kron(self.measurement_operators['pauli_x'],
                           self.measurement_operators['pauli_x']),
                   self.measurement_operators['pauli_x'])
        ]
        
        # Error syndrome lookup table
        error_correction['syndrome_table'] = {
            (0, 0): "no_error",
            (1, 0): "error_qubit_1",
            (0, 1): "error_qubit_3", 
            (1, 1): "error_qubit_2"
        }
        
        return error_correction
        
    def perform_quantum_measurement(self, 
                                  parameter_values: np.ndarray,
                                  measurement_operator: str = "position") -> Dict[str, np.ndarray]:
        """
        Perform quantum-enhanced precision measurement
        
        Args:
            parameter_values: Parameters to estimate
            measurement_operator: Type of measurement operator
            
        Returns:
            Measurement results with quantum enhancement
        """
        n_measurements = self.config.n_measurements
        
        # Get measurement operator
        if measurement_operator in self.measurement_operators:
            M = self.measurement_operators[measurement_operator]
        else:
            self.logger.warning(f"Unknown operator {measurement_operator}, using position")
            M = self.measurement_operators['position']
            
        # Expected value and variance from quantum state
        psi = self.quantum_state
        
        # Check for valid quantum state
        if np.all(np.isnan(psi)) or np.all(psi == 0):
            # Return classical measurement if quantum state is invalid
            self.logger.warning("Invalid quantum state, falling back to classical measurement")
            return {
                'value': np.random.normal(0, 1e-6),
                'uncertainty': 1e-6,
                'enhancement_factor': 1.0,
                'squeezing_improvement': 1.0,
                'precision': 1e-6
            }
        
        expectation_value = np.real(np.conj(psi) @ M @ psi)
        
        # Compute variance
        M_squared = M @ M
        second_moment = np.real(np.conj(psi) @ M_squared @ psi)
        variance = max(second_moment - expectation_value**2, 1e-30)  # Ensure positive variance
        
        # Apply quantum enhancement
        if self.config.use_quantum_squeezing:
            # Reduce variance below shot noise limit
            squeezing_factor = 10**(-self.config.squeezing_parameter / 10)
            variance *= squeezing_factor
            
        # Generate measurement results
        measurements = np.random.normal(
            expectation_value + np.sum(parameter_values),  # Parameters shift expectation
            np.sqrt(variance / n_measurements),  # Reduced by √N measurements
            n_measurements
        )
        
        # Apply quantum error correction
        if self.config.use_quantum_error_correction:
            corrected_measurements = self._apply_error_correction(measurements)
            measurements = corrected_measurements
            
        # Compute precision metrics
        measurement_precision = np.std(measurements) / np.sqrt(n_measurements)
        
        # Heisenberg limit comparison
        standard_quantum_limit = np.sqrt(HBAR / (2 * n_measurements))
        heisenberg_limit = np.sqrt(HBAR / (4 * n_measurements))  # With squeezing
        
        enhancement_factor = standard_quantum_limit / measurement_precision
        
        results = {
            'measurements': measurements,
            'expectation_value': expectation_value,
            'variance': variance,
            'precision': measurement_precision,
            'standard_quantum_limit': standard_quantum_limit,
            'heisenberg_limit': heisenberg_limit,
            'enhancement_factor': enhancement_factor,
            'quantum_efficiency': self.config.quantum_efficiency
        }
        
        return results
        
    def _apply_error_correction(self, measurements: np.ndarray) -> np.ndarray:
        """Apply quantum error correction to measurements"""
        corrected = measurements.copy()
        
        # Simulate random errors
        error_probability = 1 - self.config.error_correction_efficiency
        n_errors = int(len(measurements) * error_probability)
        
        if n_errors > 0:
            error_indices = np.random.choice(len(measurements), n_errors, replace=False)
            
            # Apply corrections (simplified)
            for idx in error_indices:
                # Detect and correct bit flip errors
                syndrome = np.random.randint(0, 4)  # Simplified syndrome
                if syndrome != 0:  # Error detected
                    # Apply correction (flip back)
                    corrected[idx] = -corrected[idx]
                    
        return corrected
        
    def multi_parameter_estimation(self, 
                                 true_parameters: np.ndarray,
                                 n_trials: int = 1000) -> Dict[str, np.ndarray]:
        """
        Perform multi-parameter estimation with Fisher information
        
        Args:
            true_parameters: True parameter values
            n_trials: Number of estimation trials
            
        Returns:
            Parameter estimation results
        """
        n_params = len(true_parameters)
        estimated_parameters = np.zeros((n_trials, n_params))
        estimation_errors = np.zeros((n_trials, n_params))
        
        # Cramér-Rao lower bound from Fisher matrix
        try:
            fisher_inverse = np.linalg.inv(self.fisher_matrix)
            cramer_rao_bounds = np.sqrt(np.diag(fisher_inverse))
        except np.linalg.LinAlgError:
            self.logger.warning("Fisher matrix singular, using identity")
            cramer_rao_bounds = np.ones(n_params) * np.sqrt(HBAR / self.config.n_measurements)
            
        for trial in range(n_trials):
            # Perform measurements for each parameter
            trial_estimates = np.zeros(n_params)
            
            for param_idx in range(n_params):
                # Measurement sensitive to this parameter
                measurement_results = self.perform_quantum_measurement(
                    np.zeros(n_params),  # Zero background
                    measurement_operator="position"
                )
                
                measurements = measurement_results['measurements']
                
                # Add parameter-dependent signal
                signal_strength = 1.0  # Normalized signal
                measurements += signal_strength * true_parameters[param_idx]
                
                # Estimate parameter from measurements
                trial_estimates[param_idx] = np.mean(measurements) / signal_strength
                
            estimated_parameters[trial] = trial_estimates
            estimation_errors[trial] = trial_estimates - true_parameters
            
        # Compute statistical properties
        mean_estimates = np.mean(estimated_parameters, axis=0)
        std_estimates = np.std(estimated_parameters, axis=0)
        bias = mean_estimates - true_parameters
        
        # Covariance matrix of estimates
        covariance_matrix = np.cov(estimated_parameters.T)
        
        # Quantum enhancement metrics
        classical_limit = np.sqrt(HBAR / self.config.n_measurements) * np.ones(n_params)
        enhancement_factors = classical_limit / std_estimates
        
        results = {
            'estimated_parameters': estimated_parameters,
            'mean_estimates': mean_estimates,
            'estimation_errors': estimation_errors,
            'standard_deviations': std_estimates,
            'bias': bias,
            'covariance_matrix': covariance_matrix,
            'cramer_rao_bounds': cramer_rao_bounds,
            'enhancement_factors': enhancement_factors,
            'fisher_matrix': self.fisher_matrix
        }
        
        return results
        
    def analyze_correlation_functions(self, 
                                   measurement_data: np.ndarray,
                                   max_lag: int = 100) -> Dict[str, np.ndarray]:
        """
        Analyze temporal correlation functions of measurements
        
        Args:
            measurement_data: Time series measurement data
            max_lag: Maximum lag for correlation analysis
            
        Returns:
            Correlation function analysis
        """
        n_points = len(measurement_data)
        lags = np.arange(-max_lag, max_lag + 1)
        
        # Autocorrelation function
        autocorrelation = np.correlate(measurement_data, measurement_data, mode='full')
        center_idx = len(autocorrelation) // 2
        autocorr_normalized = autocorrelation[center_idx - max_lag:center_idx + max_lag + 1]
        
        # Safe normalization - check bounds
        if len(autocorr_normalized) > max_lag and max_lag < len(autocorr_normalized):
            autocorr_normalized = autocorr_normalized / autocorr_normalized[max_lag]  # Normalize
        elif len(autocorr_normalized) > 0:
            autocorr_normalized = autocorr_normalized / autocorr_normalized[0]  # Use first element
        else:
            autocorr_normalized = np.ones_like(lags, dtype=float)  # Fallback
        
        # Power spectral density
        frequencies = np.fft.fftfreq(n_points, d=self.config.measurement_time / n_points)
        power_spectrum = np.abs(np.fft.fft(measurement_data))**2
        
        # Allan variance for frequency stability analysis
        allan_times = np.logspace(-3, np.log10(self.config.measurement_time), 20)
        allan_variance = np.zeros_like(allan_times)
        
        for i, tau in enumerate(allan_times):
            n_samples = int(tau * n_points / self.config.measurement_time)
            if n_samples >= 2:
                allan_variance[i] = self._compute_allan_variance(measurement_data, n_samples)
            else:
                allan_variance[i] = np.var(measurement_data)
                
        # Quantum limit comparison
        quantum_limited_variance = HBAR / (2 * self.config.n_measurements)
        
        results = {
            'lags': lags,
            'autocorrelation': autocorr_normalized,
            'frequencies': frequencies[:n_points//2],
            'power_spectrum': power_spectrum[:n_points//2],
            'allan_times': allan_times,
            'allan_variance': allan_variance,
            'quantum_limit_variance': quantum_limited_variance,
            'noise_spectrum': self.noise_spectrum
        }
        
        return results
        
    def _compute_allan_variance(self, data: np.ndarray, n_samples: int) -> float:
        """Compute Allan variance for given averaging time"""
        n_groups = len(data) // n_samples
        if n_groups < 2:
            return np.var(data)
            
        group_averages = np.zeros(n_groups)
        for i in range(n_groups):
            start_idx = i * n_samples
            end_idx = start_idx + n_samples
            group_averages[i] = np.mean(data[start_idx:end_idx])
            
        # Allan variance: (1/2) * <(y_{i+1} - y_i)^2>
        if len(group_averages) > 1:
            allan_var = 0.5 * np.mean(np.diff(group_averages)**2)
        else:
            allan_var = np.var(group_averages)
            
        return allan_var
        
    def optimize_measurement_protocol(self, 
                                    target_precision: float,
                                    optimization_parameters: List[str]) -> Dict[str, Any]:
        """
        Optimize measurement protocol to achieve target precision
        
        Args:
            target_precision: Target measurement precision
            optimization_parameters: Parameters to optimize
            
        Returns:
            Optimized protocol parameters
        """
        def objective_function(params):
            """Objective function for optimization"""
            # Update configuration with trial parameters
            config_copy = self.config
            
            for i, param_name in enumerate(optimization_parameters):
                if param_name == 'squeezing_parameter':
                    config_copy.squeezing_parameter = params[i]
                elif param_name == 'n_measurements':
                    config_copy.n_measurements = int(params[i])
                elif param_name == 'measurement_time':
                    config_copy.measurement_time = params[i]
                    
            # Simulate measurement with trial parameters
            temp_simulator = EnhancedPrecisionMeasurementSimulator(config_copy)
            measurement_result = temp_simulator.perform_quantum_measurement(
                np.array([0.0])  # Test parameter
            )
            
            achieved_precision = measurement_result['precision']
            
            # Objective: minimize difference from target precision
            return abs(achieved_precision - target_precision)
            
        # Initial parameter values
        initial_params = []
        bounds = []
        
        for param_name in optimization_parameters:
            if param_name == 'squeezing_parameter':
                initial_params.append(self.config.squeezing_parameter)
                bounds.append((0, 50))  # 0-50 dB squeezing
            elif param_name == 'n_measurements':
                initial_params.append(self.config.n_measurements)
                bounds.append((100, 100000))  # 100 to 100k measurements
            elif param_name == 'measurement_time':
                initial_params.append(self.config.measurement_time)
                bounds.append((0.001, 1000))  # 1 ms to 1000 s
                
        # Perform optimization
        optimization_result = minimize(
            objective_function,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        # Extract optimized parameters
        optimized_config = {}
        for i, param_name in enumerate(optimization_parameters):
            optimized_config[param_name] = optimization_result.x[i]
            
        results = {
            'optimized_parameters': optimized_config,
            'achieved_precision': optimization_result.fun + target_precision,
            'optimization_success': optimization_result.success,
            'optimization_message': optimization_result.message,
            'target_precision': target_precision
        }
        
        return results
        
    def generate_measurement_report(self, 
                                  measurement_results: Dict[str, Any]) -> str:
        """Generate comprehensive measurement report"""
        report = []
        report.append("=" * 60)
        report.append("ENHANCED PRECISION MEASUREMENT REPORT")
        report.append("=" * 60)
        
        report.append(f"\nMeasurement Configuration:")
        report.append(f"  Type: {self.config.measurement_type}")
        report.append(f"  Number of measurements: {self.config.n_measurements}")
        report.append(f"  Measurement time: {self.config.measurement_time:.3f} s")
        report.append(f"  Quantum efficiency: {self.config.quantum_efficiency:.3f}")
        
        if 'precision' in measurement_results:
            report.append(f"\nPrecision Analysis:")
            report.append(f"  Achieved precision: {measurement_results['precision']:.2e}")
            report.append(f"  Standard quantum limit: {measurement_results['standard_quantum_limit']:.2e}")
            report.append(f"  Heisenberg limit: {measurement_results['heisenberg_limit']:.2e}")
            report.append(f"  Enhancement factor: {measurement_results['enhancement_factor']:.2f}")
            
        if self.config.use_quantum_squeezing:
            report.append(f"\nQuantum Enhancement:")
            report.append(f"  Squeezing parameter: {self.config.squeezing_parameter:.1f} dB")
            squeezing_improvement = 10**(self.config.squeezing_parameter / 10)
            report.append(f"  Sensitivity improvement: {squeezing_improvement:.1f}×")
            
        if self.config.use_quantum_error_correction:
            report.append(f"\nQuantum Error Correction:")
            report.append(f"  Error correction efficiency: {self.config.error_correction_efficiency:.3f}")
            
        report.append(f"\nNoise Analysis:")
        shot_noise_level = np.sqrt(HBAR / self.config.n_measurements)
        report.append(f"  Shot noise level: {shot_noise_level:.2e}")
        report.append(f"  Technical noise level: {self.config.technical_noise_level:.2e}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)

def create_precision_measurement_simulator(config: Optional[PrecisionMeasurementConfig] = None) -> EnhancedPrecisionMeasurementSimulator:
    """
    Factory function to create precision measurement simulator
    
    Args:
        config: Optional configuration
        
    Returns:
        Configured measurement simulator
    """
    if config is None:
        config = PrecisionMeasurementConfig(
            measurement_type="quantum_interferometry",
            n_measurements=10000,
            use_quantum_squeezing=True,
            squeezing_parameter=10.0,
            use_quantum_error_correction=True
        )
        
    return EnhancedPrecisionMeasurementSimulator(config)

if __name__ == "__main__":
    # Demonstration
    logging.basicConfig(level=logging.INFO)
    
    # Create precision measurement simulator
    measurement_config = PrecisionMeasurementConfig(
        measurement_type="quantum_interferometry",
        n_measurements=5000,
        use_quantum_squeezing=True,
        squeezing_parameter=15.0,  # 15 dB squeezing
        use_quantum_error_correction=True
    )
    
    measurement_simulator = create_precision_measurement_simulator(measurement_config)
    
    # Perform single parameter measurement
    test_parameter = np.array([1e-12])  # Test parameter value
    measurement_results = measurement_simulator.perform_quantum_measurement(test_parameter)
    
    # Multi-parameter estimation
    true_params = np.array([1e-12, 2e-12, 5e-13])
    multi_param_results = measurement_simulator.multi_parameter_estimation(true_params, n_trials=100)
    
    # Correlation analysis
    test_data = measurement_results['measurements']
    correlation_results = measurement_simulator.analyze_correlation_functions(test_data)
    
    # Generate report
    report = measurement_simulator.generate_measurement_report(measurement_results)
    print(report)
    
    print(f"\nMulti-parameter estimation:")
    print(f"  True parameters: {true_params}")
    print(f"  Estimated parameters: {multi_param_results['mean_estimates']}")
    print(f"  Standard deviations: {multi_param_results['standard_deviations']}")
    print(f"  Enhancement factors: {multi_param_results['enhancement_factors']}")
    print(f"  Cramér-Rao bounds: {multi_param_results['cramer_rao_bounds']}")
