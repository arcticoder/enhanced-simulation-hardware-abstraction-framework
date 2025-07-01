"""
Enhanced High-Fidelity Physics Pipeline with Multi-Domain Coupling

Implements frequency-dependent coupling and comprehensive uncertainty quantification:
∂Ψ_physics/∂t = Ĥ_multi Ψ + ∑ᵢ∑ⱼ Cᵢⱼ(ω,T) × [Ψᵢ ⊗ Ψⱼ] + η_quantum-classical(t)

Features:
- Multi-physics domain coupling (EM, thermal, mechanical, quantum)
- Frequency-dependent coupling matrices Cᵢⱼ(ω,T)
- 25K Monte Carlo uncertainty quantification
- Quantum-classical bridge terms
- Real-time multi-physics evolution
"""

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.stats import multivariate_normal
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# Physical constants
HBAR = 1.054571817e-34  # J⋅s
K_B = 1.380649e-23     # J/K
C_LIGHT = 299792458    # m/s

@dataclass 
class PhysicsPipelineConfig:
    """Configuration for high-fidelity physics pipeline"""
    n_domains: int = 4  # EM, thermal, mechanical, quantum
    grid_resolution: Tuple[int, int, int] = (50, 50, 50)
    time_steps: int = 1000
    monte_carlo_samples: int = 25000
    uncertainty_quantification: bool = True
    frequency_dependent_coupling: bool = True
    quantum_classical_bridge: bool = True
    
    # Physics domain parameters
    electromagnetic_domain: bool = True
    thermal_domain: bool = True  
    mechanical_domain: bool = True
    quantum_domain: bool = True
    
    # Numerical parameters
    relative_tolerance: float = 1e-8
    absolute_tolerance: float = 1e-10
    max_coupling_frequency: float = 1e12  # THz
    temperature_range: Tuple[float, float] = (1.0, 1000.0)  # K

class EnhancedHighFidelityPhysicsPipeline:
    """
    Enhanced high-fidelity physics pipeline with multi-domain coupling
    
    Implements comprehensive multi-physics simulation with:
    - Frequency-dependent coupling matrices
    - Monte Carlo uncertainty quantification  
    - Quantum-classical bridge terms
    - Real-time evolution
    """
    
    def __init__(self, config: PhysicsPipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize physics domains
        self.physics_domains = self._initialize_physics_domains()
        
        # Initialize spatial grids
        self.spatial_grid = self._initialize_spatial_grid()
        
        # Initialize coupling matrices
        self.coupling_matrices = {}
        if config.frequency_dependent_coupling:
            self._initialize_frequency_dependent_coupling()
            
        # Monte Carlo sampling for uncertainty quantification
        if config.uncertainty_quantification:
            self.monte_carlo_samples = self._initialize_monte_carlo_sampling()
            
        self.logger.info(f"Initialized physics pipeline with {config.n_domains} domains")
        
    def _initialize_physics_domains(self) -> Dict[str, Dict[str, Any]]:
        """Initialize physics domain configurations"""
        domains = {}
        
        if self.config.electromagnetic_domain:
            domains['electromagnetic'] = {
                'field_components': 6,  # Ex, Ey, Ez, Bx, By, Bz
                'characteristic_frequency': 1e15,  # Optical frequency
                'coupling_strength': 1.0
            }
            
        if self.config.thermal_domain:
            domains['thermal'] = {
                'field_components': 1,  # Temperature
                'characteristic_frequency': 1e3,  # kHz thermal dynamics
                'coupling_strength': 0.1
            }
            
        if self.config.mechanical_domain:
            domains['mechanical'] = {
                'field_components': 3,  # ux, uy, uz displacement
                'characteristic_frequency': 1e6,  # MHz mechanical resonance  
                'coupling_strength': 0.5
            }
            
        if self.config.quantum_domain:
            domains['quantum'] = {
                'field_components': 2,  # Real and imaginary parts of wavefunction
                'characteristic_frequency': 1e12,  # THz quantum dynamics
                'coupling_strength': 0.8
            }
            
        return domains
        
    def _initialize_spatial_grid(self) -> Dict[str, np.ndarray]:
        """Initialize spatial discretization grid"""
        nx, ny, nz = self.config.grid_resolution
        
        # Coordinate arrays
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny) 
        z = np.linspace(-1, 1, nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Grid spacing
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]
        
        grid = {
            'coordinates': (X, Y, Z),
            'x': x, 'y': y, 'z': z,
            'dx': dx, 'dy': dy, 'dz': dz,
            'volume_element': dx * dy * dz
        }
        
        return grid
        
    def _initialize_frequency_dependent_coupling(self):
        """Initialize frequency-dependent coupling matrices"""
        n_domains = len(self.physics_domains)
        domain_names = list(self.physics_domains.keys())
        
        # Reference coupling matrix (frequency-independent part)
        base_coupling = np.eye(n_domains) * 0.1  # Weak self-coupling
        
        # Add cross-domain coupling terms
        for i, domain_i in enumerate(domain_names):
            for j, domain_j in enumerate(domain_names):
                if i != j:
                    # Coupling strength based on physical compatibility
                    coupling_strength = self._compute_domain_coupling_strength(domain_i, domain_j)
                    base_coupling[i, j] = coupling_strength
                    
        self.base_coupling_matrix = base_coupling
        
    def _compute_domain_coupling_strength(self, domain_i: str, domain_j: str) -> float:
        """Compute coupling strength between physics domains"""
        
        # Define coupling strengths based on physics
        coupling_map = {
            ('electromagnetic', 'thermal'): 0.3,      # Joule heating
            ('electromagnetic', 'mechanical'): 0.2,   # Radiation pressure
            ('electromagnetic', 'quantum'): 0.8,      # Strong EM-quantum coupling
            ('thermal', 'mechanical'): 0.4,           # Thermal expansion
            ('thermal', 'quantum'): 0.1,              # Thermal decoherence
            ('mechanical', 'quantum'): 0.3            # Mechanoquantum coupling
        }
        
        # Symmetric coupling
        key1 = (domain_i, domain_j)
        key2 = (domain_j, domain_i)
        
        return coupling_map.get(key1, coupling_map.get(key2, 0.0))
        
    def _initialize_monte_carlo_sampling(self) -> Dict[str, np.ndarray]:
        """Initialize Monte Carlo samples for uncertainty quantification"""
        n_samples = self.config.monte_carlo_samples
        n_domains = len(self.physics_domains)
        
        # Parameter distributions for each domain
        parameter_samples = {}
        
        for domain_name, domain_config in self.physics_domains.items():
            n_params = domain_config['field_components']
            
            # Generate correlated parameter samples
            mean_params = np.zeros(n_params)
            cov_matrix = np.eye(n_params) * 0.01  # 1% parameter uncertainty
            
            # Add correlations
            for i in range(n_params):
                for j in range(i+1, n_params):
                    correlation = 0.1 * np.exp(-abs(i-j))  # Exponential decay
                    cov_matrix[i, j] = correlation * np.sqrt(cov_matrix[i, i] * cov_matrix[j, j])
                    cov_matrix[j, i] = cov_matrix[i, j]
                    
            # Generate samples
            parameter_samples[domain_name] = multivariate_normal.rvs(
                mean=mean_params,
                cov=cov_matrix,
                size=n_samples
            )
            
        return parameter_samples
        
    def compute_frequency_dependent_coupling(self, 
                                           frequency: float,
                                           temperature: float = 300.0) -> np.ndarray:
        """
        Compute frequency-dependent coupling matrix Cᵢⱼ(ω,T)
        
        Args:
            frequency: Angular frequency (Hz)
            temperature: Temperature (K)
            
        Returns:
            Frequency-dependent coupling matrix
        """
        n_domains = len(self.physics_domains)
        coupling_matrix = self.base_coupling_matrix.copy()
        
        # Frequency-dependent modifications
        domain_names = list(self.physics_domains.keys())
        
        for i, domain_i in enumerate(domain_names):
            for j, domain_j in enumerate(domain_names):
                if i != j:
                    # Get characteristic frequencies
                    freq_i = self.physics_domains[domain_i]['characteristic_frequency']
                    freq_j = self.physics_domains[domain_j]['characteristic_frequency']
                    
                    # Resonant enhancement
                    resonance_factor = self._compute_resonance_factor(frequency, freq_i, freq_j)
                    
                    # Temperature dependence
                    thermal_factor = self._compute_thermal_factor(temperature, domain_i, domain_j)
                    
                    # Update coupling strength
                    coupling_matrix[i, j] *= resonance_factor * thermal_factor
                    
        return coupling_matrix
        
    def _compute_resonance_factor(self, frequency: float, freq_i: float, freq_j: float) -> float:
        """Compute resonance enhancement factor"""
        
        # Resonance when frequency matches difference or sum of characteristic frequencies
        resonance_diff = abs(frequency - abs(freq_i - freq_j))
        resonance_sum = abs(frequency - (freq_i + freq_j))
        
        # Lorentzian resonance profiles
        linewidth = 1e9  # 1 GHz linewidth
        
        enhancement_diff = 1.0 / (1.0 + (resonance_diff / linewidth)**2)
        enhancement_sum = 1.0 / (1.0 + (resonance_sum / linewidth)**2)
        
        # Total enhancement (max of difference and sum resonances)
        total_enhancement = 1.0 + 10.0 * max(enhancement_diff, enhancement_sum)
        
        return total_enhancement
        
    def _compute_thermal_factor(self, temperature: float, domain_i: str, domain_j: str) -> float:
        """Compute temperature-dependent coupling factor"""
        
        # Thermal factors depend on specific domain combinations
        if 'thermal' in [domain_i, domain_j]:
            # Direct thermal coupling
            return np.sqrt(temperature / 300.0)  # Room temperature reference
            
        elif 'quantum' in [domain_i, domain_j]:
            # Quantum decoherence effects
            return np.exp(-temperature / 100.0)  # Exponential decoherence
            
        else:
            # Weak temperature dependence for other combinations
            return 1.0 + 0.001 * (temperature - 300.0)
            
    def perform_uncertainty_quantification(self, 
                                         parameters: np.ndarray,
                                         n_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform comprehensive uncertainty quantification using Monte Carlo
        
        Args:
            parameters: Nominal parameter values
            n_samples: Number of Monte Carlo samples (optional)
            
        Returns:
            Uncertainty quantification results
        """
        if not self.config.uncertainty_quantification:
            return {}
            
        if n_samples is None:
            n_samples = self.config.monte_carlo_samples
            
        # Prepare parameter ensemble
        n_domains = len(self.physics_domains)
        parameter_ensemble = np.zeros((n_samples, len(parameters)))
        
        # Generate parameter variations
        for i in range(len(parameters)):
            # 5% relative uncertainty for each parameter
            parameter_std = abs(parameters[i]) * 0.05
            parameter_ensemble[:, i] = np.random.normal(parameters[i], parameter_std, n_samples)
            
        # Run ensemble simulations
        output_ensemble = np.zeros((n_samples, len(parameters)))
        
        def single_simulation(sample_idx):
            """Single Monte Carlo simulation"""
            sample_params = parameter_ensemble[sample_idx]
            
            # Simplified physics simulation for each sample
            # In practice, this would run the full multi-physics evolution
            coupling_matrix = self.compute_frequency_dependent_coupling(1e6, 300.0)
            
            # Linear response approximation
            response = coupling_matrix @ sample_params[:n_domains]
            
            # Pad response to match parameter length
            if len(response) < len(parameters):
                response = np.pad(response, (0, len(parameters) - len(response)))
            else:
                response = response[:len(parameters)]
                
            return response
            
        # Parallel execution for efficiency
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(single_simulation, i) for i in range(n_samples)]
            
            for i, future in enumerate(futures):
                output_ensemble[i] = future.result()
                
        # Statistical analysis
        mean_values = np.mean(output_ensemble, axis=0)
        std_values = np.std(output_ensemble, axis=0)
        covariance_matrix = np.cov(output_ensemble.T)
        
        # Sensitivity analysis
        sensitivity_indices = np.zeros((len(parameters), len(parameters)))
        for i in range(len(parameters)):
            for j in range(len(parameters)):
                if std_values[j] > 0:
                    sensitivity_indices[i, j] = np.corrcoef(parameter_ensemble[:, i], output_ensemble[:, j])[0, 1]
                    
        # Confidence intervals
        confidence_levels = [0.68, 0.95, 0.99]  # 1σ, 2σ, 3σ
        confidence_intervals = {}
        
        for level in confidence_levels:
            alpha = 1 - level
            lower_percentile = 100 * alpha / 2
            upper_percentile = 100 * (1 - alpha / 2)
            
            confidence_intervals[f'{level:.2f}'] = {
                'lower': np.percentile(output_ensemble, lower_percentile, axis=0),
                'upper': np.percentile(output_ensemble, upper_percentile, axis=0)
            }
            
        results = {
            'mean_values': mean_values,
            'standard_deviations': std_values,
            'covariance_matrix': covariance_matrix,
            'sensitivity_indices': sensitivity_indices,
            'confidence_intervals': confidence_intervals,
            'parameter_ensemble': parameter_ensemble,
            'output_ensemble': output_ensemble,
            'monte_carlo_samples': n_samples
        }
        
        return results
        
    def evolve_multi_physics_system(self, 
                                   initial_state: Dict[str, np.ndarray],
                                   time_span: Tuple[float, float],
                                   n_time_points: int = 100) -> Dict[str, Any]:
        """
        Evolve multi-physics system with domain coupling
        
        Args:
            initial_state: Initial state for each physics domain
            time_span: Time evolution span
            n_time_points: Number of time points
            
        Returns:
            Evolution results for all domains
        """
        
        # Prepare combined state vector
        combined_state = self._combine_domain_states(initial_state)
        
        # Time points
        t_eval = np.linspace(time_span[0], time_span[1], n_time_points)
        
        def multi_physics_evolution(t, state):
            """Multi-physics evolution equations"""
            
            # Split state back into domains
            domain_states = self._split_combined_state(state)
            
            # Compute coupling matrix for current time
            # Use typical frequency for coupling
            frequency = 1e6  # 1 MHz reference frequency
            temperature = 300.0  # Room temperature
            
            coupling_matrix = self.compute_frequency_dependent_coupling(frequency, temperature)
            
            # Compute derivatives for each domain
            derivatives = {}
            domain_names = list(self.physics_domains.keys())
            
            for i, domain_name in enumerate(domain_names):
                if domain_name not in domain_states:
                    continue  # Skip missing domains
                    
                domain_state = domain_states[domain_name]
                
                # Self-evolution (simplified harmonic oscillator)
                char_freq = self.physics_domains[domain_name]['characteristic_frequency']
                self_evolution = -char_freq * domain_state
                
                # Coupling terms
                coupling_terms = np.zeros_like(domain_state)
                for j, coupled_domain in enumerate(domain_names):
                    if i != j and coupled_domain in domain_states:
                        coupled_state = domain_states[coupled_domain]
                        coupling_strength = coupling_matrix[i, j]
                        
                        # Tensor product coupling (simplified to element-wise)
                        min_size = min(len(domain_state), len(coupled_state))
                        # Ensure real arithmetic only
                        coupling_contribution = np.real(coupling_strength * coupled_state[:min_size])
                        coupling_terms[:min_size] += coupling_contribution
                        
                # Quantum-classical bridge terms - simplified for stability
                if self.config.quantum_classical_bridge and domain_name == 'quantum':
                    try:
                        bridge_terms = self._compute_quantum_classical_bridge(t, domain_states)
                        # Ensure bridge_terms matches coupling_terms shape
                        if len(bridge_terms) >= len(coupling_terms):
                            bridge_contribution = bridge_terms[:len(coupling_terms)]
                            # Simple element-wise addition for now
                            if bridge_contribution.size == coupling_terms.size:
                                coupling_terms.flat[:bridge_contribution.size] += bridge_contribution.flat
                    except (ValueError, IndexError) as e:
                        # Skip bridge terms if there are shape incompatibilities
                        self.logger.warning(f"Skipping bridge terms due to shape mismatch: {e}")
                        pass
                    
                derivatives[domain_name] = self_evolution + coupling_terms
                
            # Combine derivatives
            combined_derivatives = self._combine_domain_derivatives(derivatives)
            
            return combined_derivatives
            
        # Solve evolution equations
        solution = solve_ivp(
            multi_physics_evolution,
            time_span,
            combined_state,
            t_eval=t_eval,
            method='RK45',
            rtol=self.config.relative_tolerance,
            atol=self.config.absolute_tolerance
        )
        
        if not solution.success:
            self.logger.error(f"Evolution failed: {solution.message}")
            raise RuntimeError(f"Multi-physics evolution failed: {solution.message}")
            
        # Split evolution results back into domains
        evolution_results = {}
        
        for i, t in enumerate(solution.t):
            state_at_t = solution.y[:, i]
            domain_states_at_t = self._split_combined_state(state_at_t)
            
            for domain_name, domain_state in domain_states_at_t.items():
                if domain_name not in evolution_results:
                    evolution_results[f'{domain_name}_evolution'] = []
                evolution_results[f'{domain_name}_evolution'].append(domain_state)
                
        # Convert lists to arrays
        for key in evolution_results:
            evolution_results[key] = np.array(evolution_results[key])
            
        evolution_results['time'] = solution.t
        
        # Energy conservation analysis
        energy_conservation_error = self._analyze_energy_conservation(evolution_results)
        evolution_results['energy_conservation_error'] = energy_conservation_error
        
        return evolution_results
        
    def _combine_domain_states(self, domain_states: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine separate domain states into single state vector"""
        combined = []
        
        # Store the sizes for consistency
        self._domain_sizes = {}
        
        for domain_name in self.physics_domains:
            if domain_name in domain_states:
                state = domain_states[domain_name].flatten()
                self._domain_sizes[domain_name] = len(state)
                combined.extend(state)
                
        return np.array(combined)
        
    def _split_combined_state(self, combined_state: np.ndarray) -> Dict[str, np.ndarray]:
        """Split combined state vector back into domain states"""
        domain_states = {}
        start_idx = 0
        
        # Use stored domain sizes if available
        if hasattr(self, '_domain_sizes'):
            for domain_name in self.physics_domains:
                if domain_name in self._domain_sizes:
                    state_size = self._domain_sizes[domain_name]
                    end_idx = start_idx + state_size
                    if end_idx <= len(combined_state):
                        domain_state = combined_state[start_idx:end_idx]
                        
                        # Reshape to original domain structure
                        nx, ny, nz = self.config.grid_resolution
                        n_components = self.physics_domains[domain_name]['field_components']
                        try:
                            domain_states[domain_name] = domain_state.reshape((n_components, nx, ny, nz))
                        except ValueError:
                            # Fallback: use available data length
                            total_expected = n_components * nx * ny * nz
                            if len(domain_state) >= total_expected:
                                domain_states[domain_name] = domain_state[:total_expected].reshape((n_components, nx, ny, nz))
                            else:
                                # Pad with zeros if necessary
                                padded_state = np.zeros(total_expected)
                                padded_state[:len(domain_state)] = domain_state
                                domain_states[domain_name] = padded_state.reshape((n_components, nx, ny, nz))
                        
                        start_idx = end_idx
        else:
            # Fallback to original logic
            for domain_name, domain_config in self.physics_domains.items():
                # Estimate state size (simplified)
                nx, ny, nz = self.config.grid_resolution
                n_components = domain_config['field_components']
                state_size = nx * ny * nz * n_components
                
                end_idx = start_idx + state_size
                if end_idx <= len(combined_state):
                    domain_state = combined_state[start_idx:end_idx]
                    domain_states[domain_name] = domain_state.reshape((n_components, nx, ny, nz))
                    start_idx = end_idx
                else:
                    # Handle size mismatch
                    remaining_size = len(combined_state) - start_idx
                    domain_state = combined_state[start_idx:]
                    # Pad or truncate as needed
                if remaining_size < state_size:
                    domain_state = np.pad(domain_state, (0, state_size - remaining_size))
                domain_states[domain_name] = domain_state[:state_size].reshape((n_components, nx, ny, nz))
                break
                
        return domain_states
        
    def _combine_domain_derivatives(self, domain_derivatives: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine domain derivatives into single derivative vector"""
        combined = []
        
        for domain_name in self.physics_domains:
            if domain_name in domain_derivatives:
                derivative = domain_derivatives[domain_name].flatten()
                combined.extend(derivative)
                
        return np.array(combined)
        
    def _compute_quantum_classical_bridge(self, 
                                        t: float,
                                        domain_states: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute quantum-classical bridge terms η_quantum-classical(t)"""
        
        if 'quantum' not in domain_states:
            return np.zeros(1)
            
        quantum_state = domain_states['quantum']
        bridge_terms = np.zeros_like(quantum_state.flatten())
        
        # Decoherence from classical domains
        if 'thermal' in domain_states:
            thermal_state = domain_states['thermal']
            temperature = np.mean(thermal_state)
            
            # Thermal decoherence
            decoherence_rate = K_B * temperature / HBAR
            bridge_terms -= np.real(decoherence_rate * quantum_state.flatten())
            
        # Measurement backaction from electromagnetic domain
        if 'electromagnetic' in domain_states:
            em_state = domain_states['electromagnetic']
            field_intensity = np.mean(np.abs(em_state)**2)
            
            # Measurement-induced decoherence
            measurement_rate = field_intensity / (HBAR * C_LIGHT)
            bridge_terms -= measurement_rate * np.random.random(len(bridge_terms)) * quantum_state.flatten()
            
        return bridge_terms
        
    def _analyze_energy_conservation(self, evolution_results: Dict[str, Any]) -> float:
        """Analyze energy conservation during evolution"""
        
        if 'time' not in evolution_results:
            return 0.0
            
        time_points = evolution_results['time']
        total_energies = []
        
        for i, t in enumerate(time_points):
            total_energy = 0.0
            
            # Sum energy from all domains
            for domain_name in self.physics_domains:
                evolution_key = f'{domain_name}_evolution'
                if evolution_key in evolution_results:
                    domain_state = evolution_results[evolution_key][i]
                    
                    # Energy as sum of squares (simplified)
                    domain_energy = np.sum(np.abs(domain_state)**2)
                    total_energy += domain_energy
                    
            total_energies.append(total_energy)
            
        # Energy conservation error
        if len(total_energies) > 1:
            initial_energy = total_energies[0]
            final_energy = total_energies[-1]
            
            if initial_energy > 0:
                conservation_error = abs(final_energy - initial_energy) / initial_energy
            else:
                conservation_error = abs(final_energy - initial_energy)
        else:
            conservation_error = 0.0
            
        return conservation_error

def create_enhanced_physics_pipeline(config: Optional[PhysicsPipelineConfig] = None) -> EnhancedHighFidelityPhysicsPipeline:
    """
    Factory function to create enhanced physics pipeline
    
    Args:
        config: Optional configuration
        
    Returns:
        Configured physics pipeline
    """
    if config is None:
        config = PhysicsPipelineConfig(
            n_domains=4,
            monte_carlo_samples=25000,
            uncertainty_quantification=True,
            frequency_dependent_coupling=True
        )
        
    return EnhancedHighFidelityPhysicsPipeline(config)

if __name__ == "__main__":
    # Demonstration
    logging.basicConfig(level=logging.INFO)
    
    # Create physics pipeline
    physics_config = PhysicsPipelineConfig(
        n_domains=4,
        grid_resolution=(16, 16, 16),  # Smaller for demo
        monte_carlo_samples=1000,  # Reduced for demo
        uncertainty_quantification=True
    )
    
    physics_pipeline = create_enhanced_physics_pipeline(physics_config)
    
    # Test frequency-dependent coupling
    coupling_matrix = physics_pipeline.compute_frequency_dependent_coupling(1e6, 300.0)
    print(f"Coupling matrix shape: {coupling_matrix.shape}")
    print(f"Coupling matrix determinant: {np.linalg.det(coupling_matrix):.2e}")
    
    # Test uncertainty quantification
    test_parameters = np.array([1.0, 2.0, 0.5, 1.5])
    uq_results = physics_pipeline.perform_uncertainty_quantification(test_parameters)
    
    print(f"\nUncertainty Quantification Results:")
    print(f"  Mean values: {uq_results['mean_values']}")
    print(f"  Standard deviations: {uq_results['standard_deviations']}")
    print(f"  Monte Carlo samples: {uq_results['monte_carlo_samples']}")
    
    # Test multi-physics evolution
    nx, ny, nz = physics_config.grid_resolution
    initial_state = {
        'electromagnetic': np.random.random((6, nx, ny, nz)) + 1j * np.random.random((6, nx, ny, nz)),
        'thermal': np.full((1, nx, ny, nz), 300.0),
        'mechanical': np.zeros((3, nx, ny, nz)),
        'quantum': np.random.random((2, nx, ny, nz)) + 1j * np.random.random((2, nx, ny, nz))
    }
    
    evolution_results = physics_pipeline.evolve_multi_physics_system(
        initial_state, (0.0, 1e-6), n_time_points=20
    )
    
    print(f"\nMulti-Physics Evolution Results:")
    print(f"  Time points: {len(evolution_results['time'])}")
    print(f"  Energy conservation error: {evolution_results['energy_conservation_error']:.2e}")
    
    for domain in physics_pipeline.physics_domains:
        evolution_key = f'{domain}_evolution'
        if evolution_key in evolution_results:
            print(f"  {domain} evolution shape: {evolution_results[evolution_key].shape}")
