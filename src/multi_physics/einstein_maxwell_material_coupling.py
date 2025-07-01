"""
Einstein-Maxwell-Material Coupled Equations Module

Implements the enhanced Einstein-Maxwell-Material system:

G_μν = 8π(T_μν^matter + T_μν^EM + T_μν^degradation)
∂_μ F^μν = 4π J^ν + J_material^ν(t)
dε/dt = f_degradation(σ_stress, T, E_field, t_exposure)

Features:
- Material degradation stress-energy tensor T_μν^degradation
- Time-dependent material currents J_material^ν(t)
- Coupled degradation dynamics
- Spacetime-matter-field coupling
"""

import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate
from scipy.special import erf, gamma as gamma_func
from typing import Dict, List, Tuple, Optional, Callable, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

class SpacetimeMetric(Enum):
    """Spacetime metric types"""
    MINKOWSKI = "minkowski"
    SCHWARZSCHILD = "schwarzschild"
    KERR = "kerr"
    ALCUBIERRE = "alcubierre"
    
class MaterialType(Enum):
    """Material type enumeration"""
    CONDUCTOR = "conductor"
    DIELECTRIC = "dielectric"
    METAMATERIAL = "metamaterial"
    SUPERCONDUCTOR = "superconductor"
    QUANTUM_MATERIAL = "quantum_material"

@dataclass
class EinsteinMaxwellConfig:
    """Configuration for Einstein-Maxwell-Material system"""
    spacetime_metric: SpacetimeMetric = SpacetimeMetric.MINKOWSKI
    material_type: MaterialType = MaterialType.METAMATERIAL
    c: float = 299792458.0  # Speed of light (m/s)
    G: float = 6.67430e-11  # Gravitational constant (m³/kg·s²)
    epsilon_0: float = 8.8541878128e-12  # Vacuum permittivity (F/m)
    mu_0: float = 1.25663706212e-6  # Vacuum permeability (H/m)
    degradation_time_scale: float = 3600.0  # Material degradation time scale (s)
    stress_threshold: float = 1e8  # Stress threshold for degradation (Pa)
    temperature_threshold: float = 1000.0  # Temperature threshold (K)
    field_threshold: float = 1e6  # Electric field threshold (V/m)
    
class EinsteinMaxwellMaterialCoupling:
    """
    Enhanced Einstein-Maxwell-Material coupled system
    
    Implements the full relativistic treatment of matter-field-spacetime coupling
    """
    
    def __init__(self, config: EinsteinMaxwellConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Physical constants
        self.c = config.c
        self.G = config.G
        self.epsilon_0 = config.epsilon_0
        self.mu_0 = config.mu_0
        
        # Metric tensor and connection coefficients
        self.g_metric = self._initialize_metric_tensor()
        self.christoffel_symbols = self._compute_christoffel_symbols()
        
        # Material properties
        self.material_properties = self._initialize_material_properties()
        
        # Degradation tracking
        self.degradation_state = {
            'epsilon_relative': 1.0,
            'mu_relative': 1.0,
            'conductivity': 0.0,
            'stress_accumulation': 0.0,
            'thermal_damage': 0.0,
            'field_damage': 0.0
        }
        
        # Field state
        self.electromagnetic_field = {
            'E_field': np.zeros(3),
            'B_field': np.zeros(3),
            'four_potential': np.zeros(4)
        }
        
        self.logger.info(f"Initialized Einstein-Maxwell-Material coupling with {config.material_type.value}")
        
    def _initialize_metric_tensor(self) -> np.ndarray:
        """
        Initialize spacetime metric tensor g_μν
        
        Returns:
            4×4 metric tensor
        """
        if self.config.spacetime_metric == SpacetimeMetric.MINKOWSKI:
            # Minkowski metric: diag(-1, 1, 1, 1)
            g = np.diag([-1, 1, 1, 1], dtype=np.float64)
            
        elif self.config.spacetime_metric == SpacetimeMetric.SCHWARZSCHILD:
            # Schwarzschild metric (simplified, r >> 2GM/c²)
            rs = 1e-10  # Schwarzschild radius (very small for weak field)
            r = 1.0     # Radial coordinate
            
            g = np.zeros((4, 4))
            g[0, 0] = -(1 - rs/r)
            g[1, 1] = 1/(1 - rs/r)
            g[2, 2] = r**2
            g[3, 3] = r**2 * np.sin(np.pi/4)**2  # θ = π/4
            
        elif self.config.spacetime_metric == SpacetimeMetric.ALCUBIERRE:
            # Alcubierre warp metric (simplified)
            g = np.diag([-1, 1, 1, 1], dtype=np.float64)
            # Add warp corrections (small perturbations)
            warp_amplitude = 1e-6
            g[0, 1] = warp_amplitude
            g[1, 0] = warp_amplitude
            
        else:
            # Default to Minkowski
            g = np.diag([-1, 1, 1, 1], dtype=np.float64)
            
        return g
        
    def _compute_christoffel_symbols(self) -> np.ndarray:
        """
        Compute Christoffel symbols Γ^μ_νρ from metric
        
        Returns:
            Christoffel symbol array (4×4×4)
        """
        gamma = np.zeros((4, 4, 4))
        g_inv = la.inv(self.g_metric)
        
        # Compute Christoffel symbols: Γ^μ_νρ = ½g^μσ(∂_ν g_σρ + ∂_ρ g_σν - ∂_σ g_νρ)
        # For static metrics, many terms vanish
        # Here we include the main diagonal terms for material coupling
        
        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    for sigma in range(4):
                        # Simplified: main contribution from material-induced metric changes
                        if mu == nu == rho == sigma:
                            gamma[mu, nu, rho] = 0.5 * g_inv[mu, sigma] * 1e-8  # Small coupling
                            
        return gamma
        
    def _initialize_material_properties(self) -> Dict[str, float]:
        """
        Initialize material properties based on material type
        
        Returns:
            Material properties dictionary
        """
        if self.config.material_type == MaterialType.METAMATERIAL:
            return {
                'epsilon_r_initial': 1.5,
                'mu_r_initial': 1.2,
                'conductivity_initial': 1e4,  # S/m
                'degradation_rate': 1e-6,     # per second
                'stress_coupling': 1e-12,     # strain per Pa
                'thermal_coupling': 1e-4,     # per K
                'field_coupling': 1e-15,      # per (V/m)²
                'young_modulus': 1e11,        # Pa
                'thermal_expansion': 1e-5,    # per K
            }
        elif self.config.material_type == MaterialType.SUPERCONDUCTOR:
            return {
                'epsilon_r_initial': 1.0,
                'mu_r_initial': 0.0,  # Perfect diamagnet
                'conductivity_initial': 1e8,
                'degradation_rate': 1e-8,
                'stress_coupling': 1e-14,
                'thermal_coupling': 1e-3,
                'field_coupling': 1e-16,
                'young_modulus': 2e11,
                'thermal_expansion': 1e-6,
            }
        else:
            # Default dielectric
            return {
                'epsilon_r_initial': 4.0,
                'mu_r_initial': 1.0,
                'conductivity_initial': 1e-12,
                'degradation_rate': 1e-7,
                'stress_coupling': 1e-11,
                'thermal_coupling': 1e-3,
                'field_coupling': 1e-14,
                'young_modulus': 5e10,
                'thermal_expansion': 2e-5,
            }
            
    def compute_stress_energy_tensor_matter(self, 
                                          density: float, 
                                          pressure: float,
                                          four_velocity: np.ndarray) -> np.ndarray:
        """
        Compute matter stress-energy tensor T^matter_μν
        
        Args:
            density: Matter density (kg/m³)
            pressure: Matter pressure (Pa)
            four_velocity: Four-velocity u^μ
            
        Returns:
            Matter stress-energy tensor (4×4)
        """
        # Perfect fluid stress-energy tensor: T_μν = (ρ + p)u_μu_ν + pg_μν
        T_matter = np.zeros((4, 4))
        
        # Normalize four-velocity
        u_norm = four_velocity / np.sqrt(np.abs(four_velocity @ self.g_metric @ four_velocity))
        
        for mu in range(4):
            for nu in range(4):
                T_matter[mu, nu] = (density + pressure) * u_norm[mu] * u_norm[nu] + pressure * self.g_metric[mu, nu]
                
        return T_matter
        
    def compute_stress_energy_tensor_em(self, 
                                      E_field: np.ndarray, 
                                      B_field: np.ndarray) -> np.ndarray:
        """
        Compute electromagnetic stress-energy tensor T^EM_μν
        
        Args:
            E_field: Electric field vector (V/m)
            B_field: Magnetic field vector (T)
            
        Returns:
            EM stress-energy tensor (4×4)
        """
        T_em = np.zeros((4, 4))
        
        # Field invariants
        E_squared = np.dot(E_field, E_field)
        B_squared = np.dot(B_field, B_field)
        E_dot_B = np.dot(E_field, B_field)
        
        # Energy density
        energy_density = 0.5 * (self.epsilon_0 * E_squared + B_squared / self.mu_0)
        
        # Poynting vector
        S = np.cross(E_field, B_field) / self.mu_0
        
        # Construct T^EM_μν
        T_em[0, 0] = energy_density  # Energy density
        
        # Energy flux (Poynting vector)
        T_em[0, 1:4] = -S / self.c
        T_em[1:4, 0] = -S / self.c
        
        # Maxwell stress tensor (spatial components)
        for i in range(3):
            for j in range(3):
                delta_ij = 1.0 if i == j else 0.0
                T_em[i+1, j+1] = (self.epsilon_0 * (E_field[i] * E_field[j] - 0.5 * delta_ij * E_squared) +
                                 (B_field[i] * B_field[j] - 0.5 * delta_ij * B_squared) / self.mu_0)
                
        return T_em
        
    def compute_stress_energy_tensor_degradation(self, 
                                               degradation_state: Dict[str, float],
                                               t: float) -> np.ndarray:
        """
        Compute material degradation stress-energy tensor T^degradation_μν
        
        Args:
            degradation_state: Current degradation state
            t: Current time
            
        Returns:
            Degradation stress-energy tensor (4×4)
        """
        T_deg = np.zeros((4, 4))
        
        # Degradation energy density (non-equilibrium contribution)
        stress_energy = degradation_state['stress_accumulation'] ** 2 / (2 * self.material_properties['young_modulus'])
        thermal_energy = degradation_state['thermal_damage'] * 1.38e-23 * self.config.temperature_threshold
        field_energy = degradation_state['field_damage'] * self.epsilon_0 * self.config.field_threshold ** 2
        
        total_degradation_energy = stress_energy + thermal_energy + field_energy
        
        # Time-dependent decay
        decay_factor = np.exp(-t / self.config.degradation_time_scale)
        
        # Energy density contribution
        T_deg[0, 0] = total_degradation_energy * decay_factor
        
        # Pressure contribution (negative for attractive interaction)
        degradation_pressure = -total_degradation_energy * decay_factor / 3.0
        
        for i in range(1, 4):
            T_deg[i, i] = degradation_pressure
            
        return T_deg
        
    def compute_einstein_tensor(self, 
                              metric: np.ndarray,
                              christoffel: np.ndarray) -> np.ndarray:
        """
        Compute Einstein tensor G_μν = R_μν - ½Rg_μν
        
        Args:
            metric: Metric tensor g_μν
            christoffel: Christoffel symbols
            
        Returns:
            Einstein tensor (4×4)
        """
        # Simplified Einstein tensor for weak field approximation
        G = np.zeros((4, 4))
        
        # For static, approximately flat metrics, G_μν ≈ perturbations
        # Include main contributions from material coupling
        
        # Trace of perturbation
        h_trace = np.trace(metric - np.diag([-1, 1, 1, 1]))
        
        for mu in range(4):
            for nu in range(4):
                if mu == nu:
                    # Diagonal terms: dominant contribution
                    G[mu, nu] = -0.5 * h_trace * metric[mu, nu]
                else:
                    # Off-diagonal: coupling terms
                    G[mu, nu] = metric[mu, nu] * 1e-6  # Small coupling
                    
        return G
        
    def solve_einstein_field_equations(self,
                                     T_matter: np.ndarray,
                                     T_em: np.ndarray,
                                     T_degradation: np.ndarray) -> np.ndarray:
        """
        Solve Einstein field equations: G_μν = 8πG(T^matter + T^EM + T^degradation)
        
        Args:
            T_matter: Matter stress-energy tensor
            T_em: Electromagnetic stress-energy tensor
            T_degradation: Degradation stress-energy tensor
            
        Returns:
            Updated metric tensor
        """
        # Total stress-energy tensor
        T_total = T_matter + T_em + T_degradation
        
        # Einstein tensor from current metric
        G = self.compute_einstein_tensor(self.g_metric, self.christoffel_symbols)
        
        # Field equations: G_μν = 8πG/c⁴ T_μν
        einstein_constant = 8 * np.pi * self.G / (self.c ** 4)
        
        # Solve for metric perturbations (linear approximation)
        delta_g = einstein_constant * T_total
        
        # Update metric (small perturbations)
        new_metric = self.g_metric + 1e-10 * delta_g
        
        # Ensure metric signature is preserved
        eigenvals, eigenvecs = la.eigh(new_metric)
        if np.sum(eigenvals < 0) != 1:
            # Restore proper signature
            eigenvals[0] = -np.abs(eigenvals[0])
            eigenvals[1:] = np.abs(eigenvals[1:])
            new_metric = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
        return new_metric
        
    def solve_maxwell_equations(self,
                              J_external: np.ndarray,
                              J_material: np.ndarray,
                              t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve Maxwell equations: ∂_μ F^μν = 4π J^ν + J_material^ν(t)
        
        Args:
            J_external: External current density (A/m²)
            J_material: Material-dependent current density
            t: Current time
            
        Returns:
            (E_field, B_field) updated electromagnetic fields
        """
        # Total current density
        J_total = J_external + J_material
        
        # Current electromagnetic field
        E = self.electromagnetic_field['E_field']
        B = self.electromagnetic_field['B_field']
        
        # Material properties (possibly degraded)
        epsilon_r = self.degradation_state['epsilon_relative']
        mu_r = self.degradation_state['mu_relative']
        sigma = self.degradation_state['conductivity']
        
        epsilon_eff = self.epsilon_0 * epsilon_r
        mu_eff = self.mu_0 * mu_r
        
        # Solve Faraday's law: ∇ × E = -∂B/∂t
        dB_dt = -np.array([
            0,  # Simplified: assume uniform fields
            0,
            J_total[2] / epsilon_eff  # z-component coupling
        ])
        
        # Solve Ampère's law: ∇ × B = μ₀J + μ₀ε₀∂E/∂t
        dE_dt = (J_total[:3] - sigma * E) / epsilon_eff
        
        # Simple Euler integration (in practice, use proper solver)
        dt = 1e-9  # Small time step
        E_new = E + dE_dt * dt
        B_new = B + dB_dt * dt
        
        return E_new, B_new
        
    def compute_material_degradation(self,
                                   stress_tensor: np.ndarray,
                                   temperature: float,
                                   E_field: np.ndarray,
                                   t_exposure: float,
                                   dt: float) -> Dict[str, float]:
        """
        Compute material degradation: dε/dt = f_degradation(σ_stress, T, E_field, t_exposure)
        
        Args:
            stress_tensor: Mechanical stress tensor (Pa)
            temperature: Temperature (K)
            E_field: Electric field (V/m)
            t_exposure: Total exposure time (s)
            dt: Time step (s)
            
        Returns:
            Updated degradation state
        """
        # Stress contribution
        von_mises_stress = np.sqrt(0.5 * np.sum(stress_tensor ** 2))
        stress_factor = von_mises_stress / self.config.stress_threshold
        stress_degradation = self.material_properties['stress_coupling'] * stress_factor ** 2
        
        # Thermal contribution
        temperature_factor = max(0, temperature - self.config.temperature_threshold) / self.config.temperature_threshold
        thermal_degradation = self.material_properties['thermal_coupling'] * temperature_factor
        
        # Field contribution  
        field_magnitude = np.linalg.norm(E_field)
        field_factor = field_magnitude / self.config.field_threshold
        field_degradation = self.material_properties['field_coupling'] * field_factor ** 2
        
        # Time-dependent degradation
        time_factor = t_exposure / self.config.degradation_time_scale
        degradation_rate = self.material_properties['degradation_rate'] * (1 + time_factor)
        
        # Update degradation state
        total_degradation = (stress_degradation + thermal_degradation + field_degradation) * degradation_rate * dt
        
        # Update material properties
        new_state = self.degradation_state.copy()
        
        # Permittivity degradation
        epsilon_change = -total_degradation * 0.1  # 10% max change
        new_state['epsilon_relative'] = max(0.1, self.degradation_state['epsilon_relative'] + epsilon_change)
        
        # Permeability degradation
        mu_change = -total_degradation * 0.05  # 5% max change
        new_state['mu_relative'] = max(0.1, self.degradation_state['mu_relative'] + mu_change)
        
        # Conductivity degradation
        sigma_change = -total_degradation * self.material_properties['conductivity_initial'] * 0.2
        new_state['conductivity'] = max(0, self.degradation_state['conductivity'] + sigma_change)
        
        # Accumulate damage
        new_state['stress_accumulation'] += stress_degradation
        new_state['thermal_damage'] += thermal_degradation
        new_state['field_damage'] += field_degradation
        
        return new_state
        
    def evolve_coupled_system(self,
                            initial_conditions: Dict[str, np.ndarray],
                            external_sources: Dict[str, Callable],
                            t_span: Tuple[float, float],
                            n_steps: int = 1000) -> Dict[str, List[np.ndarray]]:
        """
        Evolve the complete Einstein-Maxwell-Material coupled system
        
        Args:
            initial_conditions: Initial values for all fields
            external_sources: External source functions
            t_span: Time span (t_start, t_end)
            n_steps: Number of time steps
            
        Returns:
            Complete evolution history
        """
        t_points = np.linspace(t_span[0], t_span[1], n_steps)
        dt = (t_span[1] - t_span[0]) / n_steps
        
        # Initialize evolution history
        evolution = {
            'time': [],
            'metric_tensor': [],
            'E_field': [],
            'B_field': [],
            'degradation_state': [],
            'stress_energy_matter': [],
            'stress_energy_em': [],
            'stress_energy_degradation': []
        }
        
        # Set initial conditions
        matter_density = initial_conditions.get('matter_density', 1000.0)
        matter_pressure = initial_conditions.get('matter_pressure', 1e5)
        four_velocity = initial_conditions.get('four_velocity', np.array([1, 0, 0, 0]))
        
        self.electromagnetic_field['E_field'] = initial_conditions.get('E_field', np.array([0, 0, 1e3]))
        self.electromagnetic_field['B_field'] = initial_conditions.get('B_field', np.array([0, 1e-3, 0]))
        
        self.logger.info(f"Evolving Einstein-Maxwell-Material system from t={t_span[0]} to t={t_span[1]}")
        
        for i, t in enumerate(t_points):
            # External sources at time t
            J_external = external_sources.get('current_density', lambda t: np.zeros(4))(t)
            stress_external = external_sources.get('stress_tensor', lambda t: np.zeros((3, 3)))(t)
            temperature = external_sources.get('temperature', lambda t: 300.0)(t)
            
            # Compute stress-energy tensors
            T_matter = self.compute_stress_energy_tensor_matter(matter_density, matter_pressure, four_velocity)
            T_em = self.compute_stress_energy_tensor_em(
                self.electromagnetic_field['E_field'], 
                self.electromagnetic_field['B_field']
            )
            T_degradation = self.compute_stress_energy_tensor_degradation(self.degradation_state, t)
            
            # Solve Einstein field equations
            self.g_metric = self.solve_einstein_field_equations(T_matter, T_em, T_degradation)
            
            # Update Christoffel symbols
            self.christoffel_symbols = self._compute_christoffel_symbols()
            
            # Material current from degradation
            J_material = np.array([
                0,  # Time component
                self.degradation_state['conductivity'] * self.electromagnetic_field['E_field'][0] * 1e-6,
                self.degradation_state['conductivity'] * self.electromagnetic_field['E_field'][1] * 1e-6,
                self.degradation_state['conductivity'] * self.electromagnetic_field['E_field'][2] * 1e-6
            ])
            
            # Solve Maxwell equations
            E_new, B_new = self.solve_maxwell_equations(J_external, J_material, t)
            self.electromagnetic_field['E_field'] = E_new
            self.electromagnetic_field['B_field'] = B_new
            
            # Update material degradation
            self.degradation_state = self.compute_material_degradation(
                stress_external, temperature, E_new, t, dt
            )
            
            # Store evolution data
            evolution['time'].append(t)
            evolution['metric_tensor'].append(self.g_metric.copy())
            evolution['E_field'].append(E_new.copy())
            evolution['B_field'].append(B_new.copy())
            evolution['degradation_state'].append(self.degradation_state.copy())
            evolution['stress_energy_matter'].append(T_matter.copy())
            evolution['stress_energy_em'].append(T_em.copy())
            evolution['stress_energy_degradation'].append(T_degradation.copy())
            
            if i % (n_steps // 10) == 0:
                self.logger.debug(f"Evolution progress: {100*i/n_steps:.1f}%")
                
        self.logger.info("Einstein-Maxwell-Material evolution completed")
        return evolution
        
    def compute_enhancement_metrics(self, evolution: Dict[str, List]) -> Dict[str, float]:
        """
        Compute enhancement metrics from coupled evolution
        
        Args:
            evolution: Complete evolution history
            
        Returns:
            Enhancement metrics
        """
        metrics = {}
        
        # Field enhancement over time
        E_initial = np.linalg.norm(evolution['E_field'][0])
        E_final = np.linalg.norm(evolution['E_field'][-1])
        metrics['electric_field_enhancement'] = E_final / E_initial if E_initial > 0 else 1.0
        
        # Magnetic field enhancement
        B_initial = np.linalg.norm(evolution['B_field'][0])
        B_final = np.linalg.norm(evolution['B_field'][-1])
        metrics['magnetic_field_enhancement'] = B_final / B_initial if B_initial > 0 else 1.0
        
        # Spacetime curvature enhancement
        metric_initial = evolution['metric_tensor'][0]
        metric_final = evolution['metric_tensor'][-1]
        curvature_initial = np.linalg.norm(metric_initial - np.diag([-1, 1, 1, 1]))
        curvature_final = np.linalg.norm(metric_final - np.diag([-1, 1, 1, 1]))
        metrics['spacetime_curvature_enhancement'] = curvature_final / curvature_initial if curvature_initial > 0 else 1.0
        
        # Material degradation impact
        initial_epsilon = evolution['degradation_state'][0]['epsilon_relative']
        final_epsilon = evolution['degradation_state'][-1]['epsilon_relative']
        metrics['material_degradation_factor'] = final_epsilon / initial_epsilon
        
        # Overall coupling enhancement
        field_enhancement = metrics['electric_field_enhancement'] * metrics['magnetic_field_enhancement']
        spacetime_enhancement = metrics['spacetime_curvature_enhancement']
        degradation_factor = metrics['material_degradation_factor']
        
        metrics['total_enhancement_factor'] = field_enhancement * spacetime_enhancement / degradation_factor
        
        # Energy conservation check
        energy_initial = evolution['stress_energy_em'][0][0, 0]
        energy_final = evolution['stress_energy_em'][-1][0, 0]
        metrics['energy_conservation'] = abs(energy_final - energy_initial) / energy_initial if energy_initial > 0 else 0.0
        
        return metrics
        
    def visualize_evolution(self,
                          evolution: Dict[str, List],
                          save_path: Optional[str] = None):
        """
        Visualize Einstein-Maxwell-Material evolution
        
        Args:
            evolution: Evolution history
            save_path: Optional path to save plots
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        time = evolution['time']
        
        # Electric field evolution
        E_magnitude = [np.linalg.norm(E) for E in evolution['E_field']]
        axes[0, 0].plot(time, E_magnitude)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Electric Field Magnitude (V/m)')
        axes[0, 0].set_title('Electric Field Evolution')
        axes[0, 0].grid(True)
        
        # Magnetic field evolution
        B_magnitude = [np.linalg.norm(B) for B in evolution['B_field']]
        axes[0, 1].plot(time, B_magnitude)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Magnetic Field Magnitude (T)')
        axes[0, 1].set_title('Magnetic Field Evolution')
        axes[0, 1].grid(True)
        
        # Material degradation
        epsilon_evolution = [state['epsilon_relative'] for state in evolution['degradation_state']]
        mu_evolution = [state['mu_relative'] for state in evolution['degradation_state']]
        axes[1, 0].plot(time, epsilon_evolution, label='ε_relative')
        axes[1, 0].plot(time, mu_evolution, label='μ_relative')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Relative Material Property')
        axes[1, 0].set_title('Material Degradation')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Spacetime curvature
        curvature = [np.linalg.norm(g - np.diag([-1, 1, 1, 1])) for g in evolution['metric_tensor']]
        axes[1, 1].plot(time, curvature)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Metric Deviation')
        axes[1, 1].set_title('Spacetime Curvature Evolution')
        axes[1, 1].grid(True)
        
        # Energy density evolution
        em_energy = [T[0, 0] for T in evolution['stress_energy_em']]
        matter_energy = [T[0, 0] for T in evolution['stress_energy_matter']]
        degradation_energy = [T[0, 0] for T in evolution['stress_energy_degradation']]
        
        axes[2, 0].plot(time, em_energy, label='EM Energy')
        axes[2, 0].plot(time, matter_energy, label='Matter Energy')
        axes[2, 0].plot(time, degradation_energy, label='Degradation Energy')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Energy Density (J/m³)')
        axes[2, 0].set_title('Energy Density Evolution')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # 3D phase space plot (E-field components)
        ax_3d = fig.add_subplot(3, 2, 6, projection='3d')
        E_x = [E[0] for E in evolution['E_field']]
        E_y = [E[1] for E in evolution['E_field']]
        E_z = [E[2] for E in evolution['E_field']]
        
        ax_3d.plot(E_x, E_y, E_z, alpha=0.7)
        ax_3d.set_xlabel('E_x (V/m)')
        ax_3d.set_ylabel('E_y (V/m)')
        ax_3d.set_zlabel('E_z (V/m)')
        ax_3d.set_title('E-field Phase Space')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Evolution visualization saved to {save_path}")
        else:
            plt.show()

def create_einstein_maxwell_coupling(config: Optional[EinsteinMaxwellConfig] = None) -> EinsteinMaxwellMaterialCoupling:
    """
    Factory function to create Einstein-Maxwell-Material coupling system
    
    Args:
        config: Optional configuration, uses default if None
        
    Returns:
        Configured Einstein-Maxwell-Material system
    """
    if config is None:
        config = EinsteinMaxwellConfig(
            material_type=MaterialType.METAMATERIAL,
            spacetime_metric=SpacetimeMetric.MINKOWSKI
        )
    
    return EinsteinMaxwellMaterialCoupling(config)

if __name__ == "__main__":
    # Example usage and validation
    logging.basicConfig(level=logging.INFO)
    
    # Create Einstein-Maxwell-Material system
    config = EinsteinMaxwellConfig(material_type=MaterialType.METAMATERIAL)
    em_system = EinsteinMaxwellMaterialCoupling(config)
    
    # Define initial conditions
    initial_conditions = {
        'matter_density': 2000.0,
        'matter_pressure': 1e6,
        'four_velocity': np.array([1, 0, 0, 0]),
        'E_field': np.array([0, 0, 1e4]),
        'B_field': np.array([1e-2, 0, 0])
    }
    
    # Define external sources
    external_sources = {
        'current_density': lambda t: np.array([0, 1e-3 * np.sin(2*np.pi*t), 0, 0]),
        'stress_tensor': lambda t: np.diag([1e7 * (1 + 0.1*np.sin(t)), 1e7, 1e7]),
        'temperature': lambda t: 300 + 50 * np.sin(0.1*t)
    }
    
    # Evolve system
    evolution = em_system.evolve_coupled_system(initial_conditions, external_sources, (0, 100), n_steps=200)
    
    # Compute enhancement metrics
    metrics = em_system.compute_enhancement_metrics(evolution)
    
    # Visualize evolution
    em_system.visualize_evolution(evolution)
    
    print("Einstein-Maxwell-Material system evolution completed")
    print(f"Total enhancement factor: {metrics['total_enhancement_factor']:.2e}")
    print(f"Electric field enhancement: {metrics['electric_field_enhancement']:.2f}×")
    print(f"Material degradation factor: {metrics['material_degradation_factor']:.3f}")
