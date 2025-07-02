"""
Enhanced Correlation Matrix Module for Digital Twin Framework

Implements advanced 5×5 UQ correlation matrix for:
- Permittivity correlations
- Permeability correlations  
- Thickness correlations
- Temperature correlations
- Frequency correlations

Provides validated correlation structure for enhanced digital twin coherence.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy.linalg import cholesky, LinAlgError

@dataclass
class CorrelationMatrixConfig:
    """Configuration for enhanced correlation matrix"""
    enable_cross_domain_correlation: bool = True
    correlation_strength: float = 1.0
    temperature_dependence: bool = True
    frequency_dependence: bool = True
    noise_floor: float = 1e-12

class EnhancedCorrelationMatrix:
    """
    Enhanced correlation matrix implementation for digital twin framework
    
    Implements validated 5×5 correlation structure:
    - Permittivity (ε') correlations
    - Permeability (μ') correlations
    - Thickness (d) correlations
    - Temperature (T) correlations
    - Frequency (ω) correlations
    """
    
    def __init__(self, config: CorrelationMatrixConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize base correlation matrix (validated structure)
        self.base_correlation_matrix = self._initialize_base_correlation_matrix()
        
        # Initialize domain mapping
        self.domain_indices = {
            'permittivity': 0,
            'permeability': 1, 
            'thickness': 2,
            'temperature': 3,
            'frequency': 4
        }
        
        # Validate matrix properties
        self._validate_correlation_matrix()
        
        self.logger.info("Initialized enhanced correlation matrix with 5×5 structure")
        
    def _initialize_base_correlation_matrix(self) -> np.ndarray:
        """
        Initialize validated 5×5 correlation matrix
        
        Returns:
            Validated correlation matrix
        """
        # Validated correlation structure from casimir-nanopositioning-platform
        correlation_matrix = np.array([
            [1.0,   0.85, 0.72, 0.63, 0.54],  # Permittivity correlations
            [0.85,  1.0,  0.78, 0.69, 0.58],  # Permeability correlations
            [0.72,  0.78, 1.0,  0.82, 0.71],  # Thickness correlations
            [0.63,  0.69, 0.82, 1.0,  0.89],  # Temperature correlations
            [0.54,  0.58, 0.71, 0.89, 1.0]    # Frequency correlations
        ])
        
        # Apply correlation strength scaling
        if self.config.correlation_strength != 1.0:
            # Scale off-diagonal elements
            correlation_matrix = self._scale_correlation_matrix(
                correlation_matrix, self.config.correlation_strength
            )
        
        return correlation_matrix
        
    def _scale_correlation_matrix(self, matrix: np.ndarray, scale: float) -> np.ndarray:
        """
        Scale correlation matrix while preserving positive definiteness
        
        Args:
            matrix: Input correlation matrix
            scale: Scaling factor for off-diagonal elements
            
        Returns:
            Scaled correlation matrix
        """
        scaled_matrix = matrix.copy()
        
        # Scale off-diagonal elements
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i != j:
                    scaled_matrix[i, j] = matrix[i, j] * scale
                    
        # Ensure positive definiteness
        eigenvals = np.linalg.eigvals(scaled_matrix)
        if np.any(eigenvals <= 0):
            self.logger.warning("Scaled matrix not positive definite, applying regularization")
            scaled_matrix += np.eye(matrix.shape[0]) * 1e-6
            
        return scaled_matrix
        
    def _validate_correlation_matrix(self) -> bool:
        """
        Validate correlation matrix properties
        
        Returns:
            True if matrix is valid
        """
        matrix = self.base_correlation_matrix
        
        # Check symmetry
        if not np.allclose(matrix, matrix.T):
            raise ValueError("Correlation matrix must be symmetric")
            
        # Check diagonal elements are 1
        if not np.allclose(np.diag(matrix), 1.0):
            raise ValueError("Diagonal elements of correlation matrix must be 1")
            
        # Check positive definiteness
        eigenvals = np.linalg.eigvals(matrix)
        if np.any(eigenvals <= 0):
            raise ValueError("Correlation matrix must be positive definite")
            
        # Check correlation bounds [-1, 1]
        if np.any(np.abs(matrix) > 1.0):
            raise ValueError("Correlation coefficients must be in [-1, 1]")
            
        self.logger.info("Correlation matrix validation passed")
        return True
        
    def get_correlation_matrix(self, 
                             temperature: Optional[float] = None,
                             frequency: Optional[float] = None) -> np.ndarray:
        """
        Get correlation matrix with optional temperature and frequency dependence
        
        Args:
            temperature: Temperature for temperature-dependent correlations
            frequency: Frequency for frequency-dependent correlations
            
        Returns:
            Correlation matrix
        """
        matrix = self.base_correlation_matrix.copy()
        
        # Apply temperature dependence
        if self.config.temperature_dependence and temperature is not None:
            temp_factor = self._compute_temperature_factor(temperature)
            matrix = self._apply_temperature_dependence(matrix, temp_factor)
            
        # Apply frequency dependence
        if self.config.frequency_dependence and frequency is not None:
            freq_factor = self._compute_frequency_factor(frequency)
            matrix = self._apply_frequency_dependence(matrix, freq_factor)
            
        return matrix
        
    def _compute_temperature_factor(self, temperature: float) -> float:
        """
        Compute temperature dependence factor
        
        Args:
            temperature: Temperature in Kelvin
            
        Returns:
            Temperature factor
        """
        # Reference temperature (room temperature)
        T_ref = 300.0  # K
        
        # Exponential temperature dependence
        temp_factor = np.exp(-(temperature - T_ref) / (2 * T_ref))
        
        return np.clip(temp_factor, 0.1, 2.0)  # Reasonable bounds
        
    def _compute_frequency_factor(self, frequency: float) -> float:
        """
        Compute frequency dependence factor
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            Frequency factor
        """
        # Reference frequency (1 MHz)
        f_ref = 1e6  # Hz
        
        # Logarithmic frequency dependence
        freq_factor = 1.0 + 0.1 * np.log10(frequency / f_ref)
        
        return np.clip(freq_factor, 0.5, 1.5)  # Reasonable bounds
        
    def _apply_temperature_dependence(self, matrix: np.ndarray, temp_factor: float) -> np.ndarray:
        """
        Apply temperature dependence to correlation matrix
        
        Args:
            matrix: Base correlation matrix
            temp_factor: Temperature factor
            
        Returns:
            Temperature-dependent correlation matrix
        """
        temp_matrix = matrix.copy()
        
        # Apply stronger temperature effect to temperature-related correlations
        temp_idx = self.domain_indices['temperature']
        for i in range(matrix.shape[0]):
            if i != temp_idx:
                temp_matrix[i, temp_idx] *= temp_factor
                temp_matrix[temp_idx, i] *= temp_factor
                
        return temp_matrix
        
    def _apply_frequency_dependence(self, matrix: np.ndarray, freq_factor: float) -> np.ndarray:
        """
        Apply frequency dependence to correlation matrix
        
        Args:
            matrix: Base correlation matrix
            freq_factor: Frequency factor
            
        Returns:
            Frequency-dependent correlation matrix
        """
        freq_matrix = matrix.copy()
        
        # Apply stronger frequency effect to frequency-related correlations
        freq_idx = self.domain_indices['frequency']
        for i in range(matrix.shape[0]):
            if i != freq_idx:
                freq_matrix[i, freq_idx] *= freq_factor
                freq_matrix[freq_idx, i] *= freq_factor
                
        return freq_matrix
        
    def generate_correlated_samples(self, 
                                  n_samples: int,
                                  mean: Optional[np.ndarray] = None,
                                  std: Optional[np.ndarray] = None,
                                  temperature: Optional[float] = None,
                                  frequency: Optional[float] = None) -> np.ndarray:
        """
        Generate correlated samples using the correlation matrix
        
        Args:
            n_samples: Number of samples to generate
            mean: Mean values for each domain (defaults to zeros)
            std: Standard deviations for each domain (defaults to ones)
            temperature: Temperature for correlation dependence
            frequency: Frequency for correlation dependence
            
        Returns:
            Array of correlated samples [n_samples, 5]
        """
        n_dims = self.base_correlation_matrix.shape[0]
        
        if mean is None:
            mean = np.zeros(n_dims)
        if std is None:
            std = np.ones(n_dims)
            
        # Get correlation matrix
        corr_matrix = self.get_correlation_matrix(temperature, frequency)
        
        # Convert to covariance matrix
        std_matrix = np.outer(std, std)
        cov_matrix = corr_matrix * std_matrix
        
        # Generate correlated samples using Cholesky decomposition
        try:
            L = cholesky(cov_matrix, lower=True)
        except LinAlgError:
            self.logger.warning("Cholesky decomposition failed, using eigenvalue decomposition")
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            eigenvals = np.maximum(eigenvals, self.config.noise_floor)
            L = eigenvecs @ np.diag(np.sqrt(eigenvals))
            
        # Generate independent samples
        independent_samples = np.random.normal(0, 1, (n_samples, n_dims))
        
        # Apply correlation structure
        correlated_samples = independent_samples @ L.T + mean[np.newaxis, :]
        
        return correlated_samples
        
    def compute_correlation_strength(self, 
                                   domain1: str, 
                                   domain2: str,
                                   temperature: Optional[float] = None,
                                   frequency: Optional[float] = None) -> float:
        """
        Compute correlation strength between two domains
        
        Args:
            domain1: First domain name
            domain2: Second domain name
            temperature: Temperature for correlation dependence
            frequency: Frequency for correlation dependence
            
        Returns:
            Correlation coefficient
        """
        if domain1 not in self.domain_indices or domain2 not in self.domain_indices:
            raise ValueError(f"Unknown domain. Available domains: {list(self.domain_indices.keys())}")
            
        idx1 = self.domain_indices[domain1]
        idx2 = self.domain_indices[domain2]
        
        corr_matrix = self.get_correlation_matrix(temperature, frequency)
        
        return corr_matrix[idx1, idx2]
        
    def analyze_correlation_structure(self) -> Dict[str, float]:
        """
        Analyze correlation matrix structure and properties
        
        Returns:
            Dictionary of correlation analysis metrics
        """
        matrix = self.base_correlation_matrix
        
        analysis = {}
        
        # Eigenvalue analysis
        eigenvals = np.linalg.eigvals(matrix)
        analysis['min_eigenvalue'] = np.min(eigenvals)
        analysis['max_eigenvalue'] = np.max(eigenvals)
        analysis['condition_number'] = np.max(eigenvals) / np.min(eigenvals)
        
        # Correlation strength metrics
        off_diagonal = matrix[np.triu_indices_from(matrix, k=1)]
        analysis['mean_correlation'] = np.mean(off_diagonal)
        analysis['max_correlation'] = np.max(off_diagonal)
        analysis['min_correlation'] = np.min(off_diagonal)
        analysis['correlation_variance'] = np.var(off_diagonal)
        
        # Determinant (measure of multicollinearity)
        analysis['determinant'] = np.linalg.det(matrix)
        
        # Frobenius norm
        analysis['frobenius_norm'] = np.linalg.norm(matrix, 'fro')
        
        self.logger.info(f"Correlation analysis: {analysis}")
        return analysis
        
    def visualize_correlation_matrix(self, save_path: Optional[str] = None):
        """
        Visualize correlation matrix structure
        
        Args:
            save_path: Optional path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap
            sns.heatmap(
                self.base_correlation_matrix,
                annot=True,
                cmap='RdBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                xticklabels=list(self.domain_indices.keys()),
                yticklabels=list(self.domain_indices.keys()),
                ax=ax
            )
            
            ax.set_title('Enhanced 5×5 Correlation Matrix')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Correlation matrix visualization saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("Matplotlib/Seaborn not available for visualization")

def create_enhanced_correlation_matrix(config: Optional[CorrelationMatrixConfig] = None) -> EnhancedCorrelationMatrix:
    """
    Factory function to create enhanced correlation matrix
    
    Args:
        config: Optional configuration, uses default if None
        
    Returns:
        Configured enhanced correlation matrix
    """
    if config is None:
        config = CorrelationMatrixConfig()
        
    return EnhancedCorrelationMatrix(config)

if __name__ == "__main__":
    # Example usage and validation
    logging.basicConfig(level=logging.INFO)
    
    # Create enhanced correlation matrix
    config = CorrelationMatrixConfig(
        correlation_strength=1.0,
        temperature_dependence=True,
        frequency_dependence=True
    )
    corr_matrix = EnhancedCorrelationMatrix(config)
    
    # Analyze correlation structure
    analysis = corr_matrix.analyze_correlation_structure()
    print(f"Correlation matrix analysis: {analysis}")
    
    # Generate correlated samples
    samples = corr_matrix.generate_correlated_samples(
        n_samples=1000,
        temperature=300.0,
        frequency=1e6
    )
    print(f"Generated {samples.shape[0]} correlated samples")
    
    # Test correlation strength computation
    corr_strength = corr_matrix.compute_correlation_strength(
        'permittivity', 'permeability',
        temperature=300.0, frequency=1e6
    )
    print(f"Permittivity-Permeability correlation: {corr_strength:.3f}")
    
    # Visualize correlation matrix
    corr_matrix.visualize_correlation_matrix()
