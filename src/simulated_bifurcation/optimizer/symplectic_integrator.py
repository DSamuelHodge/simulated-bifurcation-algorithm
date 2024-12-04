import torch
from typing import Tuple, Any, Union
import numpy as np

class SymplecticIntegrator:
    """
    Enhanced Symplectic Integrator that preserves weight structure during optimization
    while following Hamiltonian quantum mechanics equations.
    """
    def __init__(
        self,
        shape: Tuple[int, int],
        original_weights: torch.Tensor,
        activation_function: Any,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        damping_factor: float = 0.99,
        boundary_softness: float = 2.0
    ):
        # Initialize base oscillator components
        self.position = self.__init_oscillator(shape, dtype, device)
        self.momentum = self.__init_oscillator(shape, dtype, device)
        self.activation_function = activation_function
        
        # Weight preservation components
        self.original_weights = torch.tensor(original_weights, dtype=dtype, device=device)
        self.weight_scales = torch.abs(self.original_weights)
        self.weight_signs = torch.sign(self.original_weights)
        
        # Hyperparameters
        self.damping_factor = damping_factor
        self.boundary_softness = boundary_softness
        
        # Initialize tracking metrics
        self.energy_history = []
        self.position_history = []
        
    @staticmethod
    def __init_oscillator(
        shape: Tuple[int, int], dtype: torch.dtype, device: Union[str, torch.device]
    ) -> torch.Tensor:
        """Initialize oscillator with scaled random values."""
        return 0.1 * (2 * torch.rand(size=shape, device=device, dtype=dtype) - 1)
    
    def position_update(self, coefficient: float) -> None:
        """Update position based on momentum with memory term."""
        memory_term = 0.1 * (self.original_weights - self.position)
        torch.add(
            self.position,
            self.momentum + memory_term,
            alpha=coefficient,
            out=self.position
        )
    
    def momentum_update(self, coefficient: float) -> None:
        """Update momentum with damping."""
        torch.add(
            self.momentum * self.damping_factor,  # Apply damping
            self.position,
            alpha=coefficient,
            out=self.momentum
        )
    
    def quadratic_momentum_update(
        self, coefficient: float, matrix: torch.Tensor
    ) -> None:
        """Update momentum using quadratic term with sign preservation bias."""
        # Calculate sign-preserving bias
        sign_bias = 0.1 * self.weight_signs
        
        # Update momentum with sign preservation
        self.momentum = torch.addmm(
            self.momentum + sign_bias,
            matrix,
            self.activation_function(self.position),
            alpha=coefficient,
        )
    
    def simulate_soft_walls(self) -> None:
        """Implement soft boundaries based on original weight scales."""
        # Calculate scaled boundaries
        bounds = torch.clamp(self.weight_scales, min=0.1)
        
        # Soft boundary function using sigmoid
        scale = torch.sigmoid(
            self.boundary_softness * (torch.abs(self.position) / bounds - 1)
        )
        
        # Apply soft boundaries to momentum and position
        self.momentum *= (1 - scale)
        self.position *= torch.exp(-scale)  # Smooth position scaling
        
        # Ensure numerical stability
        torch.clip(self.position, -bounds * 1.2, bounds * 1.2, out=self.position)
    
    def compute_energy(self, matrix: torch.Tensor) -> float:
        """Compute current system energy including weight preservation term."""
        ising_energy = -0.5 * torch.sum(
            self.position @ matrix @ self.activation_function(self.position)
        )
        preservation_energy = torch.sum(
            torch.abs(self.position - self.original_weights)
        )
        return ising_energy.item() + 0.1 * preservation_energy.item()
    
    def step(
        self,
        momentum_coefficient: float,
        position_coefficient: float,
        quadratic_coefficient: float,
        matrix: torch.Tensor,
    ) -> None:
        """Perform one complete integration step."""
        # Standard updates
        self.momentum_update(momentum_coefficient)
        self.position_update(position_coefficient)
        self.quadratic_momentum_update(quadratic_coefficient, matrix)
        
        # Apply soft boundaries
        self.simulate_soft_walls()
        
        # Track system state
        self.energy_history.append(self.compute_energy(matrix))
        self.position_history.append(self.position.clone().cpu())
    
    def sample_spins(self) -> torch.Tensor:
        """Sample spins while preserving weight structure."""
        # Compute normalized positions
        normalized_pos = self.position / self.weight_scales.clamp(min=1e-6)
        
        # Generate spins with magnitude preservation
        spins = torch.sign(normalized_pos) * self.weight_scales
        
        # Apply smooth transition near zero
        small_weights_mask = self.weight_scales < 0.1
        spins[small_weights_mask] = self.position[small_weights_mask]
        
        return spins
    
    def get_optimization_metrics(self) -> dict:
        """Return metrics about the optimization process."""
        return {
            'energy_history': self.energy_history,
            'position_history': self.position_history,
            'final_position': self.position.clone(),
            'weight_preservation': torch.mean(
                torch.sign(self.position) == torch.sign(self.original_weights)
            ).item()
        }
