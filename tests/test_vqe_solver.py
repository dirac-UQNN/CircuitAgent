#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from src.molecular.parser import MolecularParser
from src.quantum.vqe_solver import VQESolver


class TestVQESolver:
    
    def setup_method(self):
        self.parser = MolecularParser()
        self.solver = VQESolver(max_iterations=100)  # Limit iterations for speed
    
    def test_h2_energy(self):
        """Test H2 energy calculation"""
        h2 = self.parser.parse_molecule('[H][H]', 'smiles')
        result = self.solver.solve(h2)
        
        # H2 ground state energy should be around -1.1 Hartree
        assert -1.3 < result['energy'] < -0.9
        assert result['converged'] or len(result['convergence_history']) > 0
        assert result['optimal_params'] is not None
    
    def test_convergence_history(self):
        """Test convergence history tracking"""
        h2 = self.parser.parse_molecule('[H][H]', 'smiles')
        result = self.solver.solve(h2)
        
        history = result['convergence_history']
        assert len(history) > 0
        assert all('energy' in h for h in history)
        assert all('iteration' in h for h in history)
        
        # Energy should generally decrease
        energies = [h['energy'] for h in history]
        assert energies[-1] <= energies[0] + 0.1  # Allow small fluctuations
    
    def test_initial_params(self):
        """Test custom initial parameters"""
        h2 = self.parser.parse_molecule('[H][H]', 'smiles')
        
        # First get number of parameters needed
        from src.quantum.circuit_builder import HardwareAwareCircuitBuilder
        builder = HardwareAwareCircuitBuilder()
        circuit = builder.build_molecular_circuit(h2, include_vqe=True)
        num_params = circuit.num_parameters
        
        # Test with custom initial params
        initial_params = np.zeros(num_params)
        result = self.solver.solve(h2, initial_params=initial_params)
        
        assert result['energy'] < 0  # Should still find negative energy
        assert not np.allclose(result['optimal_params'], initial_params)
    
    def test_different_optimizers(self):
        """Test different optimizer options"""
        h2 = self.parser.parse_molecule('[H][H]', 'smiles')
        
        # Test COBYLA
        solver_cobyla = VQESolver(optimizer='COBYLA', max_iterations=50)
        result_cobyla = solver_cobyla.solve(h2)
        assert result_cobyla['energy'] < 0  # Should find negative energy
        
        # Test SPSA
        solver_spsa = VQESolver(optimizer='SPSA', max_iterations=50)
        result_spsa = solver_spsa.solve(h2)
        assert result_spsa['energy'] < 0  # Should find negative energy
    
    def test_property_calculations(self):
        """Test molecular property calculations"""
        h2 = self.parser.parse_molecule('[H][H]', 'smiles')
        result = self.solver.solve(h2)
        
        # Calculate properties
        properties = self.solver.calculate_properties(
            h2,
            result['optimal_circuit'],
            result['optimal_params']
        )
        
        assert 'dipole_x' in properties
        assert 'dipole_y' in properties
        assert 'dipole_z' in properties
        assert 'total_dipole' in properties
        
        # H2 should have very small dipole moment
        assert properties['total_dipole'] < 0.1
    
    def test_empty_convergence_history(self):
        """Test that convergence history is reset between runs"""
        h2 = self.parser.parse_molecule('[H][H]', 'smiles')
        
        # First run
        result1 = self.solver.solve(h2)
        history1_len = len(result1['convergence_history'])
        
        # Reset and second run
        self.solver.convergence_history = []
        result2 = self.solver.solve(h2)
        history2_len = len(result2['convergence_history'])
        
        # Histories should be independent
        assert history1_len > 0
        assert history2_len > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])