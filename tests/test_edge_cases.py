#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from src import QuantumCircuitAgent
from src.molecular.parser import MolecularParser


class TestEdgeCases:
    
    def setup_method(self):
        self.agent = QuantumCircuitAgent(optimization_level=0)  # Minimal optimization for speed
        self.parser = MolecularParser()
    
    def test_single_atom(self):
        """Test single atom molecules"""
        # Helium atom
        result = self.agent.simulate_molecule('[He]', method='vqe')
        assert result.energy < 0
        assert result.circuit.num_qubits == 2  # 1 orbital * 2 spins
    
    def test_charged_species(self):
        """Test charged molecules"""
        # H+ ion (0 electrons)
        with pytest.raises(Exception):  # Should fail with 0 electrons
            self.agent.simulate_molecule('[H+]', charge=1, method='vqe')
        
        # H- ion (2 electrons)
        result = self.agent.simulate_molecule('[H-]', charge=-1, method='vqe')
        assert result.energy < 0
    
    def test_radical_species(self):
        """Test radical species with unpaired electrons"""
        # CH3 radical (spin=2)
        result = self.agent.simulate_molecule(
            '[CH3]', 
            spin_multiplicity=2,
            method='vqe'
        )
        assert result.energy < 0
    
    def test_empty_circuit_optimization(self):
        """Test optimization of circuits with no gates"""
        from src.optimization.circuit_optimizer import CircuitOptimizer
        from qiskit import QuantumCircuit
        
        optimizer = CircuitOptimizer()
        empty_circuit = QuantumCircuit(2)
        optimized = optimizer.optimize_circuit(empty_circuit)
        
        assert optimized.num_qubits == 2
        assert optimized.depth() == 0
    
    def test_large_molecule_warning(self):
        """Test handling of molecules too large for simulation"""
        # Benzene - should work but be slow
        benzene = self.parser.parse_molecule('c1ccccc1', 'smiles')
        
        # Check resource estimation
        resources = self.agent.estimate_resources(benzene)
        assert resources['num_qubits'] > 20  # Should need many qubits
        print(f"Benzene would need {resources['num_qubits']} qubits")
    
    def test_invalid_method(self):
        """Test invalid simulation method"""
        with pytest.raises(ValueError):
            self.agent.simulate_molecule('[H][H]', method='invalid_method')
    
    def test_zero_iterations(self):
        """Test VQE with zero iterations"""
        from src.quantum.vqe_solver import VQESolver
        
        solver = VQESolver(max_iterations=0)
        h2 = self.parser.parse_molecule('[H][H]', 'smiles')
        
        # Should still return a result, just not optimized
        result = solver.solve(h2)
        assert 'energy' in result
    
    def test_circuit_with_no_parameters(self):
        """Test circuit without VQE parameters"""
        h2 = self.parser.parse_molecule('[H][H]', 'smiles')
        from src.quantum.circuit_builder import HardwareAwareCircuitBuilder
        
        builder = HardwareAwareCircuitBuilder()
        circuit = builder.build_molecular_circuit(h2, include_vqe=False)
        
        assert circuit.num_parameters == 0
        
        # Should still be able to evaluate energy (just not optimize)
        from src.quantum.hamiltonian_builder import MolecularHamiltonian
        ham_builder = MolecularHamiltonian()
        hamiltonian = ham_builder.build_hamiltonian(h2)
        
        assert hamiltonian.num_qubits == circuit.num_qubits
    
    def test_unsupported_format(self):
        """Test unsupported molecular format"""
        with pytest.raises(ValueError):
            self.parser.parse_molecule('H2O', 'unsupported_format')
    
    def test_malformed_xyz(self):
        """Test malformed XYZ format"""
        bad_xyz = """2
        Missing coordinates
        H 0.0 0.0
        """
        with pytest.raises(Exception):
            self.parser.parse_molecule(bad_xyz, 'xyz')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])