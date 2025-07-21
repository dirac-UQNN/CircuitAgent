#!/usr/bin/env python3
"""
Quick integration tests to verify the system works end-to-end
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src import QuantumCircuitAgent


def test_hydrogen_simulation():
    """Test basic H2 simulation works"""
    agent = QuantumCircuitAgent(optimization_level=0)
    
    result = agent.simulate_molecule(
        '[H][H]',
        method='vqe',
        calculate_properties=False
    )
    
    assert result.energy < 0
    assert result.circuit.num_qubits == 4
    assert result.execution_time < 60  # Should be fast


def test_resource_estimation():
    """Test resource estimation doesn't crash"""
    agent = QuantumCircuitAgent()
    
    from src.molecular.parser import MolecularParser
    parser = MolecularParser()
    
    h2o = parser.parse_molecule('O', 'smiles')
    resources = agent.estimate_resources(h2o)
    
    assert resources['num_qubits'] > 0
    assert resources['circuit_depth'] > 0
    assert resources['total_gates'] > 0


def test_different_formats():
    """Test different input formats"""
    agent = QuantumCircuitAgent(optimization_level=0)
    parser = agent.parser
    
    # SMILES
    mol1 = parser.parse_molecule('[H][H]', 'smiles')
    assert len(mol1.atoms) == 2
    
    # XYZ
    xyz = """2
H2 molecule
H 0.0 0.0 0.0
H 0.74 0.0 0.0
"""
    mol2 = parser.parse_molecule(xyz, 'xyz')
    assert len(mol2.atoms) == 2


def test_circuit_optimization_levels():
    """Test different optimization levels don't crash"""
    from src.molecular.parser import MolecularParser
    parser = MolecularParser()
    h2 = parser.parse_molecule('[H][H]', 'smiles')
    
    for level in range(4):
        agent = QuantumCircuitAgent(optimization_level=level)
        circuit = agent.build_custom_circuit(h2, num_layers=1)
        assert circuit.num_qubits == 4


def test_error_handling():
    """Test proper error handling"""
    agent = QuantumCircuitAgent()
    
    # Invalid SMILES
    with pytest.raises(ValueError):
        agent.simulate_molecule('InvalidMolecule', method='vqe')
    
    # Invalid method
    with pytest.raises(ValueError):
        agent.simulate_molecule('[H][H]', method='invalid_method')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])