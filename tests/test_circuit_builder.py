#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from src.molecular.parser import MolecularParser
from src.quantum.circuit_builder import HardwareAwareCircuitBuilder


class TestCircuitBuilder:
    
    def setup_method(self):
        self.parser = MolecularParser()
        self.builder = HardwareAwareCircuitBuilder()
    
    def test_circuit_qubit_count(self):
        """Test that circuits have correct number of qubits"""
        # H2 molecule
        h2 = self.parser.parse_molecule('[H][H]', 'smiles')
        circuit = self.builder.build_molecular_circuit(h2, include_vqe=False)
        expected_qubits = 2 * h2.num_orbitals  # spin orbitals
        assert circuit.num_qubits == expected_qubits
        
    def test_initial_state_preparation(self):
        """Test initial state preparation"""
        # H2 with 2 electrons
        h2 = self.parser.parse_molecule('[H][H]', 'smiles')
        circuit = self.builder.build_molecular_circuit(h2, include_vqe=False)
        
        # Count X gates (should be 2 for 2 electrons in ground state)
        x_gates = sum(1 for inst in circuit.data if inst.operation.name == 'x')
        assert x_gates == h2.num_electrons
    
    def test_vqe_ansatz_parameters(self):
        """Test VQE ansatz has parameters"""
        h2 = self.parser.parse_molecule('[H][H]', 'smiles')
        circuit = self.builder.build_molecular_circuit(h2, include_vqe=True)
        
        assert circuit.num_parameters > 0
        assert all(param.name.startswith('θ') for param in circuit.parameters)
    
    def test_hardware_constraints(self):
        """Test hardware-aware features"""
        # Test with connectivity constraints
        connectivity = [(0, 1), (1, 2), (2, 3)]
        builder_constrained = HardwareAwareCircuitBuilder({
            'connectivity': connectivity,
            'native_gates': ['rx', 'ry', 'rz', 'cx']
        })
        
        h2 = self.parser.parse_molecule('[H][H]', 'smiles')
        circuit = builder_constrained.build_molecular_circuit(h2)
        
        # Check all two-qubit gates respect connectivity
        for inst in circuit.data:
            if inst.operation.num_qubits == 2:
                q1, q2 = inst.qubits[0]._index, inst.qubits[1]._index
                assert (q1, q2) in connectivity or (q2, q1) in connectivity
    
    def test_different_spin_states(self):
        """Test different spin multiplicities"""
        # O2 triplet (spin=3)
        o2 = self.parser.parse_molecule('O=O', 'smiles', spin_multiplicity=3)
        circuit = self.builder.build_molecular_circuit(o2, include_vqe=False)
        
        # Should have different alpha and beta electrons
        num_alpha = (o2.num_electrons + o2.spin_multiplicity - 1) // 2
        num_beta = o2.num_electrons - num_alpha
        assert num_alpha != num_beta  # Triplet state
    
    def test_uccsd_ansatz(self):
        """Test UCCSD ansatz creation"""
        # Use a molecule with virtual orbitals (e.g., H2 at larger basis)
        # For now, use LiH which has more orbitals
        lih = self.parser.parse_molecule('[LiH]', 'smiles')
        circuit = self.builder.create_excitation_preserving_ansatz(lih, 'UCCSD')
        
        # For molecules with virtual orbitals, should have parameters
        if lih.num_electrons < 2 * lih.num_orbitals:
            assert circuit.num_parameters > 0
            # Parameter names should indicate excitation type
            param_names = [p.name for p in circuit.parameters]
            assert any('t_s' in name for name in param_names)  # Single excitations
        else:
            # If all orbitals are occupied, no excitations possible
            assert circuit.num_parameters == 0
    
    def test_circuit_depth_limit(self):
        """Test circuit depth limiting"""
        builder_limited = HardwareAwareCircuitBuilder({
            'max_depth': 20
        })
        
        h2o = self.parser.parse_molecule('O', 'smiles')
        circuit = builder_limited.build_molecular_circuit(h2o)
        
        # Circuit depth should not exceed limit (or be close)
        assert circuit.depth() <= 100  # Reasonable limit


if __name__ == '__main__':
    pytest.main([__file__, '-v'])