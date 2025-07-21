from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
import numpy as np
from typing import List, Tuple, Dict, Optional
from openfermion import jordan_wigner, FermionOperator
from ..molecular.parser import MolecularInfo


class HardwareAwareCircuitBuilder:
    
    def __init__(self, backend_properties: Optional[Dict] = None):
        
        self.backend_properties = backend_properties or {}
        self.native_gates = self.backend_properties.get('native_gates', ['rx', 'ry', 'rz', 'cx'])
        self.connectivity = self.backend_properties.get('connectivity', None)
        self.max_circuit_depth = self.backend_properties.get('max_depth', 100)
        
    def build_molecular_circuit(self, 
                               mol_info: MolecularInfo,
                               mapping_type: str = 'jordan_wigner',
                               include_vqe: bool = True) -> QuantumCircuit:
        
        num_qubits = self._calculate_num_qubits(mol_info)
        
        qreg = QuantumRegister(num_qubits, 'q')
        creg = ClassicalRegister(num_qubits, 'c')
        circuit = QuantumCircuit(qreg, creg)
        
        circuit = self._prepare_initial_state(circuit, mol_info)
        
        if include_vqe:
            circuit = self._add_vqe_ansatz(circuit, mol_info)
        
        circuit = self._optimize_for_hardware(circuit)
        
        return circuit
    
    def _calculate_num_qubits(self, mol_info: MolecularInfo) -> int:
        
        spin_orbitals = 2 * mol_info.num_orbitals
        return spin_orbitals
    
    def _prepare_initial_state(self, 
                              circuit: QuantumCircuit, 
                              mol_info: MolecularInfo) -> QuantumCircuit:
        
        num_alpha = (mol_info.num_electrons + mol_info.spin_multiplicity - 1) // 2
        num_beta = mol_info.num_electrons - num_alpha
        
        for i in range(num_alpha):
            circuit.x(2 * i)
        
        for i in range(num_beta):
            circuit.x(2 * i + 1)
            
        return circuit
    
    def _add_vqe_ansatz(self, 
                       circuit: QuantumCircuit, 
                       mol_info: MolecularInfo) -> QuantumCircuit:
        
        num_qubits = circuit.num_qubits
        num_layers = min(3, self.max_circuit_depth // 10)
        
        params = ParameterVector('θ', num_qubits * num_layers * 3)
        param_idx = 0
        
        for layer in range(num_layers):
            for qubit in range(num_qubits):
                circuit.ry(params[param_idx], qubit)
                param_idx += 1
            
            for qubit in range(0, num_qubits - 1, 2):
                if self._check_connectivity(qubit, qubit + 1):
                    circuit.cx(qubit, qubit + 1)
            
            for qubit in range(1, num_qubits - 1, 2):
                if self._check_connectivity(qubit, qubit + 1):
                    circuit.cx(qubit, qubit + 1)
            
            for qubit in range(num_qubits):
                circuit.rz(params[param_idx], qubit)
                param_idx += 1
                circuit.ry(params[param_idx], qubit)
                param_idx += 1
        
        return circuit
    
    def _check_connectivity(self, qubit1: int, qubit2: int) -> bool:
        
        if self.connectivity is None:
            return True
            
        return (qubit1, qubit2) in self.connectivity or (qubit2, qubit1) in self.connectivity
    
    def _optimize_for_hardware(self, circuit: QuantumCircuit) -> QuantumCircuit:
        
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import (
            UnrollCustomDefinitions, Optimize1qGates, CXCancellation,
            CommutativeCancellation, OptimizeSwapBeforeMeasure
        )
        
        pm = PassManager([
            UnrollCustomDefinitions(self.native_gates),
            Optimize1qGates(),
            CXCancellation(),
            CommutativeCancellation(),
            OptimizeSwapBeforeMeasure()
        ])
        
        optimized_circuit = pm.run(circuit)
        
        if optimized_circuit.depth() > self.max_circuit_depth:
            print(f"Warning: Circuit depth ({optimized_circuit.depth()}) exceeds max depth ({self.max_circuit_depth})")
        
        return optimized_circuit
    
    def create_excitation_preserving_ansatz(self, 
                                          mol_info: MolecularInfo,
                                          excitation_type: str = 'UCCSD') -> QuantumCircuit:
        
        num_qubits = self._calculate_num_qubits(mol_info)
        circuit = QuantumCircuit(num_qubits)
        
        circuit = self._prepare_initial_state(circuit, mol_info)
        
        # For spin orbitals: we have 2*num_orbitals spin orbitals total
        # The first num_electrons spin orbitals are occupied
        num_spin_orbitals = 2 * mol_info.num_orbitals
        occupied_indices = list(range(mol_info.num_electrons))
        virtual_indices = list(range(mol_info.num_electrons, num_spin_orbitals))
        
        params = []
        
        if 'S' in excitation_type:
            for i in occupied_indices:
                for a in virtual_indices:
                    param = Parameter(f't_s_{i}_{a}')
                    params.append(param)
                    circuit = self._add_single_excitation(circuit, i, a, param)
        
        if 'D' in excitation_type:
            for i in occupied_indices:
                for j in occupied_indices:
                    if i < j:
                        for a in virtual_indices:
                            for b in virtual_indices:
                                if a < b:
                                    param = Parameter(f't_d_{i}_{j}_{a}_{b}')
                                    params.append(param)
                                    circuit = self._add_double_excitation(circuit, i, j, a, b, param)
        
        return self._optimize_for_hardware(circuit)
    
    def _add_single_excitation(self, 
                              circuit: QuantumCircuit,
                              i: int, a: int, 
                              param: Parameter) -> QuantumCircuit:
        
        # i and a are already spin-orbital indices
        # Single excitation from occupied spin-orbital i to virtual spin-orbital a
        if i < circuit.num_qubits and a < circuit.num_qubits:
            circuit.h(i)
            circuit.h(a)
            circuit.cx(i, a)
            circuit.ry(param, a)
            circuit.cx(i, a)
            circuit.h(i)
            circuit.h(a)
        
        return circuit
    
    def _add_double_excitation(self,
                              circuit: QuantumCircuit,
                              i: int, j: int, a: int, b: int,
                              param: Parameter) -> QuantumCircuit:
        
        # i, j, a, b are already spin-orbital indices
        # Double excitation from occupied spin-orbitals i,j to virtual spin-orbitals a,b
        if (i < circuit.num_qubits and j < circuit.num_qubits and 
            a < circuit.num_qubits and b < circuit.num_qubits):
            circuit.cx(i, j)
            circuit.cx(a, b)
            circuit.h(j)
            circuit.h(b)
            circuit.cx(j, b)
            circuit.ry(param/4, b)
            circuit.cx(j, b)
            circuit.h(j)
            circuit.h(b)
            circuit.cx(i, j)
            circuit.cx(a, b)
        
        return circuit