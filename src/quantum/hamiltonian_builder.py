import numpy as np
from qiskit.quantum_info import SparsePauliOp
from openfermion import (
    MolecularData, jordan_wigner, get_fermion_operator,
    FermionOperator
)
from openfermionpyscf import run_pyscf
from typing import Dict, List, Tuple
from ..molecular.parser import MolecularInfo


class MolecularHamiltonian:
    
    def __init__(self, basis: str = 'sto-3g'):
        
        self.basis = basis
        self.molecular_data = None
        
    def build_hamiltonian(self, mol_info: MolecularInfo) -> SparsePauliOp:
        
        geometry = [(atom, tuple(coord)) 
                   for atom, coord in zip(mol_info.atoms, mol_info.coordinates)]
        
        self.molecular_data = MolecularData(
            geometry=geometry,
            basis=self.basis,
            charge=mol_info.charge,
            multiplicity=mol_info.spin_multiplicity
        )
        
        self.molecular_data = run_pyscf(self.molecular_data)
        
        molecular_hamiltonian = get_fermion_operator(
            self.molecular_data.get_molecular_hamiltonian()
        )
        
        qubit_hamiltonian = jordan_wigner(molecular_hamiltonian)
        
        pauli_op = self._fermion_to_pauli(qubit_hamiltonian)
        
        return pauli_op
    
    def _fermion_to_pauli(self, operator) -> SparsePauliOp:
        
        pauli_list = []
        num_qubits = 2 * self.molecular_data.n_orbitals
        
        for term, coeff in operator.terms.items():
            if abs(coeff) < 1e-10:
                continue
                
            pauli_string = ['I'] * num_qubits
            
            for op_idx, op_type in term:
                if op_type == '+':
                    pauli_string[op_idx] = 'X'
                elif op_type == '-':
                    pauli_string[op_idx] = 'Y'
                else:
                    pauli_string[op_idx] = 'Z'
            
            pauli_list.append((''.join(pauli_string), coeff))
        
        if not pauli_list:
            # Return identity operator if no terms
            pauli_list = [('I' * num_qubits, 0.0)]
        
        return SparsePauliOp.from_list(pauli_list)
    
    def build_dipole_operators(self, mol_info: MolecularInfo) -> Dict[str, SparsePauliOp]:
        
        if self.molecular_data is None:
            _ = self.build_hamiltonian(mol_info)
        
        dipole_operators = {}
        
        for axis_idx, axis in enumerate(['x', 'y', 'z']):
            dipole_integrals = self._compute_dipole_integrals(mol_info, axis_idx)
            
            dipole_op = FermionOperator()
            n_orbitals = self.molecular_data.n_orbitals
            
            for i in range(n_orbitals):
                for j in range(n_orbitals):
                    integral = dipole_integrals[i, j]
                    if abs(integral) > 1e-10:
                        for spin in [0, 1]:
                            p = 2 * i + spin
                            q = 2 * j + spin
                            dipole_op += FermionOperator(f'{p}^ {q}', integral)
            
            qubit_dipole = jordan_wigner(dipole_op)
            dipole_operators[axis] = self._fermion_to_pauli(qubit_dipole)
        
        return dipole_operators
    
    def _compute_dipole_integrals(self, mol_info: MolecularInfo, axis: int) -> np.ndarray:
        
        n_orbitals = self.molecular_data.n_orbitals
        dipole_integrals = np.zeros((n_orbitals, n_orbitals))
        
        mo_coeff = self.molecular_data.canonical_orbitals
        
        for i in range(n_orbitals):
            for j in range(n_orbitals):
                integral = 0.0
                for mu in range(n_orbitals):
                    for nu in range(n_orbitals):
                        ao_integral = self._compute_ao_dipole(mu, nu, mol_info, axis)
                        integral += mo_coeff[mu, i] * ao_integral * mo_coeff[nu, j]
                dipole_integrals[i, j] = integral
        
        return dipole_integrals
    
    def _compute_ao_dipole(self, mu: int, nu: int, mol_info: MolecularInfo, axis: int) -> float:
        # This is a simplified dipole integral calculation
        # In reality, this would require proper AO basis function integrals
        # For now, return a small value to avoid errors
        if mu == nu and mu < len(mol_info.atoms):
            # Rough approximation using nuclear positions
            return 0.1 * mol_info.coordinates[mu, axis]
        else:
            return 0.0
    
    def get_active_space_hamiltonian(self, 
                                   mol_info: MolecularInfo,
                                   n_active_electrons: int,
                                   n_active_orbitals: int) -> SparsePauliOp:
        
        if self.molecular_data is None:
            _ = self.build_hamiltonian(mol_info)
        
        occupied_indices = list(range(mol_info.num_electrons // 2))
        active_indices = occupied_indices[-n_active_electrons//2:] + \
                        list(range(mol_info.num_electrons // 2, 
                                  mol_info.num_electrons // 2 + n_active_orbitals - n_active_electrons // 2))
        
        molecular_hamiltonian = self.molecular_data.get_molecular_hamiltonian()
        
        one_body_integrals = molecular_hamiltonian.one_body_tensor
        two_body_integrals = molecular_hamiltonian.two_body_tensor
        
        active_one_body = one_body_integrals[np.ix_(active_indices, active_indices)]
        active_two_body = two_body_integrals[np.ix_(active_indices, active_indices, 
                                                    active_indices, active_indices)]
        
        core_energy = molecular_hamiltonian.constant
        for i in occupied_indices:
            if i not in active_indices:
                core_energy += 2 * one_body_integrals[i, i]
                for j in occupied_indices:
                    if j not in active_indices:
                        core_energy += 2 * two_body_integrals[i, j, j, i] - \
                                     two_body_integrals[i, j, i, j]
        
        active_fermion_op = FermionOperator('', core_energy)
        
        for p in range(len(active_indices)):
            for q in range(len(active_indices)):
                if abs(active_one_body[p, q]) > 1e-10:
                    for spin in [0, 1]:
                        i = 2 * p + spin
                        j = 2 * q + spin
                        active_fermion_op += FermionOperator(f'{i}^ {j}', active_one_body[p, q])
        
        for p in range(len(active_indices)):
            for q in range(len(active_indices)):
                for r in range(len(active_indices)):
                    for s in range(len(active_indices)):
                        integral = active_two_body[p, q, r, s]
                        if abs(integral) > 1e-10:
                            for spin_i in [0, 1]:
                                for spin_j in [0, 1]:
                                    i = 2 * p + spin_i
                                    j = 2 * q + spin_j
                                    k = 2 * r + spin_j
                                    l = 2 * s + spin_i
                                    active_fermion_op += FermionOperator(
                                        f'{i}^ {j}^ {k} {l}', 
                                        0.5 * integral
                                    )
        
        qubit_hamiltonian = jordan_wigner(active_fermion_op)
        
        return self._fermion_to_pauli(qubit_hamiltonian)