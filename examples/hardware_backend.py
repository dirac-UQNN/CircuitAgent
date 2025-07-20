#!/usr/bin/env python3

import sys
sys.path.append('..')

from src import QuantumCircuitAgent
from qiskit_ibm_runtime import QiskitRuntimeService


def run_on_hardware():
    # Initialize runtime service (requires IBM Quantum account)
    # service = QiskitRuntimeService()
    # backend = service.backend('ibmq_qasm_simulator')  # or real hardware like 'ibm_kyoto'
    
    # For demo purposes, we'll use a mock backend
    from qiskit.providers.fake_provider import FakeQuito
    backend = FakeQuito()
    
    # Initialize agent with hardware backend
    agent = QuantumCircuitAgent(
        backend=backend,
        optimization_level=3,  # Maximum optimization for hardware
        basis_set='sto-3g'
    )
    
    # Simulate a simple molecule that fits on the hardware
    print("=== Running on Hardware Backend ===")
    print(f"Backend: {backend.name}")
    print(f"Number of qubits: {backend.configuration().n_qubits}")
    print(f"Native gates: {backend.configuration().basis_gates}\n")
    
    # Hydrogen molecule - small enough for most quantum hardware
    result = agent.simulate_molecule(
        molecule_input='[H][H]',
        input_format='smiles',
        charge=0,
        spin_multiplicity=1,
        method='vqe',
        calculate_properties=True
    )
    
    print(f"Ground state energy: {result.energy:.6f} Hartree")
    print(f"Optimized circuit depth: {result.circuit.depth()}")
    print(f"Gate reduction: {result.optimization_stats['gate_reduction']} gates")
    
    # Show circuit optimization stats
    print("\n=== Circuit Optimization Stats ===")
    for key, value in result.optimization_stats.items():
        print(f"{key}: {value}")
    
    # Custom circuit building
    print("\n=== Building Custom Hardware-Efficient Circuit ===")
    from src.molecular.parser import MolecularParser
    parser = MolecularParser()
    mol_info = parser.parse_molecule('[H][H]', 'smiles')
    
    custom_circuit = agent.build_custom_circuit(
        mol_info,
        ansatz_type='hardware_efficient',
        num_layers=2
    )
    
    print(f"Custom circuit depth: {custom_circuit.depth()}")
    print(f"Custom circuit gates: {custom_circuit.count_ops()}")


def demonstrate_active_space():
    """Demonstrate active space reduction for larger molecules"""
    
    agent = QuantumCircuitAgent(optimization_level=2)
    
    print("\n=== Active Space Calculation ===")
    
    # For larger molecules, we can use active space approximation
    # This reduces the number of qubits needed
    from src.molecular.parser import MolecularParser
    from src.quantum.hamiltonian_builder import MolecularHamiltonian
    
    parser = MolecularParser()
    mol_info = parser.parse_molecule('CC', 'smiles')  # Ethane
    
    print(f"Full system: {mol_info.num_orbitals} orbitals, {mol_info.num_electrons} electrons")
    print(f"Would require {2 * mol_info.num_orbitals} qubits for full simulation")
    
    # Build active space Hamiltonian (e.g., 4 electrons in 4 orbitals)
    hamiltonian_builder = MolecularHamiltonian()
    active_hamiltonian = hamiltonian_builder.get_active_space_hamiltonian(
        mol_info,
        n_active_electrons=4,
        n_active_orbitals=4
    )
    
    print(f"\nActive space: 4 electrons in 4 orbitals")
    print(f"Requires only 8 qubits")
    print(f"Number of Pauli terms: {len(active_hamiltonian)}")


if __name__ == '__main__':
    run_on_hardware()
    demonstrate_active_space()