#!/usr/bin/env python3

import sys
import os
# Add the parent directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.quantum_circuit_agent import QuantumCircuitAgent


def main():
    # Initialize the quantum circuit agent
    agent = QuantumCircuitAgent(
        backend=None,  # Use simulator
        optimization_level=2,
        basis_set='sto-3g'
    )
    
    # Example 1: Water molecule (H2O)
    print("=== Simulating Water (H2O) ===")
    result_h2o = agent.simulate_molecule(
        molecule_input='O',  # Water SMILES
        input_format='smiles',
        charge=0,
        spin_multiplicity=1,
        method='vqe',
        calculate_properties=True
    )
    
    print(f"Energy: {result_h2o.energy:.6f} Hartree")
    print(f"Dipole moment: {result_h2o.properties.get('total_dipole', 0):.4f} Debye")
    print(f"Circuit depth: {result_h2o.circuit.depth()}")
    print(f"Execution time: {result_h2o.execution_time:.2f} seconds\n")
    
    # Example 2: Hydrogen molecule (H2)
    print("=== Simulating Hydrogen (H2) ===")
    result_h2 = agent.simulate_molecule(
        molecule_input='[H][H]',
        input_format='smiles',
        charge=0,
        spin_multiplicity=1,
        method='vqe'
    )
    
    print(f"Energy: {result_h2.energy:.6f} Hartree")
    print(f"Circuit depth: {result_h2.circuit.depth()}")
    
    # Example 3: Lithium Hydride (LiH)
    print("\n=== Simulating Lithium Hydride (LiH) ===")
    result_lih = agent.simulate_molecule(
        molecule_input='[LiH]',  # Correct SMILES for LiH
        input_format='smiles',
        charge=0,
        spin_multiplicity=1,
        method='vqe'
    )
    
    print(f"Energy: {result_lih.energy:.6f} Hartree")
    
    # Example 4: Resource estimation
    print("\n=== Resource Estimation for Methane (CH4) ===")
    from src.molecular.parser import MolecularParser
    parser = MolecularParser()
    ch4_info = parser.parse_molecule('C', 'smiles')
    
    resources = agent.estimate_resources(ch4_info)
    print(f"Number of qubits required: {resources['num_qubits']}")
    print(f"Circuit depth: {resources['circuit_depth']}")
    print(f"Total number of gates: {resources['total_gates']}")
    print(f"CNOT gates: {resources['cnot_gates']}")
    print(f"Estimated execution time: {resources['estimated_time_seconds']:.3f} seconds")
    
    # Save results
    agent.save_results(result_h2o, 'water_simulation.json')
    print("\nResults saved to water_simulation.json")


if __name__ == '__main__':
    main()