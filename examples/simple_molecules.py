#!/usr/bin/env python3
"""
Simple molecular simulations for common small molecules.
These are designed to run quickly on simulators.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import QuantumCircuitAgent
import time


def simulate_molecule(agent, smiles, name, charge=0, spin=1):
    """Helper function to simulate a molecule and print results"""
    print(f"\n{'='*50}")
    print(f"Simulating {name}")
    print(f"{'='*50}")
    
    start_time = time.time()
    try:
        result = agent.simulate_molecule(
            molecule_input=smiles,
            charge=charge,
            spin_multiplicity=spin,
            method='vqe',
            calculate_properties=False  # Skip for speed
        )
        
        elapsed = time.time() - start_time
        
        print(f"✓ Success!")
        print(f"  Energy: {result.energy:.6f} Hartree")
        print(f"  Circuit: {result.circuit.num_qubits} qubits, depth {result.circuit.depth()}")
        print(f"  Time: {elapsed:.2f} seconds")
        
        return result
        
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        return None


def main():
    # Initialize agent with minimal optimization for speed
    agent = QuantumCircuitAgent(optimization_level=1)
    
    print("Small Molecule Quantum Simulations")
    print("="*50)
    
    # Dictionary of molecules to simulate
    molecules = [
        # (SMILES, Name, Charge, Spin)
        ('[H][H]', 'Hydrogen (H₂)', 0, 1),
        ('[He]', 'Helium atom', 0, 1),
        ('[LiH]', 'Lithium Hydride', 0, 1),
        ('[BeH2]', 'Beryllium Hydride', 0, 1),
        ('N#N', 'Nitrogen (N₂)', 0, 1),
        ('O=O', 'Oxygen (O₂)', 0, 3),  # Triplet ground state
        ('F[F]', 'Fluorine (F₂)', 0, 1),
        ('[H-]', 'Hydride ion (H⁻)', -1, 1),
        ('[He+]', 'Helium cation (He⁺)', 1, 2),  # Doublet
        ('C#C', 'Acetylene (C₂H₂)', 0, 1),
    ]
    
    results = []
    for smiles, name, charge, spin in molecules:
        result = simulate_molecule(agent, smiles, name, charge, spin)
        if result:
            results.append((name, result))
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Successfully simulated {len(results)}/{len(molecules)} molecules")
    
    if results:
        print("\nEnergy Summary:")
        for name, result in results:
            print(f"  {name:<25} {result.energy:>10.6f} Ha")
        
        # Find smallest and largest circuits
        min_qubits = min(r.circuit.num_qubits for _, r in results)
        max_qubits = max(r.circuit.num_qubits for _, r in results)
        print(f"\nCircuit sizes: {min_qubits} - {max_qubits} qubits")


if __name__ == '__main__':
    main()