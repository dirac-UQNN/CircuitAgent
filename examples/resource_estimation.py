#!/usr/bin/env python3
"""
Resource estimation for various molecules.
Shows the quantum resources needed without running full simulations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import QuantumCircuitAgent
from src.molecular.parser import MolecularParser


def main():
    agent = QuantumCircuitAgent()
    parser = MolecularParser()
    
    print("Quantum Resource Estimation for Molecular Simulations")
    print("="*70)
    print(f"{'Molecule':<20} {'Qubits':<10} {'Depth':<10} {'Gates':<10} {'CNOT':<10}")
    print("-"*70)
    
    molecules = [
        ('[H]', 'Hydrogen atom'),
        ('[H][H]', 'H₂'),
        ('O', 'Water'),
        ('N', 'Ammonia'),
        ('C', 'Methane'),
        ('CC', 'Ethane'),
        ('CCO', 'Ethanol'),
        ('c1ccccc1', 'Benzene'),
        ('CC(=O)O', 'Acetic acid'),
        ('N[C@@H](C)C(=O)O', 'Alanine'),
    ]
    
    for smiles, name in molecules:
        try:
            mol_info = parser.parse_molecule(smiles, 'smiles')
            resources = agent.estimate_resources(mol_info)
            
            print(f"{name:<20} {resources['num_qubits']:<10} "
                  f"{resources['circuit_depth']:<10} {resources['total_gates']:<10} "
                  f"{resources['cnot_gates']:<10}")
                  
        except Exception as e:
            print(f"{name:<20} Error: {str(e)[:40]}...")
    
    print("\n" + "="*70)
    print("Notes:")
    print("- Qubits = 2 × number of molecular orbitals (for spin)")
    print("- Depth affects circuit runtime and decoherence")
    print("- CNOT gates are typically the slowest and most error-prone")
    print("\nMolecules with >20 qubits are challenging for current hardware")


if __name__ == '__main__':
    main()