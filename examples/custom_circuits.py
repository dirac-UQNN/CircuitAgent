#!/usr/bin/env python3
"""
Demonstrate custom circuit building with different ansatz types.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import QuantumCircuitAgent
from src.molecular.parser import MolecularParser
import matplotlib.pyplot as plt


def visualize_circuit_comparison():
    """Compare different ansatz types visually"""
    
    parser = MolecularParser()
    agent = QuantumCircuitAgent()
    
    # Parse a simple molecule
    mol_info = parser.parse_molecule('[H][H]', 'smiles')
    
    print("Custom Circuit Building Examples")
    print("="*50)
    
    # Build different circuit types
    circuits = {}
    
    # 1. Hardware-efficient ansatz
    print("\n1. Hardware-Efficient Ansatz")
    circuit_hw = agent.build_custom_circuit(
        mol_info,
        ansatz_type='hardware_efficient',
        num_layers=2
    )
    circuits['Hardware-Efficient'] = circuit_hw
    print(f"   Qubits: {circuit_hw.num_qubits}")
    print(f"   Depth: {circuit_hw.depth()}")
    print(f"   Parameters: {circuit_hw.num_parameters}")
    
    # 2. UCCSD ansatz
    print("\n2. UCCSD Ansatz")
    circuit_uccsd = agent.build_custom_circuit(
        mol_info,
        ansatz_type='uccsd'
    )
    circuits['UCCSD'] = circuit_uccsd
    print(f"   Qubits: {circuit_uccsd.num_qubits}")
    print(f"   Depth: {circuit_uccsd.depth()}")
    print(f"   Parameters: {circuit_uccsd.num_parameters}")
    
    # 3. Custom circuit from scratch
    print("\n3. Custom Circuit (Minimal)")
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    
    num_qubits = 2 * mol_info.num_orbitals
    custom_circuit = QuantumCircuit(num_qubits)
    
    # Initial state (HF state for H2)
    custom_circuit.x(0)  # Electron 1 in orbital 0
    custom_circuit.x(1)  # Electron 2 in orbital 0 (spin down)
    
    # Simple parameterized rotation
    theta = ParameterVector('θ', 1)
    custom_circuit.ry(theta[0], 0)
    custom_circuit.cx(0, 1)
    
    circuits['Custom Minimal'] = custom_circuit
    print(f"   Qubits: {custom_circuit.num_qubits}")
    print(f"   Depth: {custom_circuit.depth()}")
    print(f"   Parameters: {custom_circuit.num_parameters}")
    
    # Gate count comparison
    print("\n" + "="*50)
    print("Gate Count Comparison:")
    print(f"{'Ansatz Type':<20} {'Total Gates':<15} {'CNOT Gates':<15}")
    print("-"*50)
    
    for name, circuit in circuits.items():
        gate_counts = circuit.count_ops()
        total = sum(gate_counts.values())
        cnots = gate_counts.get('cx', 0)
        print(f"{name:<20} {total:<15} {cnots:<15}")
    
    # Save circuit drawings
    print("\n" + "="*50)
    print("Saving circuit diagrams...")
    
    for name, circuit in circuits.items():
        filename = f"circuit_{name.lower().replace(' ', '_').replace('-', '_')}.txt"
        try:
            with open(filename, 'w') as f:
                f.write(f"{name} Circuit:\n")
                f.write("="*50 + "\n")
                f.write(str(circuit.draw(output='text')))
            print(f"✓ Saved {filename}")
        except Exception as e:
            print(f"✗ Failed to save {filename}: {e}")


def demonstrate_parameterized_circuit():
    """Show how to work with parameterized circuits"""
    
    print("\n" + "="*50)
    print("Parameterized Circuit Example")
    print("="*50)
    
    parser = MolecularParser()
    agent = QuantumCircuitAgent()
    
    # Build a parameterized circuit
    mol_info = parser.parse_molecule('[H][H]', 'smiles')
    circuit = agent.build_custom_circuit(mol_info, num_layers=1)
    
    # Get parameter names
    params = circuit.parameters
    print(f"\nCircuit has {len(params)} parameters:")
    for i, param in enumerate(list(params)[:5]):  # Show first 5
        print(f"  {i}: {param.name}")
    if len(params) > 5:
        print(f"  ... and {len(params)-5} more")
    
    # Bind specific values
    import numpy as np
    param_values = np.random.uniform(-np.pi, np.pi, len(params))
    bound_circuit = circuit.assign_parameters(param_values)
    
    print(f"\nOriginal circuit parameters: {circuit.num_parameters}")
    print(f"Bound circuit parameters: {bound_circuit.num_parameters}")


def main():
    visualize_circuit_comparison()
    demonstrate_parameterized_circuit()
    
    print("\n" + "="*50)
    print("Custom circuit building complete!")
    print("\nKey takeaways:")
    print("- Hardware-efficient ansatz: Good for NISQ devices")
    print("- UCCSD ansatz: Chemically motivated, but deeper")
    print("- Custom circuits: Maximum control for specific problems")


if __name__ == '__main__':
    main()