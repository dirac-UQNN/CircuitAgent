#!/usr/bin/env python3
"""
Compare different optimization levels for circuit construction.
Shows the trade-off between optimization time and circuit quality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import QuantumCircuitAgent
import time


def compare_optimization_levels(smiles='[H][H]', name='H₂'):
    """Compare different optimization levels"""
    
    print(f"\nOptimization Comparison for {name}")
    print("="*60)
    print(f"{'Level':<10} {'Depth':<10} {'Gates':<10} {'Time (s)':<12} {'Energy (Ha)':<15}")
    print("-"*60)
    
    for opt_level in range(4):
        agent = QuantumCircuitAgent(optimization_level=opt_level)
        
        start_time = time.time()
        try:
            result = agent.simulate_molecule(
                smiles,
                method='vqe',
                calculate_properties=False
            )
            elapsed = time.time() - start_time
            
            total_gates = sum(result.circuit.count_ops().values())
            
            print(f"{opt_level:<10} {result.circuit.depth():<10} "
                  f"{total_gates:<10} {elapsed:<12.2f} {result.energy:<15.6f}")
                  
        except Exception as e:
            print(f"{opt_level:<10} Error: {str(e)[:40]}...")


def main():
    print("Circuit Optimization Level Comparison")
    print("="*60)
    print("\nOptimization Levels:")
    print("  0 - Minimal optimization (fastest)")
    print("  1 - Light optimization")
    print("  2 - Standard optimization (default)")
    print("  3 - Heavy optimization (slowest)")
    
    # Test molecules
    molecules = [
        ('[H][H]', 'Hydrogen (H₂)'),
        ('O', 'Water (H₂O)'),
        ('[LiH]', 'Lithium Hydride (LiH)'),
    ]
    
    for smiles, name in molecules:
        compare_optimization_levels(smiles, name)
    
    print("\n" + "="*60)
    print("Recommendations:")
    print("- Use level 0-1 for rapid prototyping and testing")
    print("- Use level 2 for standard simulations")
    print("- Use level 3 for hardware execution or final results")


if __name__ == '__main__':
    main()