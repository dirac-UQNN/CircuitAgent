# Quantum Circuit Agent

A hardware-ready quantum circuit builder for molecular simulation. This framework focuses on creating optimized quantum circuits that can run on real quantum hardware, supporting any molecular configuration with VQE-based energy and property calculations.

## Features

- **Hardware-Ready Circuits**: Optimized for real quantum devices with native gate decomposition
- **Universal Molecular Support**: Parse molecules from SMILES, MOL, SDF, or XYZ formats
- **Advanced VQE Implementation**: Including adaptive VQE for improved convergence
- **Property Calculations**: Energy, dipole moment, and other molecular properties
- **Circuit Optimization**: Multiple optimization levels to reduce circuit depth
- **Backend Agnostic**: Works with simulators and real quantum hardware

## Installation

```bash
cd quantum_circuit_agent
pip install -e .
```

## Quick Start

```python
from src import QuantumCircuitAgent

# Initialize the agent
agent = QuantumCircuitAgent(
    backend=None,  # Use simulator
    optimization_level=2
)

# Simulate a water molecule
result = agent.simulate_molecule(
    molecule_input='O',  # SMILES for water
    method='vqe',
    calculate_properties=True
)

print(f"Ground state energy: {result.energy:.6f} Hartree")
print(f"Dipole moment: {result.properties['total_dipole']:.4f} Debye")
```

## Architecture

### Core Components

1. **Molecular Parser** (`src/molecular/parser.py`)
   - Converts molecular inputs to quantum-ready format
   - Extracts atomic coordinates, electrons, and orbitals
   - Supports multiple input formats

2. **Circuit Builder** (`src/quantum/circuit_builder.py`)
   - Hardware-aware circuit construction
   - Multiple ansatz types (hardware-efficient, UCCSD)
   - Connectivity-aware gate placement

3. **VQE Solver** (`src/quantum/vqe_solver.py`)
   - Variational quantum eigensolver implementation
   - Adaptive VQE for automatic circuit growth
   - Property calculations (dipole moment, etc.)

4. **Circuit Optimizer** (`src/optimization/circuit_optimizer.py`)
   - Multi-level optimization strategies
   - Gate cancellation and consolidation
   - Hardware-specific transpilation

## Usage Examples

### Basic Molecular Simulation

```python
# Hydrogen molecule
result = agent.simulate_molecule('[H][H]', method='vqe')

# Lithium hydride with properties
result = agent.simulate_molecule(
    '[Li]H',
    calculate_properties=True
)
```

### Hardware Backend

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
backend = service.backend('ibm_kyoto')

agent = QuantumCircuitAgent(
    backend=backend,
    optimization_level=3
)
```

### Custom Circuit Building

```python
# Build UCCSD ansatz
circuit = agent.build_custom_circuit(
    mol_info,
    ansatz_type='uccsd'
)

# Build hardware-efficient ansatz
circuit = agent.build_custom_circuit(
    mol_info,
    ansatz_type='hardware_efficient',
    num_layers=3
)
```

### Resource Estimation

```python
resources = agent.estimate_resources(mol_info)
print(f"Qubits needed: {resources['num_qubits']}")
print(f"Circuit depth: {resources['circuit_depth']}")
print(f"CNOT count: {resources['cnot_gates']}")
```

## Advanced Features

### Active Space Calculations

For larger molecules, use active space approximation:

```python
from src.quantum.hamiltonian_builder import MolecularHamiltonian

hamiltonian_builder = MolecularHamiltonian()
active_hamiltonian = hamiltonian_builder.get_active_space_hamiltonian(
    mol_info,
    n_active_electrons=4,
    n_active_orbitals=4
)
```

### Adaptive VQE

Automatically grow the ansatz for better accuracy:

```python
result = agent.simulate_molecule(
    molecule_input='O',
    method='adapt-vqe'
)
```

## Hardware Considerations

- **Gate Set**: Circuits are decomposed to native gates (RX, RY, RZ, CX)
- **Connectivity**: Respects device coupling maps
- **Circuit Depth**: Optimized to stay within coherence limits
- **Error Mitigation**: Compatible with Qiskit's error mitigation techniques

## Contributing

Contributions are welcome! Areas for improvement:

- Additional ansatz types
- More molecular property calculations
- Enhanced optimization algorithms
- Error mitigation strategies
- Noise-aware circuit construction

## License

MIT License

## Citation

If you use this software in your research, please cite:

```
@software{quantum_circuit_agent,
  title = {Quantum Circuit Agent: Hardware-Ready Molecular Simulation},
  year = {2024},
  url = {https://github.com/yourusername/quantum-circuit-agent}
}
```