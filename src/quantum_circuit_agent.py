import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import json
import yaml

from .molecular.parser import MolecularParser, MolecularInfo
from .quantum.circuit_builder import HardwareAwareCircuitBuilder
from .quantum.vqe_solver import VQESolver
from .quantum.hamiltonian_builder import MolecularHamiltonian
from .optimization.circuit_optimizer import CircuitOptimizer


@dataclass
class SimulationResult:
    molecule: MolecularInfo
    circuit: Any
    energy: float
    properties: Dict[str, float]
    optimization_stats: Dict
    execution_time: float
    convergence_history: List[Dict]
    

class QuantumCircuitAgent:
    
    def __init__(self, 
                 backend=None,
                 optimization_level: int = 2,
                 basis_set: str = 'sto-3g'):
        
        self.backend = backend
        self.optimization_level = optimization_level
        self.basis_set = basis_set
        
        self.parser = MolecularParser()
        self.circuit_builder = None
        self.optimizer = CircuitOptimizer(
            target_backend=backend,
            optimization_level=optimization_level
        )
        
        self._initialize_backend_properties()
        
    def _initialize_backend_properties(self):
        
        if self.backend is not None:
            try:
                backend_config = self.backend.configuration()
                backend_props = self.backend.properties()
                
                native_gates = backend_config.basis_gates
                coupling_map = backend_config.coupling_map
                
                self.backend_properties = {
                    'native_gates': native_gates,
                    'connectivity': coupling_map,
                    'max_depth': 10000,
                    'num_qubits': backend_config.n_qubits
                }
            except:
                self.backend_properties = {}
        else:
            self.backend_properties = {}
        
        self.circuit_builder = HardwareAwareCircuitBuilder(self.backend_properties)
    
    def simulate_molecule(self,
                         molecule_input: Union[str, Dict],
                         input_format: str = 'smiles',
                         charge: int = 0,
                         spin_multiplicity: int = 1,
                         method: str = 'vqe',
                         calculate_properties: bool = True) -> SimulationResult:
        
        import time
        start_time = time.time()
        
        if isinstance(molecule_input, dict):
            mol_info = self._parse_molecule_dict(molecule_input)
        else:
            mol_info = self.parser.parse_molecule(
                molecule_input, 
                input_format,
                charge,
                spin_multiplicity
            )
        
        print(f"Parsed molecule: {mol_info.smiles}")
        print(f"Number of atoms: {len(mol_info.atoms)}")
        print(f"Number of electrons: {mol_info.num_electrons}")
        print(f"Number of orbitals: {mol_info.num_orbitals}")
        
        # Validate molecule
        if mol_info.num_electrons == 0:
            raise ValueError("Cannot simulate molecule with 0 electrons")
        
        if method == 'vqe':
            result = self._run_vqe_simulation(mol_info, calculate_properties)
        elif method == 'adapt-vqe':
            result = self._run_adaptive_vqe(mol_info, calculate_properties)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        execution_time = time.time() - start_time
        
        return SimulationResult(
            molecule=mol_info,
            circuit=result['circuit'],
            energy=result['energy'],
            properties=result.get('properties', {}),
            optimization_stats=self.optimizer.optimization_stats,
            execution_time=execution_time,
            convergence_history=result.get('convergence_history', [])
        )
    
    def _parse_molecule_dict(self, mol_dict: Dict) -> MolecularInfo:
        
        return MolecularInfo(
            smiles=mol_dict.get('smiles', ''),
            atoms=mol_dict['atoms'],
            coordinates=np.array(mol_dict['coordinates']),
            num_electrons=mol_dict['num_electrons'],
            num_orbitals=mol_dict.get('num_orbitals', len(mol_dict['atoms']) * 5),
            charge=mol_dict.get('charge', 0),
            spin_multiplicity=mol_dict.get('spin_multiplicity', 1),
            nuclear_charges=mol_dict.get('nuclear_charges', []),
            molecular_weight=mol_dict.get('molecular_weight', 0.0)
        )
    
    def _run_vqe_simulation(self, 
                           mol_info: MolecularInfo,
                           calculate_properties: bool) -> Dict:
        
        # First build the Hamiltonian to get the correct number of qubits
        from .quantum.hamiltonian_builder import MolecularHamiltonian
        hamiltonian_builder = MolecularHamiltonian(basis=self.basis_set)
        hamiltonian = hamiltonian_builder.build_hamiltonian(mol_info)
        
        # Update mol_info with the correct number of orbitals from PySCF
        mol_info.num_orbitals = hamiltonian_builder.molecular_data.n_orbitals
        
        circuit = self.circuit_builder.build_molecular_circuit(mol_info, include_vqe=True)
        
        print(f"Initial circuit depth: {circuit.depth()}")
        print(f"Number of parameters: {circuit.num_parameters}")
        
        optimized_circuit = self.optimizer.optimize_circuit(circuit)
        
        print(f"Optimized circuit depth: {optimized_circuit.depth()}")
        print(f"Depth reduction: {self.optimizer.optimization_stats['depth_reduction']:.1f}%")
        
        solver = VQESolver(backend=self.backend)
        vqe_result = solver.solve(mol_info, optimized_circuit, pre_computed_hamiltonian=hamiltonian)
        
        print(f"Ground state energy: {vqe_result['energy']:.6f} Ha")
        
        properties = {}
        if calculate_properties:
            properties = solver.calculate_properties(
                mol_info,
                vqe_result['optimal_circuit'],
                vqe_result['optimal_params']
            )
            print(f"Dipole moment: {properties['total_dipole']:.4f} Debye")
        
        return {
            'circuit': vqe_result['optimal_circuit'],
            'energy': vqe_result['energy'],
            'properties': properties,
            'convergence_history': vqe_result['convergence_history'],
            'optimal_params': vqe_result['optimal_params']
        }
    
    def _run_adaptive_vqe(self,
                         mol_info: MolecularInfo,
                         calculate_properties: bool) -> Dict:
        
        solver = VQESolver(backend=self.backend)
        adapt_result = solver.adaptive_vqe(mol_info)
        
        print(f"Adaptive VQE completed after {adapt_result['num_adaptations']} adaptations")
        print(f"Final energy: {adapt_result['final_energy']:.6f} Ha")
        
        properties = {}
        if calculate_properties:
            final_result = adapt_result['adaptation_history'][-1]
            properties = solver.calculate_properties(
                mol_info,
                final_result['optimal_circuit'],
                final_result['optimal_params']
            )
        
        return {
            'circuit': adapt_result['final_circuit'],
            'energy': adapt_result['final_energy'],
            'properties': properties,
            'convergence_history': adapt_result['adaptation_history']
        }
    
    def build_custom_circuit(self,
                           mol_info: MolecularInfo,
                           ansatz_type: str = 'hardware_efficient',
                           num_layers: int = 3) -> Any:
        
        if ansatz_type == 'hardware_efficient':
            circuit = self.circuit_builder.build_molecular_circuit(mol_info)
        elif ansatz_type == 'uccsd':
            circuit = self.circuit_builder.create_excitation_preserving_ansatz(mol_info, 'UCCSD')
        else:
            raise ValueError(f"Unknown ansatz type: {ansatz_type}")
        
        return self.optimizer.optimize_circuit(circuit)
    
    def estimate_resources(self, mol_info: MolecularInfo) -> Dict[str, int]:
        
        circuit = self.circuit_builder.build_molecular_circuit(mol_info)
        optimized = self.optimizer.optimize_circuit(circuit)
        
        gate_counts = optimized.count_ops()
        
        return {
            'num_qubits': optimized.num_qubits,
            'circuit_depth': optimized.depth(),
            'num_parameters': optimized.num_parameters,
            'total_gates': sum(gate_counts.values()),
            'cnot_gates': gate_counts.get('cx', 0),
            'single_qubit_gates': sum(v for k, v in gate_counts.items() if k != 'cx'),
            'estimated_shots': 10000,
            'estimated_time_seconds': self._estimate_execution_time(optimized)
        }
    
    def _estimate_execution_time(self, circuit) -> float:
        
        if self.backend is None:
            gate_time = 1e-6
            measurement_time = 1e-3
        else:
            try:
                gate_time = self.backend.properties().gate_length('cx', [0, 1])
                measurement_time = self.backend.properties().readout_length(0)
            except:
                gate_time = 1e-6
                measurement_time = 1e-3
        
        total_time = (circuit.depth() * gate_time + measurement_time) * 10000
        
        return total_time
    
    def save_results(self, result: SimulationResult, filename: str):
        
        data = {
            'molecule': {
                'smiles': result.molecule.smiles,
                'atoms': result.molecule.atoms,
                'coordinates': result.molecule.coordinates.tolist(),
                'num_electrons': result.molecule.num_electrons,
                'charge': result.molecule.charge,
                'spin_multiplicity': result.molecule.spin_multiplicity
            },
            'energy': result.energy,
            'properties': result.properties,
            'optimization_stats': result.optimization_stats,
            'execution_time': result.execution_time,
            'circuit_depth': result.circuit.depth() if hasattr(result.circuit, 'depth') else None,
            'num_qubits': result.circuit.num_qubits if hasattr(result.circuit, 'num_qubits') else None
        }
        
        if filename.endswith('.yaml'):
            with open(filename, 'w') as f:
                yaml.dump(data, f)
        else:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)