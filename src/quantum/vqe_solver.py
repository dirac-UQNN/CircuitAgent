import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import Optimizer, SPSA, COBYLA
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Callable
from ..molecular.parser import MolecularInfo
from .circuit_builder import HardwareAwareCircuitBuilder
from .hamiltonian_builder import MolecularHamiltonian


class VQESolver:
    
    def __init__(self, 
                 backend=None,
                 optimizer: str = 'COBYLA',
                 max_iterations: int = 1000):
        
        self.backend = backend
        self.optimizer_name = optimizer
        self.max_iterations = max_iterations
        self.convergence_history = []
        
    def solve(self,
              mol_info: MolecularInfo,
              circuit: Optional[QuantumCircuit] = None,
              initial_params: Optional[np.ndarray] = None,
              pre_computed_hamiltonian: Optional[SparsePauliOp] = None) -> Dict:
        
        if circuit is None:
            builder = HardwareAwareCircuitBuilder()
            circuit = builder.build_molecular_circuit(mol_info, include_vqe=True)
        
        if pre_computed_hamiltonian is not None:
            hamiltonian = pre_computed_hamiltonian
        else:
            hamiltonian_builder = MolecularHamiltonian()
            hamiltonian = hamiltonian_builder.build_hamiltonian(mol_info)
        
        # Ensure circuit has the correct number of qubits
        if circuit.num_qubits != hamiltonian.num_qubits:
            print(f"WARNING: Circuit qubit mismatch ({circuit.num_qubits} vs {hamiltonian.num_qubits}). Creating new circuit.")
            builder = HardwareAwareCircuitBuilder()
            circuit = builder.build_molecular_circuit(mol_info, include_vqe=True)
        
        num_params = circuit.num_parameters
        if initial_params is None:
            initial_params = np.random.uniform(-np.pi, np.pi, num_params)
        
        optimizer = self._get_optimizer()
        
        if self.backend is None:
            # VQE still requires V1 interface
            estimator = Estimator()
        else:
            from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimator
            estimator = RuntimeEstimator(backend=self.backend)
        
        vqe = VQE(
            estimator=estimator,
            ansatz=circuit,
            optimizer=optimizer,
            initial_point=initial_params,
            callback=self._callback
        )
        
        result = vqe.compute_minimum_eigenvalue(hamiltonian)
        
        return {
            'energy': result.eigenvalue.real,
            'optimal_params': result.optimal_point,
            'optimal_circuit': result.optimal_circuit,
            'convergence_history': self.convergence_history,
            'num_iterations': len(self.convergence_history),
            'converged': result.optimizer_result.fun < 1e-6
        }
    
    def _get_optimizer(self) -> Optimizer:
        
        if self.optimizer_name == 'COBYLA':
            return COBYLA(maxiter=self.max_iterations)
        elif self.optimizer_name == 'SPSA':
            return SPSA(maxiter=self.max_iterations)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    def _callback(self, eval_count: int, params: np.ndarray, value: float, metadata: Dict):
        
        self.convergence_history.append({
            'iteration': eval_count,
            'energy': value,
            'params': params.copy()
        })
    
    def calculate_properties(self,
                           mol_info: MolecularInfo,
                           optimal_circuit: QuantumCircuit,
                           optimal_params: np.ndarray) -> Dict[str, float]:
        
        properties = {}
        
        hamiltonian_builder = MolecularHamiltonian()
        dipole_ops = hamiltonian_builder.build_dipole_operators(mol_info)
        
        # In newer Qiskit versions, use assign_parameters
        bound_circuit = optimal_circuit.assign_parameters(optimal_params)
        
        if self.backend is None:
            # VQE still requires V1 interface
            estimator = Estimator()
        else:
            from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimator
            estimator = RuntimeEstimator(backend=self.backend)
        
        for axis, op in dipole_ops.items():
            result = estimator.run([bound_circuit], [op]).result()
            properties[f'dipole_{axis}'] = result.values[0].real
        
        properties['total_dipole'] = np.sqrt(
            sum(properties[f'dipole_{axis}']**2 for axis in ['x', 'y', 'z'])
        )
        
        return properties
    
    def adaptive_vqe(self,
                    mol_info: MolecularInfo,
                    threshold: float = 1e-3) -> Dict:
        
        builder = HardwareAwareCircuitBuilder()
        circuit = builder.create_excitation_preserving_ansatz(mol_info, 'S')
        
        current_energy = float('inf')
        results = []
        
        while True:
            result = self.solve(mol_info, circuit)
            results.append(result)
            
            if abs(result['energy'] - current_energy) < threshold:
                break
                
            current_energy = result['energy']
            
            circuit = self._add_adaptive_gate(circuit, mol_info, result)
            
            if circuit.num_parameters > 100:
                print("Warning: Circuit becoming too deep, stopping adaptation")
                break
        
        return {
            'final_energy': current_energy,
            'final_circuit': circuit,
            'adaptation_history': results,
            'num_adaptations': len(results)
        }