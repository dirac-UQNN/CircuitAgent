from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class MolecularInfo:
    smiles: str
    atoms: List[str]
    coordinates: np.ndarray
    num_electrons: int
    num_orbitals: int
    charge: int
    spin_multiplicity: int
    nuclear_charges: List[int]
    molecular_weight: float
    
    
class MolecularParser:
    
    def __init__(self):
        self.supported_formats = ['smiles', 'mol', 'sdf', 'xyz']
        
    def parse_molecule(self, 
                      input_data: Union[str, Chem.Mol], 
                      input_format: str = 'smiles',
                      charge: int = 0,
                      spin_multiplicity: int = 1) -> MolecularInfo:
        
        if input_format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {input_format}")
        
        mol = self._get_mol_object(input_data, input_format)
        
        if mol is None:
            raise ValueError("Failed to parse molecular input")
        
        # Only add hydrogens if not XYZ format (XYZ has explicit atoms)
        if input_format != 'xyz':
            mol = Chem.AddHs(mol)
        
        AllChem.EmbedMolecule(mol, randomSeed=42)
        if mol.GetNumConformers() > 0:
            AllChem.MMFFOptimizeMolecule(mol)
        
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        nuclear_charges = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        
        conf = mol.GetConformer()
        coordinates = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        
        num_electrons = sum(nuclear_charges) - charge
        num_orbitals = self._estimate_num_orbitals(atoms)
        
        molecular_weight = Descriptors.MolWt(mol)
        
        return MolecularInfo(
            smiles=Chem.MolToSmiles(mol),
            atoms=atoms,
            coordinates=coordinates,
            num_electrons=num_electrons,
            num_orbitals=num_orbitals,
            charge=charge,
            spin_multiplicity=spin_multiplicity,
            nuclear_charges=nuclear_charges,
            molecular_weight=molecular_weight
        )
    
    def _get_mol_object(self, input_data: Union[str, Chem.Mol], input_format: str) -> Optional[Chem.Mol]:
        
        if isinstance(input_data, Chem.Mol):
            return input_data
            
        if input_format == 'smiles':
            return Chem.MolFromSmiles(input_data)
        elif input_format == 'mol':
            return Chem.MolFromMolBlock(input_data)
        elif input_format == 'sdf':
            supplier = Chem.SDMolSupplier()
            supplier.SetData(input_data)
            return next(supplier, None)
        elif input_format == 'xyz':
            return self._parse_xyz(input_data)
        
        return None
    
    def _parse_xyz(self, xyz_string: str) -> Optional[Chem.Mol]:
        
        lines = xyz_string.strip().split('\n')
        if len(lines) < 3:
            return None
            
        try:
            num_atoms = int(lines[0])
            
            mol = Chem.RWMol()
            conf = Chem.Conformer(num_atoms)
            
            for i in range(num_atoms):
                parts = lines[i + 2].split()
                if len(parts) < 4:
                    return None
                    
                symbol = parts[0]
                x, y, z = map(float, parts[1:4])
                
                atom = Chem.Atom(symbol)
                idx = mol.AddAtom(atom)
                conf.SetAtomPosition(idx, (x, y, z))
            
            mol.AddConformer(conf)
            
            # Don't add implicit hydrogens for XYZ format
            # XYZ should have all atoms explicit
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_ADJUSTHS)
            
            return mol
            
        except (ValueError, IndexError):
            return None
    
    def _estimate_num_orbitals(self, atoms: List[str]) -> int:
        # For minimal basis set (STO-3G), estimate number of orbitals
        # This is a rough estimate - the actual value will be determined by PySCF
        orbital_counts = {
            'H': 1, 'He': 1,
            'Li': 2, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 5, 'F': 5, 'Ne': 5,
            'Na': 6, 'Mg': 6, 'Al': 7, 'Si': 8, 'P': 9, 'S': 9, 'Cl': 9, 'Ar': 9
        }
        
        return sum(orbital_counts.get(atom, 9) for atom in atoms)
    
    def get_molecular_properties(self, mol_info: MolecularInfo) -> Dict[str, float]:
        
        mol = Chem.MolFromSmiles(mol_info.smiles)
        
        properties = {
            'molecular_weight': mol_info.molecular_weight,
            'num_atoms': len(mol_info.atoms),
            'num_bonds': mol.GetNumBonds(),
            'num_rings': Chem.Descriptors.RingCount(mol),
            'num_aromatic_rings': Chem.Descriptors.NumAromaticRings(mol),
            'tpsa': Chem.Descriptors.TPSA(mol),
            'logp': Chem.Descriptors.MolLogP(mol),
            'num_rotatable_bonds': Chem.Descriptors.NumRotatableBonds(mol),
            'num_hbd': Chem.Descriptors.NumHDonors(mol),
            'num_hba': Chem.Descriptors.NumHAcceptors(mol)
        }
        
        return properties