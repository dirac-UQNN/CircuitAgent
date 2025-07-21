#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.molecular.parser import MolecularParser, MolecularInfo


class TestMolecularParser:
    
    def setup_method(self):
        self.parser = MolecularParser()
    
    def test_parse_simple_molecules(self):
        """Test parsing of simple molecules"""
        # Test H2
        h2 = self.parser.parse_molecule('[H][H]', 'smiles')
        assert len(h2.atoms) == 2
        assert h2.num_electrons == 2
        assert h2.charge == 0
        assert all(atom == 'H' for atom in h2.atoms)
        
        # Test water
        h2o = self.parser.parse_molecule('O', 'smiles')
        assert len(h2o.atoms) == 3
        assert h2o.num_electrons == 10
        assert h2o.atoms.count('H') == 2
        assert h2o.atoms.count('O') == 1
        
        # Test methane
        ch4 = self.parser.parse_molecule('C', 'smiles')
        assert len(ch4.atoms) == 5
        assert ch4.num_electrons == 10
        assert ch4.atoms.count('C') == 1
        assert ch4.atoms.count('H') == 4
    
    def test_charged_molecules(self):
        """Test parsing of charged molecules"""
        # H+ ion
        h_plus = self.parser.parse_molecule('[H+]', 'smiles', charge=1)
        assert h_plus.num_electrons == 0
        assert h_plus.charge == 1
        
        # OH- ion
        oh_minus = self.parser.parse_molecule('[OH-]', 'smiles', charge=-1)
        assert oh_minus.num_electrons == 10
        assert oh_minus.charge == -1
    
    def test_invalid_smiles(self):
        """Test handling of invalid SMILES"""
        with pytest.raises(ValueError):
            self.parser.parse_molecule('InvalidSMILES123', 'smiles')
    
    def test_xyz_format(self):
        """Test XYZ format parsing"""
        xyz_string = """3
Water molecule
O  0.000000  0.000000  0.000000
H  0.757000  0.586000  0.000000
H -0.757000  0.586000  0.000000
"""
        mol = self.parser.parse_molecule(xyz_string, 'xyz')
        assert len(mol.atoms) == 3
        assert mol.atoms == ['O', 'H', 'H']
        assert mol.coordinates.shape == (3, 3)
    
    def test_molecular_properties(self):
        """Test molecular property calculations"""
        benzene = self.parser.parse_molecule('c1ccccc1', 'smiles')
        props = self.parser.get_molecular_properties(benzene)
        
        assert props['num_atoms'] == len(benzene.atoms)  # 6 C + 6 H = 12
        assert props['num_rings'] == 1
        assert props['num_aromatic_rings'] == 1
        assert props['num_rotatable_bonds'] == 0
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Single atom
        he = self.parser.parse_molecule('[He]', 'smiles')
        assert len(he.atoms) == 1
        assert he.num_electrons == 2
        
        # Empty coordinates should not crash
        assert he.coordinates.shape[0] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])