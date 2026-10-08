#!/usr/bin/env python3
"""Filter backbones by compactness, secondary structure, clashes, and hotspot proximity."""

import os
import sys
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
import csv
from multiprocessing import Pool, cpu_count
import numpy as np
from collections import defaultdict


try:
    import mdtraj as md
    HAS_MDTRAJ = True
except ImportError:
    HAS_MDTRAJ = False


@dataclass
class ScreeningResult:
    """ScreeningResult."""
    pdb_file: str
    task_id: str
    helix_pct: float
    sheet_pct: float
    loop_pct: float
    n_binder_residues: int
    n_contacting_residues: int
    min_distance_to_hotspot: float
    avg_distance_to_hotspot: float
    n_clash_atoms: int
    clash_ratio: float
    
    n_ligand_atoms: int = 0
    n_covered_ligand_atoms: int = 0
    ligand_coverage_ratio: float = 0.0
    avg_ligand_distance: float = 999.0
    
    directional_coverage_ratio: float = 0.0
    n_directions_covered: int = 0
    n_directions_total: int = 6
    
    radius_of_gyration: float = 0.0
    rg_threshold: float = 0.0
    
    passed_ss_filter: bool = False
    passed_proximity_filter: bool = False
    passed_clash_filter: bool = False
    passed_ligand_coverage_filter: bool = False
    passed_directional_coverage_filter: bool = False
    passed_rg_filter: bool = False
    passed_overall: bool = False
    method: str = 'DSSP'  # 'DSSP' or 'Phi-Psi'
    error: Optional[str] = None


class PDBParser:
    """PDBParser."""
    
    @staticmethod
    def parse_pdb(pdb_file: str) -> Dict[str, List[Dict]]:
        """parse pdb."""
        chains = defaultdict(list)
        file_path = Path(pdb_file)
        file_ext = file_path.suffix.lower()
        
        
        if file_ext in ['.cif', '.mmcif']:
            return PDBParser._parse_cif(pdb_file)
        
        
        with open(pdb_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    chain_id = line[21].strip()
                    res_num = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    
                    chains[chain_id].append({
                        'atom_name': atom_name,
                        'res_name': res_name,
                        'res_num': res_num,
                        'x': x, 'y': y, 'z': z,
                        'chain': chain_id
                    })
        
        return dict(chains)
    
    @staticmethod
    def _parse_cif(cif_file: str) -> Dict[str, List[Dict]]:
        """ parse cif."""
        chains = defaultdict(list)
        
        try:
            
            try:
                from Bio.PDB import MMCIFParser
                parser = MMCIFParser(QUIET=True)
                structure = parser.get_structure('structure', cif_file)
                
                
                for model in structure:
                    for chain in structure[0]:
                        chain_id = chain.id.strip()
                        for residue in chain:
                            
                            if residue.id[0] == ' ':
                                res_num = residue.id[1]  
                                res_name = residue.get_resname()
                                
                                for atom in residue:
                                    atom_name = atom.id
                                    coord = atom.coord
                                    
                                    chains[chain_id].append({
                                        'atom_name': atom_name,
                                        'res_name': res_name,
                                        'res_num': res_num,
                                        'x': coord[0], 'y': coord[1], 'z': coord[2],
                                        'chain': chain_id
                                    })
                
                if chains:
                    return dict(chains)
            except ImportError:
                
                pass
            except Exception:
                
                pass
            
            
            
            with open(cif_file, 'r', encoding='utf-8', errors='ignore') as f:
                in_atom_site = False
                atom_site_fields = []
                for line in f:
                    line_stripped = line.strip()
                    if line_stripped.startswith('_atom_site.'):
                        in_atom_site = True
                        
                        field_name = line_stripped.split('.', 1)[1].split()[0]
                        atom_site_fields.append(field_name)
                        continue
                    if in_atom_site and line_stripped.startswith('ATOM'):
                        
                        parts = line_stripped.split()
                        if len(parts) >= 10:
                            try:
                                
                                atom_name_idx = atom_site_fields.index('label_atom_id') if 'label_atom_id' in atom_site_fields else 3
                                res_name_idx = atom_site_fields.index('label_comp_id') if 'label_comp_id' in atom_site_fields else 5
                                chain_id_idx = atom_site_fields.index('label_asym_id') if 'label_asym_id' in atom_site_fields else 6
                                res_num_idx = atom_site_fields.index('label_seq_id') if 'label_seq_id' in atom_site_fields else 8
                                x_idx = atom_site_fields.index('Cartn_x') if 'Cartn_x' in atom_site_fields else 10
                                y_idx = atom_site_fields.index('Cartn_y') if 'Cartn_y' in atom_site_fields else 11
                                z_idx = atom_site_fields.index('Cartn_z') if 'Cartn_z' in atom_site_fields else 12
                                
                                atom_name = parts[atom_name_idx] if atom_name_idx < len(parts) else ''
                                res_name = parts[res_name_idx] if res_name_idx < len(parts) else ''
                                chain_id = parts[chain_id_idx] if chain_id_idx < len(parts) else ''
                                res_num_str = parts[res_num_idx] if res_num_idx < len(parts) else '?'
                                x = float(parts[x_idx]) if x_idx < len(parts) else 0.0
                                y = float(parts[y_idx]) if y_idx < len(parts) else 0.0
                                z = float(parts[z_idx]) if z_idx < len(parts) else 0.0
                                
                                
                                try:
                                    res_num = int(res_num_str) if res_num_str != '?' else 0
                                except ValueError:
                                    res_num = 0
                                
                                chains[chain_id].append({
                                    'atom_name': atom_name,
                                    'res_name': res_name,
                                    'res_num': res_num,
                                    'x': x, 'y': y, 'z': z,
                                    'chain': chain_id
                                })
                            except (ValueError, IndexError, AttributeError):
                                
                                try:
                                    if len(parts) >= 13:
                                        atom_name = parts[3]  # label_atom_id
                                        res_name = parts[5]    # label_comp_id
                                        chain_id = parts[6]    # label_asym_id
                                        res_num_str = parts[8] # label_seq_id
                                        x = float(parts[10])  # Cartn_x
                                        y = float(parts[11])  # Cartn_y
                                        z = float(parts[12])  # Cartn_z
                                        
                                        try:
                                            res_num = int(res_num_str) if res_num_str != '?' else 0
                                        except ValueError:
                                            res_num = 0
                                        
                                        chains[chain_id].append({
                                            'atom_name': atom_name,
                                            'res_name': res_name,
                                            'res_num': res_num,
                                            'x': x, 'y': y, 'z': z,
                                            'chain': chain_id
                                        })
                                except (ValueError, IndexError):
                                    continue
                    elif in_atom_site and line_stripped.startswith('#'):
                        
                        break
            
            return dict(chains)
        except Exception as e:
            print(f" [WARN] readCIFfile {Path(cif_file).name} failed: {e}")
            return {}
    
    @staticmethod
    def get_backbone_atoms(chains: Dict[str, List[Dict]], chain_id: str) -> Dict[int, Dict]:
        """get backbone atoms."""
        backbone = defaultdict(dict)
        
        for atom in chains.get(chain_id, []):
            if atom['atom_name'] in ['N', 'CA', 'C', 'O']:
                res_num = atom['res_num']
                backbone[res_num][atom['atom_name']] = np.array([atom['x'], atom['y'], atom['z']])
        
        
        complete_residues = {
            res_num: atoms for res_num, atoms in backbone.items()
            if all(atom in atoms for atom in ['N', 'CA', 'C'])
        }
        
        return complete_residues
    
    @staticmethod
    def calculate_dihedral(p1, p2, p3, p4):
        """calculate dihedral."""
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        m1 = np.cross(n1, b2 / np.linalg.norm(b2))
        
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        
        angle = np.degrees(np.arctan2(y, x))
        return angle


class SecondaryStructureCalculator:
    """SecondaryStructureCalculator."""
    
    @staticmethod
    def has_mdtraj() -> bool:
        """has mdtraj."""
        return HAS_MDTRAJ
    
    @staticmethod
    def has_dssp() -> bool:
        """has dssp."""
        return shutil.which('dssp') is not None or shutil.which('mkdssp') is not None
    
    @staticmethod
    def run_mdtraj_dssp(pdb_file: str, chain_id: str = None) -> Optional[Dict[str, float]]:
        """run mdtraj dssp."""
        if not HAS_MDTRAJ:
            return None
        
        try:
            
            traj = md.load(pdb_file)
            
            
            dssp = md.compute_dssp(traj, simplified=True)
            
            
            ss_array = dssp[0]
            
            
            if chain_id:
                topology = traj.topology
                chain_residues = [r for r in topology.residues if r.chain.chain_id == chain_id]
                
                if not chain_residues:
                    return None
                
                chain_indices = [r.index for r in chain_residues]
                ss_to_count = [ss_array[i] for i in chain_indices]
            else:
                ss_to_count = ss_array
            
            
            ss_counts = {'H': 0, 'E': 0, 'C': 0}
            for ss in ss_to_count:
                if ss in ss_counts:
                    ss_counts[ss] += 1
            
            total = len(ss_to_count)
            if total == 0:
                return None
            
            return {
                'helix': ss_counts['H'] / total * 100,
                'sheet': ss_counts['E'] / total * 100,
                'loop': ss_counts['C'] / total * 100
            }
        
        except Exception as e:
            print(f"[WARN] MDTraj DSSP failed: {e}")
            return None
    
    @staticmethod
    def run_dssp(pdb_file: str, chain_id: str = None) -> Optional[Dict[str, float]]:
        """run dssp."""
        dssp_cmd = 'dssp' if shutil.which('dssp') else 'mkdssp'
        
        try:
            
            temp_pdb = None
            if chain_id:
                temp_pdb = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
                with open(pdb_file, 'r') as f:
                    for line in f:
                        if line.startswith('ATOM') or line.startswith('HETATM'):
                            if len(line) > 21 and line[21] == chain_id:
                                temp_pdb.write(line)
                        elif line.startswith('TER') or line.startswith('END'):
                            temp_pdb.write(line)
                temp_pdb.close()
                input_file = temp_pdb.name
            else:
                input_file = pdb_file
            
            result = subprocess.run(
                [dssp_cmd, input_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            
            if temp_pdb:
                try:
                    os.unlink(temp_pdb.name)
                except:
                    pass
            
            if result.returncode != 0:
                return None
            
            
            ss_counts = {'H': 0, 'E': 0, 'C': 0}  # Helix, Sheet, Coil/Loop
            total = 0
            
            in_data = False
            for line in result.stdout.split('\n'):
                if line.startswith('  #  RESIDUE'):
                    in_data = True
                    continue
                
                if in_data and len(line) > 16:
                    ss = line[16]
                    if ss in 'HGI':  # α-helix, 3-10 helix, π-helix
                        ss_counts['H'] += 1
                        total += 1
                    elif ss in 'EB':  # β-sheet, β-bridge
                        ss_counts['E'] += 1
                        total += 1
                    elif ss in ' TC':  # Loop/Coil/Turn
                        ss_counts['C'] += 1
                        total += 1
            
            if total == 0:
                return None
            
            return {
                'helix': ss_counts['H'] / total * 100,
                'sheet': ss_counts['E'] / total * 100,
                'loop': ss_counts['C'] / total * 100
            }
        
        except Exception as e:
            print(f"[WARN] DSSP execution failed: {e}")
            
            if temp_pdb:
                try:
                    os.unlink(temp_pdb.name)
                except:
                    pass
            return None
    
    @staticmethod
    def calculate_phi_psi(backbone: Dict[int, Dict]) -> Dict[int, Tuple[float, float]]:
        """calculate phi psi."""
        residues = sorted(backbone.keys())
        phi_psi = {}
        
        for i, res_num in enumerate(residues):
            phi = psi = None
            
            
            if i > 0:
                prev_res = residues[i-1]
                if 'C' in backbone[prev_res] and all(atom in backbone[res_num] for atom in ['N', 'CA', 'C']):
                    phi = PDBParser.calculate_dihedral(
                        backbone[prev_res]['C'],
                        backbone[res_num]['N'],
                        backbone[res_num]['CA'],
                        backbone[res_num]['C']
                    )
            
            
            if i < len(residues) - 1:
                next_res = residues[i+1]
                if all(atom in backbone[res_num] for atom in ['N', 'CA', 'C']) and 'N' in backbone[next_res]:
                    psi = PDBParser.calculate_dihedral(
                        backbone[res_num]['N'],
                        backbone[res_num]['CA'],
                        backbone[res_num]['C'],
                        backbone[next_res]['N']
                    )
            
            if phi is not None and psi is not None:
                phi_psi[res_num] = (phi, psi)
        
        return phi_psi
    
    @staticmethod
    def classify_ss_from_phi_psi(phi: float, psi: float) -> str:
        """classify ss from phi psi."""
        # α-helix region: phi ~ -60°±30°, psi ~ -45°±30°
        
        if -100 < phi < -30 and -80 < psi < 50:
            return 'H'
        
        # β-sheet region
        
        
        
        elif phi <= -100 and 50 <= psi <= 180:
            return 'E'
        elif -100 < phi <= -70 and 90 < psi < 140:
            return 'E'
        
        
        elif phi <= -100 and -180 < psi < -100:
            return 'E'
        
        
        
        
        elif 50 < phi < 180 and -180 < psi < -100:
            return 'E'
        elif 50 < phi < 180 and -100 < psi < -50:
            return 'E'
        
        # Left-handed helix (rare, mainly GLY, PRO)
        
        elif 30 < phi < 100 and -60 < psi < 60:
            return 'H'
        
        # Everything else is loop/coil (turns, bends, etc.)
        else:
            return 'C'
    
    @staticmethod
    def calculate_ss_phi_psi(backbone: Dict[int, Dict]) -> Dict[str, float]:
        """calculate ss phi psi."""
        phi_psi = SecondaryStructureCalculator.calculate_phi_psi(backbone)
        
        if not phi_psi:
            return {'helix': 0, 'sheet': 0, 'loop': 100}
        
        ss_counts = {'H': 0, 'E': 0, 'C': 0}
        
        for phi, psi in phi_psi.values():
            ss = SecondaryStructureCalculator.classify_ss_from_phi_psi(phi, psi)
            if ss in ss_counts:
                ss_counts[ss] += 1
        
        total = sum(ss_counts.values())
        if total == 0:
            return {'helix': 0, 'sheet': 0, 'loop': 100}
        
        return {
            'helix': ss_counts['H'] / total * 100,
            'sheet': ss_counts['E'] / total * 100,
            'loop': ss_counts['C'] / total * 100
        }


class LigandCoverageCalculator:
    """LigandCoverageCalculator."""
    
    @staticmethod
    def calculate_ligand_coverage(binder_chain: List[Dict], all_chains: Dict[str, List[Dict]],
                                 ligand_chain: str = 'B', ligand_res_nums: List[int] = None,
                                 coverage_distance: float = 4.0) -> Dict:
        """calculate ligand coverage."""
        
        binder_atoms = []
        for atom in binder_chain:
            if atom['atom_name'] != 'H':  
                binder_atoms.append(np.array([atom['x'], atom['y'], atom['z']]))
        
        if not binder_atoms:
            return {
                'n_ligand_atoms': 0,
                'n_covered_atoms': 0,
                'coverage_ratio': 0.0,
                'avg_distance': 999.0
            }
        
        
        ligand_atoms = []
        if ligand_chain in all_chains:
            for atom in all_chains[ligand_chain]:
                
                if ligand_res_nums is not None:
                    if atom['res_num'] in ligand_res_nums:
                        if atom['atom_name'] != 'H':
                            ligand_atoms.append(np.array([atom['x'], atom['y'], atom['z']]))
                else:
                    
                    if atom['atom_name'] != 'H':
                        ligand_atoms.append(np.array([atom['x'], atom['y'], atom['z']]))
        
        if not ligand_atoms:
            return {
                'n_ligand_atoms': 0,
                'n_covered_atoms': 0,
                'coverage_ratio': 0.0,
                'avg_distance': 999.0
            }
        
        
        covered_count = 0
        min_distances = []
        
        for lig_atom in ligand_atoms:
            distances = [np.linalg.norm(lig_atom - binder_atom) for binder_atom in binder_atoms]
            min_dist = min(distances)
            min_distances.append(min_dist)
            
            if min_dist < coverage_distance:
                covered_count += 1
        
        coverage_ratio = covered_count / len(ligand_atoms) if ligand_atoms else 0.0
        avg_distance = np.mean(min_distances) if min_distances else 999.0
        
        
        directional_coverage = LigandCoverageCalculator._calculate_directional_coverage(
            ligand_atoms, binder_atoms, coverage_distance
        )
        
        return {
            'n_ligand_atoms': len(ligand_atoms),
            'n_covered_atoms': covered_count,
            'coverage_ratio': coverage_ratio,
            'avg_distance': avg_distance,
            'directional_coverage': directional_coverage['coverage_ratio'],
            'n_directions_covered': directional_coverage['n_covered'],
            'n_directions_total': directional_coverage['n_total']
        }
    
    @staticmethod
    def _calculate_directional_coverage(ligand_atoms: List[np.ndarray], 
                                       binder_atoms: List[np.ndarray],
                                       distance_threshold: float = 4.0) -> Dict:
        """ calculate directional coverage."""
        if not ligand_atoms or not binder_atoms:
            return {'n_total': 6, 'n_covered': 0, 'coverage_ratio': 0.0}
        
        
        ligand_centroid = np.mean(ligand_atoms, axis=0)
        
        
        directions = {
            '+X': np.array([1, 0, 0]),
            '-X': np.array([-1, 0, 0]),
            '+Y': np.array([0, 1, 0]),
            '-Y': np.array([0, -1, 0]),
            '+Z': np.array([0, 0, 1]),
            '-Z': np.array([0, 0, -1])
        }
        
        
        covered_directions = 0
        for direction_name, direction_vec in directions.items():
            
            min_dist_in_direction = float('inf')
            
            for binder_atom in binder_atoms:
                
                vec_to_binder = binder_atom - ligand_centroid
                distance = np.linalg.norm(vec_to_binder)
                
                if distance == 0:
                    continue
                
                
                vec_to_binder_norm = vec_to_binder / distance
                
                
                cosine_similarity = np.dot(vec_to_binder_norm, direction_vec)
                
                
                if cosine_similarity > 0.5:
                    min_dist_in_direction = min(min_dist_in_direction, distance)
            
            
            if min_dist_in_direction < distance_threshold:
                covered_directions += 1
        
        coverage_ratio = covered_directions / len(directions)
        
        return {
            'n_total': len(directions),
            'n_covered': covered_directions,
            'coverage_ratio': coverage_ratio
        }


class ProximityCalculator:
    """ProximityCalculator."""
    
    @staticmethod
    def calculate_distances(binder_chain: List[Dict], all_chains: Dict[str, List[Dict]], 
                          hotspot_residues: Dict[str, List[int]], contact_distance: float = 7.0) -> Dict:
        """calculate distances."""
        
        binder_ca = {}
        for atom in binder_chain:
            if atom['atom_name'] == 'CA':
                binder_ca[atom['res_num']] = np.array([atom['x'], atom['y'], atom['z']])
        
        
        hotspot_atoms = []
        for chain_id, res_nums in hotspot_residues.items():
            if chain_id not in all_chains:
                continue  
            
            chain_atoms = all_chains[chain_id]
            for atom in chain_atoms:
                if atom['res_num'] in res_nums and atom['atom_name'] != 'H':
                    hotspot_atoms.append(np.array([atom['x'], atom['y'], atom['z']]))
        
        if not binder_ca or not hotspot_atoms:
            return {
                'n_binder_residues': len(binder_ca),
                'n_contacting': 0,
                'min_distance': 999.0,
                'avg_distance': 999.0
            }
        
        
        min_distances = []
        contacting_residues = 0
        
        for ca_pos in binder_ca.values():
            distances = [np.linalg.norm(ca_pos - hotspot_atom) for hotspot_atom in hotspot_atoms]
            min_dist = min(distances)
            min_distances.append(min_dist)
            
            if min_dist < contact_distance:
                contacting_residues += 1
        
        return {
            'n_binder_residues': len(binder_ca),
            'n_contacting': contacting_residues,
            'min_distance': min(min_distances) if min_distances else 999.0,
            'avg_distance': np.mean(min_distances) if min_distances else 999.0
        }


class RGCalculator:
    """RGCalculator."""
    
    @staticmethod
    def calculate_rg(binder_chain: List[Dict], use_ca_only: bool = True) -> Dict:
        """calculate rg."""
        
        ca_atoms = []
        for atom in binder_chain:
            if atom['atom_name'] == 'CA':
                ca_atoms.append(np.array([atom['x'], atom['y'], atom['z']]))
        
        if len(ca_atoms) < 2:
            return {
                'rg': 0.0,
                'rg_threshold': 0.0,
                'length': len(ca_atoms)
            }
        
        ca_coords = np.array(ca_atoms)
        
        
        centroid = ca_coords.mean(axis=0)
        
        
        squared_distances = np.sum((ca_coords - centroid) ** 2, axis=-1)
        
        
        rg = np.sqrt(np.mean(squared_distances) + 1e-8)
        
        
        
        length = len(ca_atoms)
        rg_threshold = 2.38 * (length ** 0.365)  
        
        return {
            'rg': float(rg),
            'rg_threshold': float(rg_threshold),
            'length': length
        }


class ClashDetector:
    """ClashDetector."""
    
    @staticmethod
    def detect_all_atom_clashes(binder_chain: List[Dict], receptor_chains: Dict[str, List[Dict]], 
                                binder_chain_id: str, clash_distance: float = 3.0) -> Dict:
        """detect all atom clashes."""
        
        binder_atoms = []
        for atom in binder_chain:
            if atom['atom_name'] != 'H':  
                binder_atoms.append(np.array([atom['x'], atom['y'], atom['z']]))
        
        
        receptor_atoms = []
        for chain_id, chain_atoms in receptor_chains.items():
            if chain_id == binder_chain_id:
                continue  
            
            for atom in chain_atoms:
                if atom['atom_name'] != 'H':  
                    receptor_atoms.append(np.array([atom['x'], atom['y'], atom['z']]))
        
        if not binder_atoms or not receptor_atoms:
            return {
                'n_binder_atoms': len(binder_atoms),
                'n_clash_atoms': 0,
                'clash_ratio': 0.0
            }
        
        
        n_clash_atoms = 0
        
        for binder_atom in binder_atoms:
            
            distances = [np.linalg.norm(binder_atom - receptor_atom) 
                        for receptor_atom in receptor_atoms]
            min_dist = min(distances)
            
            if min_dist < clash_distance:
                n_clash_atoms += 1
        
        clash_ratio = n_clash_atoms / len(binder_atoms) if binder_atoms else 0.0
        
        return {
            'n_binder_atoms': len(binder_atoms),
            'n_clash_atoms': n_clash_atoms,
            'clash_ratio': clash_ratio
        }


def screen_single_pdb(args: Tuple) -> ScreeningResult:
    """screen single pdb."""
    pdb_file, config = args
    
    try:
        
        parser = PDBParser()
        chains = parser.parse_pdb(pdb_file)
        
        if not chains:
            return ScreeningResult(
                pdb_file=str(pdb_file), task_id='', helix_pct=0, sheet_pct=0, loop_pct=0,
                n_binder_residues=0, n_contacting_residues=0,
                min_distance_to_hotspot=999, avg_distance_to_hotspot=999,
                n_clash_atoms=0, clash_ratio=0.0,
                passed_ss_filter=False, passed_proximity_filter=False, passed_clash_filter=False,
                passed_overall=False, method='', error='noparsePDBfile'
            )
        
        
        binder_chain_id = config['binder_chain']
        
        if binder_chain_id not in chains:
            return ScreeningResult(
                pdb_file=str(pdb_file), task_id='', helix_pct=0, sheet_pct=0, loop_pct=0,
                n_binder_residues=0, n_contacting_residues=0,
                min_distance_to_hotspot=999, avg_distance_to_hotspot=999,
                n_clash_atoms=0, clash_ratio=0.0,
                passed_ss_filter=False, passed_proximity_filter=False, passed_clash_filter=False,
                passed_overall=False, method='', error=f'Binder chain {binder_chain_id} not found in PDB'
            )
        
        
        
        ss_calc = SecondaryStructureCalculator()
        method = 'Unknown'
        ss_result = None
        
        
        if ss_calc.has_mdtraj():
            ss_result = ss_calc.run_mdtraj_dssp(str(pdb_file), chain_id=binder_chain_id)
            if ss_result:
                method = 'MDTraj-DSSP'
        
        
        if ss_result is None and ss_calc.has_dssp():
            ss_result = ss_calc.run_dssp(str(pdb_file), chain_id=binder_chain_id)
            if ss_result:
                method = 'DSSP'
        
        
        if ss_result is None:
            backbone = parser.get_backbone_atoms(chains, binder_chain_id)
            ss_result = ss_calc.calculate_ss_phi_psi(backbone)
            method = 'Phi-Psi-v1.6'
        
        helix_pct = ss_result['helix']
        sheet_pct = ss_result['sheet']
        loop_pct = ss_result['loop']
        
        
        
        if config['hotspot_residues']:
            proximity_calc = ProximityCalculator()
            proximity_result = proximity_calc.calculate_distances(
                chains[binder_chain_id],
                chains,  
                config['hotspot_residues'],  # {chain_id: [res_nums]}
                contact_distance=config['contact_distance']  
            )
            passed_proximity = (
                proximity_result['n_contacting'] >= config['min_contacting_residues']
            )
        else:
            
            proximity_result = {
                'n_binder_residues': len(set(atom['res_num'] for atom in chains[binder_chain_id])),
                'n_contacting': 0,
                'min_distance': 0,
                'avg_distance': 0
            }
            passed_proximity = True  
        
        
        if 'ligand_coverage_enabled' in config and config['ligand_coverage_enabled']:
            coverage_calc = LigandCoverageCalculator()
            coverage_result = coverage_calc.calculate_ligand_coverage(
                chains[binder_chain_id],
                chains,
                ligand_chain=config.get('ligand_chain', 'B'),
                ligand_res_nums=config.get('ligand_res_nums', None),
                coverage_distance=config.get('ligand_coverage_distance', 4.0)
            )
            
            passed_ligand_coverage = (
                coverage_result['coverage_ratio'] >= config.get('min_ligand_coverage', 0.7)
            )
            
            passed_directional_coverage = (
                coverage_result['directional_coverage'] >= config.get('min_directional_coverage', 0.5)
            )
            
            if config.get('require_ligand', False):
                
                if coverage_result['n_ligand_atoms'] == 0:
                    passed_ligand_coverage = False
                    passed_directional_coverage = False
        else:
            
            coverage_result = {
                'n_ligand_atoms': 0,
                'n_covered_atoms': 0,
                'coverage_ratio': 0.0,
                'avg_distance': 0.0,
                'directional_coverage': 0.0,
                'n_directions_covered': 0,
                'n_directions_total': 6
            }
            
            if config.get('require_ligand', False):
                
                ligand_chain = config.get('ligand_chain', 'B')
                has_ligand = False
                if ligand_chain in chains:
                    for atom in chains[ligand_chain]:
                        
                        
                        if atom.get('res_name', '') not in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                                                             'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                                                             'THR', 'TRP', 'TYR', 'VAL']:
                            has_ligand = True
                            break
                        
                        ligand_res_nums = config.get('ligand_res_nums', None)
                        if ligand_res_nums is not None and atom['res_num'] in ligand_res_nums:
                            has_ligand = True
                            break
                passed_ligand_coverage = has_ligand
                passed_directional_coverage = has_ligand
            else:
                passed_ligand_coverage = True  
                passed_directional_coverage = True  
        
        
        clash_detector = ClashDetector()
        clash_result = clash_detector.detect_all_atom_clashes(
            chains[binder_chain_id],
            chains,
            binder_chain_id,
            clash_distance=3.0  
        )
        
        
        rg_calculator = RGCalculator()
        rg_result = rg_calculator.calculate_rg(chains[binder_chain_id], use_ca_only=True)
        
        
        passed_ss = (
            sheet_pct < config['max_sheet_pct'] and
            loop_pct < config['max_loop_pct']
        )
        
        
        passed_clash = (
            clash_result['clash_ratio'] < config['max_clash_ratio']
        )
        
        
        if config.get('rg_filter_enabled', False):
            max_rg = config.get('max_rg', None)
            if max_rg is not None:
                
                passed_rg = rg_result['rg'] < max_rg
            else:
                
                passed_rg = rg_result['rg'] < rg_result['rg_threshold']
        else:
            passed_rg = True  
        
        
        if config.get('use_directional_coverage', False):
            
            passed_overall = passed_ss and passed_proximity and passed_clash and passed_directional_coverage and passed_rg
        else:
            
            passed_overall = passed_ss and passed_proximity and passed_clash and passed_ligand_coverage and passed_rg
        
        
        
        
        parent_name = Path(pdb_file).parent.name
        if parent_name.startswith('task_'):
            task_id = parent_name
        else:
            
            filename = Path(pdb_file).stem
            if filename.startswith('structure_'):
                
                task_id = filename.replace('structure_', '', 1)
            else:
                
                task_id = filename
        
        return ScreeningResult(
            pdb_file=str(pdb_file),
            task_id=task_id,
            helix_pct=helix_pct,
            sheet_pct=sheet_pct,
            loop_pct=loop_pct,
            n_binder_residues=proximity_result['n_binder_residues'],
            n_contacting_residues=proximity_result['n_contacting'],
            min_distance_to_hotspot=proximity_result['min_distance'],
            avg_distance_to_hotspot=proximity_result['avg_distance'],
            n_clash_atoms=clash_result['n_clash_atoms'],
            clash_ratio=clash_result['clash_ratio'],
            n_ligand_atoms=coverage_result['n_ligand_atoms'],
            n_covered_ligand_atoms=coverage_result['n_covered_atoms'],
            ligand_coverage_ratio=coverage_result['coverage_ratio'],
            avg_ligand_distance=coverage_result['avg_distance'],
            directional_coverage_ratio=coverage_result['directional_coverage'],
            n_directions_covered=coverage_result['n_directions_covered'],
            n_directions_total=coverage_result['n_directions_total'],
            radius_of_gyration=rg_result['rg'],
            rg_threshold=rg_result['rg_threshold'],
            passed_ss_filter=passed_ss,
            passed_proximity_filter=passed_proximity,
            passed_clash_filter=passed_clash,
            passed_ligand_coverage_filter=passed_ligand_coverage,
            passed_directional_coverage_filter=passed_directional_coverage,
            passed_rg_filter=passed_rg,
            passed_overall=passed_overall,
            method=method,
            error=None
        )
    
    except Exception as e:
        return ScreeningResult(
            pdb_file=str(pdb_file), task_id='', helix_pct=0, sheet_pct=0, loop_pct=0,
            n_binder_residues=0, n_contacting_residues=0,
            min_distance_to_hotspot=999, avg_distance_to_hotspot=999,
            n_clash_atoms=0, clash_ratio=0.0,
            passed_ss_filter=False, passed_proximity_filter=False, passed_clash_filter=False,
            passed_overall=False, method='', error=str(e)
        )


def collect_pdb_files(output_dir: Path) -> Tuple[List[Path], List[Path]]:
    """collect pdb files."""
    pdb_files = []
    skipped_files = []
    
    
    has_task_dirs = any(d.is_dir() and d.name.startswith('task_') for d in output_dir.iterdir())
    
    if has_task_dirs:
        print(f"[INFO] Detected old directory structure (task_* subdirectories).")
        
        for task_dir in sorted(output_dir.glob('task_*')):
            if task_dir.is_dir():
                
                for pdb_file in task_dir.glob('*.pdb'):
                    pdb_files.append(pdb_file)
                for cif_file in task_dir.glob('*.cif'):
                    pdb_files.append(cif_file)
                for cif_file in task_dir.glob('*.mmcif'):
                    pdb_files.append(cif_file)
    else:
        print(f"[INFO] Detected new directory structure (PDBs/CIFs directly in output_dir).")
        
        for pdb_file in sorted(output_dir.glob('*.pdb')):
            if pdb_file.parent.name != "screen_results":  
                pdb_files.append(pdb_file)
        for cif_file in sorted(output_dir.glob('*.cif')):
            if cif_file.parent.name != "screen_results":
                pdb_files.append(cif_file)
        for cif_file in sorted(output_dir.glob('*.mmcif')):
            if cif_file.parent.name != "screen_results":
                pdb_files.append(cif_file)
    
    return pdb_files, skipped_files


def main():
    parser = argparse.ArgumentParser(
        description='RFdiffusion2backbonescreen -:backbone',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # CD32 project (use MAPPED residue numbers! A20,A24,A37 are on chain A)
  python backbone_screen.py CD32_covalent_outputs --hotspots A20,A24,A37 --target-chain B
  
  # PD-L1 project (hotspots on chain A, Binder is chain B)
  python backbone_screen.py pdl1_outputs --hotspots A111,A115,A122,A123 --target-chain B
  
  # Custom thresholds (relax criteria)
  python backbone_screen.py CD32_covalent_outputs --hotspots A20,A24,A37 --max-sheet 60 --max-loop 25 --max-clash 0.4
  
  # Enable RG filter with fixed threshold
  python backbone_screen.py CD32_covalent_outputs --hotspots A20,A24,A37 --rg-filter --max-rg 15.0
  
  # Enable RG filter with empirical formula (2.38 * L^0.365)
  python backbone_screen.py CD32_covalent_outputs --hotspots A20,A24,A37 --rg-filter
  
  # Parallel processing (specify number of processes)
  python backbone_screen.py CD32_covalent_outputs --hotspots A20,A24,A37 --processes 32
  
  # Strict clash filter (no clashes allowed)
  python backbone_screen.py CD32_covalent_outputs --hotspots A20,A24,A37 --max-clash 0.1

Important:
  - --target-chain: BINDER chain to analyze (default: B)
  - --hotspots: Hotspot residues with chain ID (e.g., A20 = chain A residue 20)
  - PDB residue numbers should be accurate (no mapping needed if using DREAM-boltz2 outputs)
  - Secondary structure criteria apply to Binder chain only
  - Proximity measures distance from Binder chain to hotspot residues
  - RG filter: Use --rg-filter to enable, --max-rg for fixed threshold (default: empirical formula)
        """
    )
    
    parser.add_argument('output_dir', type=str, help='RFdiffusion2 output directory (e.g., CD32_covalent_outputs)')
    parser.add_argument('--hotspots', type=str, required=False, default=None,
                       help='Hotspot residues with chain ID, format: A20,A24,A37 (chain A residues 20,24,37). Optional for non-PPI tasks.')
    parser.add_argument('--target-chain', type=str, default='B',
                       help='Binder chain ID to analyze (default: B). This is the designed chain to be screened.')
    parser.add_argument('--max-sheet', type=float, default=70.0,
                       help='Max beta sheet percentage in BINDER chain (default: 70%%)')
    parser.add_argument('--max-loop', type=float, default=30.0,
                       help='Max loop region percentage in BINDER chain (default: 30%%)')
    parser.add_argument('--min-contacts', type=int, default=5,
                       help='Min number of BINDER residues near target hotspots (default: 5)')
    parser.add_argument('--contact-distance', type=float, default=7.0,
                       help='Contact distance threshold in Angstroms (default: 7.0)')
    parser.add_argument('--max-clash', type=float, default=0.3,
                       help='Max clash ratio (Binder-receptor all heavy atoms <3A, default: 0.3 = 30%%)')
    parser.add_argument('--ligand-coverage', action='store_true',
                       help='Enable ligand coverage filter (for small molecule/peptide binders)')
    parser.add_argument('--require-ligand', action='store_true',
                       help='Only keep structures that contain ligand (filter out structures without ligand)')
    parser.add_argument('--ligand-chain', type=str, default='B',
                       help='Chain ID where ligand HETATM resides (default: B)')
    parser.add_argument('--ligand-res', type=str, default=None,
                       help='Ligand residue numbers to check coverage (e.g., 2,3 for B2 and B3). If not specified, uses all HETATM.')
    parser.add_argument('--min-coverage', type=float, default=0.7,
                       help='Min ligand coverage ratio (default: 0.7 = 70%% of ligand atoms within coverage distance)')
    parser.add_argument('--coverage-distance', type=float, default=4.0,
                       help='Ligand coverage distance threshold in Angstroms (default: 4.0)')
    parser.add_argument('--use-directional-coverage', action='store_true',
                       help='Use directional coverage (multi-direction wrapping) instead of simple proximity (RECOMMENDED for true coverage)')
    parser.add_argument('--min-directional-coverage', type=float, default=0.5,
                       help='Min directional coverage ratio (default: 0.5 = 3/6 directions, i.e., 50%% of 6 main directions)')
    parser.add_argument('--rg-filter', action='store_true',
                       help='Enable radius of gyration (RG) filter')
    parser.add_argument('--max-rg', type=float, default=None,
                       help='Max radius of gyration in Angstroms (default: use empirical formula 2.38 * L^0.365)')
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of parallel processes (default: CPU cores - 1)')
    
    args = parser.parse_args()
    
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"[ERROR] Directory not found: {output_dir}")
        sys.exit(1)
    
    
    hotspot_residues = {}  # {chain_id: [res_nums]}
    if args.hotspots:
        for hs in args.hotspots.split(','):
            hs = hs.strip()
            if len(hs) < 2:
                print(f"[WARN] Invalid hotspot format: {hs}, skipped")
                continue
            
            chain_id = hs[0]  
            try:
                res_num = int(hs[1:])  
                if chain_id not in hotspot_residues:
                    hotspot_residues[chain_id] = []
                hotspot_residues[chain_id].append(res_num)
            except ValueError:
                print(f"[WARN] Invalid hotspot format: {hs}, skipped")
                continue
    
    if not hotspot_residues:
        print(f"[WARN] No hotspot residues specified - proximity filter will be disabled")
        print(f"        Only secondary structure and clash filters will be applied")
    
    
    ligand_res_nums = None
    if args.ligand_res:
        ligand_res_nums = [int(x.strip()) for x in args.ligand_res.split(',')]
    
    
    config = {
        'binder_chain': args.target_chain,  
        'hotspot_residues': hotspot_residues,  # {chain_id: [res_nums]}
        'max_sheet_pct': args.max_sheet,
        'max_loop_pct': args.max_loop,
        'min_contacting_residues': args.min_contacts,
        'contact_distance': args.contact_distance,  
        'max_clash_ratio': args.max_clash,
        
        'ligand_coverage_enabled': args.ligand_coverage,
        'require_ligand': args.require_ligand,  
        'ligand_chain': args.ligand_chain,
        'ligand_res_nums': ligand_res_nums,
        'min_ligand_coverage': args.min_coverage,
        'ligand_coverage_distance': args.coverage_distance,
        'use_directional_coverage': args.use_directional_coverage,
        'min_directional_coverage': args.min_directional_coverage,
        
        'rg_filter_enabled': args.rg_filter,
        'max_rg': args.max_rg
    }
    
    
    screen_results_dir = output_dir / 'screen_results'
    screen_results_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("RFdiffusion2 Backbone Screening Tool - Stage 1")
    print("=" * 70)
    print(f"Input directory: {output_dir}")
    print(f"Output directory: {screen_results_dir}")
    print(f"Binder chain (to analyze): {args.target_chain}")
    print(f"Hotspot residues: {args.hotspots if args.hotspots else 'None (proximity filter disabled)'}")
    print(f"Ligand coverage: {'Enabled' if args.ligand_coverage else 'Disabled'}")
    if args.require_ligand:
        print(f" - Require ligand: YES (will filter out structures without ligand)")
    if args.ligand_coverage or args.require_ligand:
        print(f"  - Ligand chain: {args.ligand_chain}")
        print(f"  - Ligand residues: {args.ligand_res if args.ligand_res else 'All HETATM'}")
        coverage_mode = "Directional (multi-direction)" if args.use_directional_coverage else "Simple (proximity)"
        print(f"  - Coverage mode: {coverage_mode}")
        if args.use_directional_coverage:
            print(f"  - Min directional coverage: {args.min_directional_coverage*100:.0f}% ({int(args.min_directional_coverage*6)}/6 directions)")
        else:
            print(f"  - Min simple coverage: {args.min_coverage*100:.0f}% of ligand atoms")
        print(f"  - Coverage distance: {args.coverage_distance}Å")
    print(f"Screening criteria:")
    print(f"   - Secondary structure (applied to Binder chain {args.target_chain}):")
    print(f"     * Beta sheet < {args.max_sheet}%")
    print(f"     * Loop region < {args.max_loop}%")
    if hotspot_residues:
        print(f"   - Proximity (Binder chain {args.target_chain} to hotspots):")
        print(f"     * At least {args.min_contacts} residues near hotspots (<{args.contact_distance}A)")
    else:
        print(f"   - Proximity: DISABLED (no hotspots specified)")
    if args.ligand_coverage:
        print(f"   - Ligand coverage (Binder coverage of ligand):")
        print(f"     * At least {args.min_coverage*100:.0f}% of ligand atoms within {args.coverage_distance}Å")
    print(f"   - Clash detection (Binder-receptor all atoms):")
    print(f"     * Clash ratio < {args.max_clash*100:.0f}% (heavy atoms <3A)")
    if args.rg_filter:
        if args.max_rg:
            print(f"   - Radius of gyration (RG) filter:")
            print(f"     * RG < {args.max_rg:.1f}Å (fixed threshold)")
        else:
            print(f"   - Radius of gyration (RG) filter:")
            print(f"     * RG < 2.38 * L^0.365 (empirical formula)")
    print("=" * 70)
    
    
    print("\n[SCAN] Scanning PDB files...")
    pdb_files, skipped_files = collect_pdb_files(output_dir)
    print(f"[OK] Found {len(pdb_files)} valid PDB files")
    
    if skipped_files:
        print(f"[WARN] Skipped {len(skipped_files)} PDB files:")
        for skipped in skipped_files[:5]:  
            print(f"   - {skipped.name}")
        if len(skipped_files) > 5:
            print(f"   ... and {len(skipped_files) - 5} more")
    
    if not pdb_files:
        print("[ERROR] No valid PDB files found!")
        sys.exit(1)
    
    
    ss_calc = SecondaryStructureCalculator()
    if ss_calc.has_mdtraj():
        print("[OK] MDTraj detected, will use MDTraj-DSSP (accuracy 95-98%, PyMOL-equivalent)")
        print("     Method: Pure Python DSSP implementation, no external programs needed")
    elif ss_calc.has_dssp():
        print("[OK] DSSP detected, will use external DSSP program (high accuracy)")
    else:
        print("[WARN] MDTraj/DSSP not detected, will use Phi-Psi v1.6 method")
        print("       Accuracy: ~85% (may miss left-handed beta-sheets)")
        print("       Recommended: mamba install -c conda-forge mdtraj")
    
    
    n_processes = args.processes if args.processes else max(1, cpu_count() - 1)
    print(f"\n[PARALLEL] Using {n_processes} processes...\n")
    
    with Pool(processes=n_processes) as pool:
        args_list = [(pdb_file, config) for pdb_file in pdb_files]
        results = pool.map(screen_single_pdb, args_list)
    
    
    total = len(results)
    passed = sum(1 for r in results if r.passed_overall)
    passed_ss = sum(1 for r in results if r.passed_ss_filter)
    passed_prox = sum(1 for r in results if r.passed_proximity_filter)
    passed_ligand_cov = sum(1 for r in results if r.passed_ligand_coverage_filter)
    passed_dir_cov = sum(1 for r in results if r.passed_directional_coverage_filter)
    passed_clash = sum(1 for r in results if r.passed_clash_filter)
    passed_rg = sum(1 for r in results if r.passed_rg_filter)
    errors = sum(1 for r in results if r.error)
    
    print("\n" + "=" * 70)
    print("Screening Results Summary")
    print("=" * 70)
    print(f"Total processed:       {total} structures")
    print(f"Passed SS filter:      {passed_ss} ({passed_ss/total*100:.1f}%)")
    print(f"Passed proximity:      {passed_prox} ({passed_prox/total*100:.1f}%)")
    
    
    if config.get('ligand_coverage_enabled', False):
        if config.get('use_directional_coverage', False):
            print(f"Passed directional coverage: {passed_dir_cov} ({passed_dir_cov/total*100:.1f}%)  [Active filter]")
            print(f"  (Simple coverage:  {passed_ligand_cov} ({passed_ligand_cov/total*100:.1f}%)  [For reference])")
        else:
            print(f"Passed ligand coverage: {passed_ligand_cov} ({passed_ligand_cov/total*100:.1f}%)  [Active filter]")
            print(f"  (Directional coverage: {passed_dir_cov} ({passed_dir_cov/total*100:.1f}%)  [For reference])")
    
    print(f"Passed clash filter:   {passed_clash} ({passed_clash/total*100:.1f}%)")
    if config.get('rg_filter_enabled', False):
        print(f"Passed RG filter:      {passed_rg} ({passed_rg/total*100:.1f}%)")
    print(f"[PASS] Overall passed: {passed} ({passed/total*100:.1f}%)")
    if errors > 0:
        print(f"[WARN] Errors:         {errors}")
    print("=" * 70)
    
    
    csv_file = screen_results_dir / 'screening_report.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'PDB_File', 'Task_ID', 'Helix_%', 'Sheet_%', 'Loop_%',
            'N_Binder_Residues', 'N_Contacting_Residues', 'Min_Distance_A', 'Avg_Distance_A',
            'N_Ligand_Atoms', 'N_Covered_Ligand_Atoms', 'Ligand_Coverage_%', 'Avg_Ligand_Distance_A',
            'Directional_Coverage_%', 'N_Directions_Covered', 'N_Directions_Total',
            'N_Clash_Atoms', 'Clash_Ratio_%',
            'RG_A', 'RG_Threshold_A',
            'Passed_SS', 'Passed_Proximity', 'Passed_Ligand_Coverage', 'Passed_Directional_Coverage', 'Passed_Clash', 'Passed_RG', 'Passed_Overall', 'Method', 'Error'
        ])
        
        for r in results:
            writer.writerow([
                Path(r.pdb_file).name, r.task_id,
                f"{r.helix_pct:.1f}", f"{r.sheet_pct:.1f}", f"{r.loop_pct:.1f}",
                r.n_binder_residues, r.n_contacting_residues,
                f"{r.min_distance_to_hotspot:.2f}", f"{r.avg_distance_to_hotspot:.2f}",
                r.n_ligand_atoms, r.n_covered_ligand_atoms,
                f"{r.ligand_coverage_ratio*100:.1f}", f"{r.avg_ligand_distance:.2f}",
                f"{r.directional_coverage_ratio*100:.1f}", r.n_directions_covered, r.n_directions_total,
                r.n_clash_atoms, f"{r.clash_ratio*100:.1f}",
                f"{r.radius_of_gyration:.2f}", f"{r.rg_threshold:.2f}",
                r.passed_ss_filter, r.passed_proximity_filter, r.passed_ligand_coverage_filter,
                r.passed_directional_coverage_filter, r.passed_clash_filter, r.passed_rg_filter, r.passed_overall,
                r.method, r.error or ''
            ])
    
    print(f"\n[SAVE] CSV report saved: {csv_file}")
    
    
    
    
    print(f"\n[COPY] Copying passed structures to {screen_results_dir}...")
    copied = 0
    for r in results:
        if r.passed_overall:
            src_pdb = Path(r.pdb_file)
            parent = src_pdb.parent.name
            if parent.startswith("task_"):
                dst_filename = f"{r.task_id}_{src_pdb.name}"
            else:
                dst_filename = src_pdb.name
            
            if len(dst_filename) > 180:
                stem = Path(dst_filename).stem
                suffix = Path(dst_filename).suffix
                dst_filename = stem[:80] + "__" + stem[-60:] + suffix
                print(f"  [WARN] truncated long name -> {dst_filename}")
            dst_pdb = screen_results_dir / dst_filename
            shutil.copy2(src_pdb, dst_pdb)
            copied += 1
    
    print(f"[OK] Copied {copied} PDB files")
    
    
    config_file = screen_results_dir / 'screening_config.json'
    
    hotspot_str = []
    for chain_id, res_nums in hotspot_residues.items():
        for res_num in res_nums:
            hotspot_str.append(f"{chain_id}{res_num}")
    
    with open(config_file, 'w') as f:
        json.dump({
            'output_dir': str(output_dir),
            'binder_chain': args.target_chain,
            'hotspot_residues': hotspot_str,  
            'hotspot_residues_by_chain': hotspot_residues,  
            'max_sheet_pct': args.max_sheet,
            'max_loop_pct': args.max_loop,
            'min_contacting_residues': args.min_contacts,
            'contact_distance': args.contact_distance,
            'max_clash_ratio': args.max_clash,
            'rg_filter_enabled': args.rg_filter,
            'max_rg': args.max_rg,
            'total_processed': total,
            'total_passed': passed,
            'total_skipped': len(skipped_files) if skipped_files else 0,
            'pass_rate': f"{passed/total*100:.1f}%"
        }, f, indent=2)
    
    print(f"[SAVE] Config saved: {config_file}")
    
    print("\n" + "=" * 70)
    print("[COMPLETE] Screening finished!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"   1. Review report: {csv_file}")
    print(f"   2. Check passed structures: {screen_results_dir}")
    print(f"   3. Use ProteinMPNN to design sequences for {passed} structures")
    print()


if __name__ == '__main__':
    main()

