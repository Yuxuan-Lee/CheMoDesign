"""Write a Boltz-2 template CIF from a dreamed backbone."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple



AA3_TO_AA1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    
    "SEP": "S",
    "TPO": "T",
    "PTR": "Y",
    "MSE": "M",
    "HYP": "P",
    "MLY": "K",
    "MLZ": "K",
    "M3L": "K",
    "ALY": "K",
    "2MR": "R",
    "MME": "M",
    "MHS": "H",
    "DAL": "A",
    "DAR": "R",
    "DSN": "N",
    "DSP": "D",
    "DCY": "C",
    "DGL": "E",
    "DGN": "Q",
    "DHI": "H",
    "DIL": "I",
    "DLE": "L",
    "DLY": "K",
    "MED": "M",
    "DPN": "F",
    "DPR": "P",
    "DSE": "S",
    "DTH": "T",
    "DTR": "W",
    "DTY": "Y",
    "DVA": "V",
    "AIB": "A",
    "NLE": "L",
    "NVA": "V",
    "ORN": "K",
    "CIR": "R",
    "SAR": "G",
    "SEC": "C",
    "UNK": "X",
    "PYL": "O",
}


def _three_to_one_letter(three_letter_code: str) -> str:
    """ three to one letter."""
    return AA3_TO_AA1.get(three_letter_code.upper(), "X")


@dataclass(frozen=True)
class TemplateCifResult:
    output_cif: Path
    chain_map_applied: Dict[str, str]
    kept_chains: list[str]


def _require_gemmi():
    try:
        import gemmi  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            'need gemmi PDB->CIF convert.'
            '(Boltz2 gemmi parse mmCIF)'
        ) from e


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except Exception as e:  # pragma: no cover
        raise ImportError('need pyyaml read YAML config.') from e
    with path.open("r", encoding="utf-8", errors="replace") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML not( dict): {path}")
    return data


def extract_roles_from_yaml(
    yaml_path: str | Path,
) -> Tuple[list[str], list[str], list[str], dict]:
    """extract roles from yaml."""
    yaml_path = Path(yaml_path)
    cfg = _load_yaml(yaml_path)

    dream = cfg.get("dream", {}) if isinstance(cfg.get("dream", {}), dict) else {}
    receptor = dream.get("receptor_chains", [])
    binder = dream.get("binder_chains", [])

    if isinstance(receptor, str):
        receptor = [receptor]
    if isinstance(binder, str):
        binder = [binder]
    if not isinstance(receptor, list) or not isinstance(binder, list):
        raise ValueError('dream.receptor_chains / dream.binder_chains required list or str')

    # ligands from sequences
    ligands: list[str] = []
    seqs = cfg.get("sequences", [])
    if isinstance(seqs, list):
        for item in seqs:
            if not isinstance(item, dict) or not item:
                continue
            entity_type = next(iter(item.keys())).lower()
            if entity_type != "ligand":
                continue
            lig = item.get("ligand", {})
            if not isinstance(lig, dict):
                continue
            ids = lig.get("id", [])
            if isinstance(ids, str):
                ligands.append(ids)
            elif isinstance(ids, list):
                ligands.extend(str(x) for x in ids)

    pdb_chain_map = dream.get("pdb_chain_map", {})
    if pdb_chain_map is None:
        pdb_chain_map = {}
    if not isinstance(pdb_chain_map, dict):
        raise ValueError('dream.pdb_chain_map required dict(YAMLchainID -> inputPDBchainID)')

    receptor = [str(x) for x in receptor]
    binder = [str(x) for x in binder]
    ligands = [str(x) for x in ligands]
    pdb_chain_map = {str(k): str(v) for k, v in pdb_chain_map.items()}

    return receptor, binder, ligands, pdb_chain_map


def pdb_to_boltz2_template_cif(
    input_pdb: str | Path,
    output_cif: str | Path,
    *,
    chain_id_map: Optional[Dict[str, str]] = None,
    keep_only_chains: Optional[Iterable[str]] = None,
    force_single_subchain_per_chain: bool = True,
) -> TemplateCifResult:
    """pdb to boltz2 template cif."""
    _require_gemmi()
    import gemmi

    input_pdb = Path(input_pdb)
    output_cif = Path(output_cif)
    output_cif.parent.mkdir(parents=True, exist_ok=True)

    structure = gemmi.read_structure(str(input_pdb))
    structure.setup_entities()

    keep_set = set(keep_only_chains) if keep_only_chains is not None else None
    applied_map: Dict[str, str] = {}

    
    model = structure[0]

    
    if keep_set is not None:
        
        for i in reversed(range(len(model))):
            if model[i].name not in keep_set:
                del model[i]

    
    if chain_id_map:
        for chain in model:
            old = chain.name
            if old in chain_id_map:
                new = chain_id_map[old]
                chain.name = new
                applied_map[old] = new

    
    if force_single_subchain_per_chain:
        for chain in model:
            target_subchain = f"{chain.name}1"
            for res in chain:
                res.subchain = target_subchain

        
        
        for entity in structure.entities:
            new_subchains = []
            for sub in entity.subchains:
                
                
                
                
                if sub:
                    new_subchains.append(sub)
            
            existing = {ch.subchain_id() for ch in model.subchains()}
            filtered = [s for s in dict.fromkeys(new_subchains) if s in existing]
            entity.subchains = filtered

        
        all_subchains = [ch.subchain_id() for ch in model.subchains()]
        for entity in structure.entities:
            for s in all_subchains:
                if s not in entity.subchains:
                    entity.subchains.append(s)

    
    
    
    subchain_to_entity: dict[str, str] = {}
    for entity in structure.entities:
        entity_id = entity.name
        for subchain_id in entity.subchains:
            subchain_to_entity[subchain_id] = entity_id
    
    
    entity_to_seq3: dict[str, list[str]] = {}
    for chain in model:
        chain_name = chain.name
        
        subchain_id = None
        for subchain in model.subchains():
            if subchain.subchain_id().startswith(chain_name):
                subchain_id = subchain.subchain_id()
                break
        if subchain_id is None:
            subchain_id = f"{chain_name}1" if chain_name else "?"
        
        
        entity_id = subchain_to_entity.get(subchain_id, chain_name[0] if chain_name else "?")
        
        if entity_id not in entity_to_seq3:
            entity_to_seq3[entity_id] = []
        
        
        residues = sorted(chain, key=lambda r: (r.seqid.num, r.seqid.icode or ""))
        for residue in residues:
            three_letter = residue.name
            if three_letter and three_letter.upper() in AA3_TO_AA1:
                entity_to_seq3[entity_id].append(three_letter.upper())
    
    
    for entity in structure.entities:
        entity_id = entity.name
        if entity_id in entity_to_seq3:
            entity.full_sequence = entity_to_seq3[entity_id]

    
    doc = structure.make_mmcif_document()

    
    
    
    try:
        block = doc[0]

        # struct_asym: A1 -> entity A, B1 -> entity B
        col = block.find_loop("_struct_asym.id")
        if col is not None:
            loop = col.get_loop()
            tags = list(loop.tags)
            if "_struct_asym.id" in tags:
                col_id = tags.index("_struct_asym.id")
                if "_struct_asym.entity_id" not in tags:
                    loop.add_columns(["_struct_asym.entity_id"])
                    tags = list(loop.tags)
                col_entity = tags.index("_struct_asym.entity_id")

                for i in range(loop.length()):
                    asym = str(loop[i, col_id])
                    entity_id = asym[0] if asym else "1"
                    loop[i, col_entity] = entity_id

        
        
        chain_to_entity: dict[str, str] = {}
        asym_col = block.find_loop("_struct_asym.id")
        if asym_col is not None:
            asym_loop = asym_col.get_loop()
            asym_tags = list(asym_loop.tags)
            asym_id_col = asym_tags.index("_struct_asym.id") if "_struct_asym.id" in asym_tags else -1
            asym_entity_col = asym_tags.index("_struct_asym.entity_id") if "_struct_asym.entity_id" in asym_tags else -1
            if asym_id_col != -1 and asym_entity_col != -1:
                for i in range(asym_loop.length()):
                    asym_id = str(asym_loop[i, asym_id_col])  
                    entity_id = str(asym_loop[i, asym_entity_col])  
                    
                    chain_name = asym_id[0] if asym_id else ""
                    if chain_name:
                        chain_to_entity[chain_name] = entity_id

        
        seq_by_entity: dict[str, list[str]] = {}  # entity_id -> [1-letter codes]
        seq3_by_entity: dict[str, list[str]] = {}  # entity_id -> [3-letter codes]
        entity_chains: dict[str, list[str]] = {}  # entity_id -> [chain_names]
        
        for chain in model:
            chain_name = chain.name
            
            entity_id = chain_to_entity.get(chain_name, chain_name[0] if chain_name else "?")
            
            if entity_id not in seq_by_entity:
                seq_by_entity[entity_id] = []
                seq3_by_entity[entity_id] = []
                entity_chains[entity_id] = []
            if chain_name not in entity_chains[entity_id]:
                entity_chains[entity_id].append(chain_name)
            
            
            residues = sorted(chain, key=lambda r: (r.seqid.num, r.seqid.icode or ""))
            for residue in residues:
                
                three_letter = residue.name
                if not three_letter:
                    continue
                if three_letter.upper() not in AA3_TO_AA1:
                    
                    continue
                one_letter = _three_to_one_letter(three_letter)
                seq_by_entity[entity_id].append(one_letter)
                seq3_by_entity[entity_id].append(three_letter.upper())
        
        
        seq_str_by_entity: dict[str, str] = {}
        for ent_id, one_letter_list in seq_by_entity.items():
            seq_str = "".join(one_letter_list)
            seq_str_by_entity[ent_id] = seq_str
            
            
            expected_len = len(one_letter_list)
            actual_len = len(seq_str)
            chains_str = ", ".join(entity_chains.get(ent_id, []))
            
            if actual_len != expected_len:
                raise ValueError(
                    f"Entity {ent_id} (chain: {chains_str}) sequenceerror！\n"
                    f" sequencelength: {actual_len}\n"
                    f" residuelistlength: {expected_len}\n"
                    f"thisnotthis,check."
                )
            
            if actual_len == 0:
                raise ValueError(
                    f"Entity {ent_id} (chain: {chains_str}) noextracttoresidue！\n"
                    f"thisas:\n"
                    f" 1) chain {chains_str} asemptyorcontainsresidue\n"
                    f" 2) PDB file,gemmi noparse\n"
                    f":check PDB file,chain {chains_str} containsresidue."
                )

        
        poly_col = block.find_loop("_entity_poly.entity_id")
        if poly_col is not None:
            poly_loop = poly_col.get_loop()
            ptags = list(poly_loop.tags)
            c_entity = (
                ptags.index("_entity_poly.entity_id") if "_entity_poly.entity_id" in ptags else -1
            )
            c_strand = (
                ptags.index("_entity_poly.pdbx_strand_id")
                if "_entity_poly.pdbx_strand_id" in ptags
                else -1
            )
            c_seq = (
                ptags.index("_entity_poly.pdbx_seq_one_letter_code")
                if "_entity_poly.pdbx_seq_one_letter_code" in ptags
                else -1
            )

            for i in range(poly_loop.length()):
                ent_id = str(poly_loop[i, c_entity]) if c_entity != -1 else ""
                if c_strand != -1:
                    poly_loop[i, c_strand] = ent_id or "?"
                if c_seq != -1 and ent_id in seq_str_by_entity:
                    poly_loop[i, c_seq] = seq_str_by_entity[ent_id]

        
        
        
        
        
        poly_seq_col = block.find_loop("_entity_poly_seq.entity_id")
        if poly_seq_col is not None:
            
            block.remove(poly_seq_col.tag)
        
        
        if not seq3_by_entity:
            raise ValueError('seq3_by_entity asempty,nowrite _entity_poly_seq ！')
        
        
        
        
        try:
            old_loop = block.find_loop("_entity_poly_seq.entity_id")
            if old_loop is not None:
                block.remove(old_loop.tag)
        except Exception:
            pass
        
        
        loop_tags = [
            "_entity_poly_seq.entity_id",
            "_entity_poly_seq.hetero",  
            "_entity_poly_seq.mon_id",
            "_entity_poly_seq.num",
        ]
        poly_seq_loop = block.init_loop("_entity_poly_seq.", loop_tags)
        
        
        
        for ent_id, three_letter_list in seq3_by_entity.items():
            if not three_letter_list:
                continue  
            for seq_num, mon_id in enumerate(three_letter_list, start=1):
                
                poly_seq_loop.add_row([ent_id, "n", mon_id, str(seq_num)])
        
        
        final_tags = list(poly_seq_loop.tags)
        if "_entity_poly_seq.hetero" not in final_tags:
            
            
            import warnings
            warnings.warn(
                'gemmi _entity_poly_seq.hetero ,inwritefile.'
            )

        
        ent_col = block.find_loop("_entity.id")
        if ent_col is not None:
            ent_loop = ent_col.get_loop()
            etags = list(ent_loop.tags)
            c_type = etags.index("_entity.type") if "_entity.type" in etags else -1
            for i in range(ent_loop.length()):
                if c_type != -1 and (str(ent_loop[i, c_type]) in {"?", ".", ""}):
                    ent_loop[i, c_type] = "polymer"

    except Exception:
        
        pass

    
    
    
    # 1. _struct_asym.id: "A1" -> "A", "B1" -> "B"
    # 2. _atom_site.label_asym_id: "A1" -> "A", "B1" -> "B"
    try:
        block = doc[0]
        
        
        
        
        subchain_to_chain: dict[str, str] = {}
        for chain in model:
            chain_name = chain.name  
            
            subchain_id = f"{chain_name}1"  
            subchain_to_chain[subchain_id] = chain_name
        
        
        asym_col = block.find_loop("_struct_asym.id")
        if asym_col is not None:
            asym_loop = asym_col.get_loop()
            asym_tags = list(asym_loop.tags)
            asym_id_col = asym_tags.index("_struct_asym.id") if "_struct_asym.id" in asym_tags else -1
            if asym_id_col != -1:
                for i in range(asym_loop.length()):
                    old_asym_id = str(asym_loop[i, asym_id_col])  
                    new_asym_id = subchain_to_chain.get(old_asym_id, old_asym_id)
                    if new_asym_id != old_asym_id:
                        asym_loop[i, asym_id_col] = new_asym_id
        
        
        atom_site_col = block.find_loop("_atom_site.label_asym_id")
        if atom_site_col is not None:
            atom_site_loop = atom_site_col.get_loop()
            atom_site_tags = list(atom_site_loop.tags)
            label_asym_col = atom_site_tags.index("_atom_site.label_asym_id") if "_atom_site.label_asym_id" in atom_site_tags else -1
            if label_asym_col != -1:
                for i in range(atom_site_loop.length()):
                    old_asym_id = str(atom_site_loop[i, label_asym_col])  
                    new_asym_id = subchain_to_chain.get(old_asym_id, old_asym_id)
                    if new_asym_id != old_asym_id:
                        atom_site_loop[i, label_asym_col] = new_asym_id
    except Exception as e:
        import warnings
        warnings.warn(f"nochain ID: {e}")
    
    doc.write_file(str(output_cif))
    
    
    
    try:
        with open(output_cif, 'r', encoding='utf-8') as f:
            cif_content = f.read()
        
        
        if '_entity_poly_seq.entity_id' in cif_content and '_entity_poly_seq.hetero' not in cif_content:
            import re
            
            pattern = r'(loop_\s+_entity_poly_seq\.entity_id\s+_entity_poly_seq\.num\s+_entity_poly_seq\.mon_id\s+)(.*?)(?=\nloop_|\n#|\Z)'
            match = re.search(pattern, cif_content, re.DOTALL | re.MULTILINE)
            if match:
                
                old_header = match.group(1)
                old_data = match.group(2)
                new_header = 'loop_\n_entity_poly_seq.entity_id\n_entity_poly_seq.hetero\n_entity_poly_seq.mon_id\n_entity_poly_seq.num\n'
                
                lines = [l.strip() for l in old_data.strip().split('\n') if l.strip() and not l.strip().startswith('#')]
                new_lines = []
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 3:  # entity_id num mon_id
                        
                        new_lines.append(f"{parts[0]} n {parts[2]} {parts[1]}")
                    else:
                        new_lines.append(line)
                new_data = '\n'.join(new_lines) + '\n'
                cif_content = cif_content[:match.start()] + new_header + new_data + cif_content[match.end():]
                
                
                with open(output_cif, 'w', encoding='utf-8') as f:
                    f.write(cif_content)
    except Exception as e:
        
        import warnings
        warnings.warn(f"no _entity_poly_seq.hetero: {e}")

    kept = [c.name for c in model]
    return TemplateCifResult(
        output_cif=output_cif,
        chain_map_applied=applied_map,
        kept_chains=kept,
    )


def pdb_to_boltz2_template_cif_from_yaml(
    *,
    input_pdb: str | Path,
    project_yaml: str | Path,
    output_cif: str | Path,
    
    normalize_output_chain_ids: bool = False,
    binder_out: str = "A",
    receptor_out: str = "B",
) -> TemplateCifResult:
    """pdb to boltz2 template cif from yaml."""
    receptor_ids, binder_ids, ligand_ids, yaml_to_pdb = extract_roles_from_yaml(project_yaml)

    
    
    
    
    
    wanted_yaml = binder_ids + receptor_ids
    wanted_pdb = [yaml_to_pdb.get(cid, cid) for cid in wanted_yaml]

    
    if normalize_output_chain_ids:
        
        chain_map: Dict[str, str] = {}
        for y in binder_ids:
            chain_map[yaml_to_pdb.get(y, y)] = binder_out
        for y in receptor_ids:
            chain_map[yaml_to_pdb.get(y, y)] = receptor_out
    else:
        chain_map = {yaml_to_pdb.get(y, y): y for y in wanted_yaml}

    return pdb_to_boltz2_template_cif(
        input_pdb=input_pdb,
        output_cif=output_cif,
        chain_id_map=chain_map,
        keep_only_chains=wanted_pdb,
        force_single_subchain_per_chain=True,
    )


