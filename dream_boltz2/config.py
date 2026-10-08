"""YAML config parser for design campaigns."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
import re


@dataclass
class LossConfig:
    """LossConfig."""
    enabled: bool = False
    weight: float = 1.0
    
    steps: Optional[int] = None


@dataclass
class OptimizationConfig:
    """OptimizationConfig."""
    num_steps: int = 200
    learning_rate: float = 0.2
    early_filter_threshold: Optional[float] = None
    max_regeneration_attempts: int = 10


@dataclass 
class MSAConfig:
    """MSAConfig."""
    max_seqs: int = 2560
    fake_msa: Union[bool, List] = False
    
    @property
    def fake_msa_enabled(self) -> bool:
        """fake msa enabled."""
        if isinstance(self.fake_msa, bool):
            return self.fake_msa
        if isinstance(self.fake_msa, list):
            return len(self.fake_msa) > 0
        return False
    
    @property
    def fake_msa_has_contacts(self) -> bool:
        """fake msa has contacts."""
        return isinstance(self.fake_msa, list) and len(self.fake_msa) > 0
    
    def get_contact_pairs(self, receptor_chains: List[str], binder_chains: List[str]) -> List[Tuple[int, int]]:
        """get contact pairs."""
        if not self.fake_msa_has_contacts:
            return []
        
        pairs = []
        for item in self.fake_msa:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue
            b_chain, b_res, r_chain, r_res = str(item[0]), int(item[1]), str(item[2]), int(item[3])
            
            if b_chain not in binder_chains:
                print(f" [WARNING] fake_msa contactvsin binder chain '{b_chain}' notin binder_chains {binder_chains} in,skip")
                continue
            if r_chain not in receptor_chains:
                print(f" [WARNING] fake_msa contactvsin receptor chain '{r_chain}' notin receptor_chains {receptor_chains} in,skip")
                continue
            # 1-based -> 0-based
            pairs.append((r_res - 1, b_res - 1))
        return pairs


@dataclass
class OutputConfig:
    """OutputConfig."""
    dir: str = "./output"
    save_initial: bool = True
    save_trajectory: bool = False


@dataclass
class ProtectedRegions:
    """ProtectedRegions."""
    
    protected_modifications: List[str] = field(default_factory=list)
    
    protected_ligands: List[str] = field(default_factory=list)
    
    protected_residues: List[str] = field(default_factory=list)


@dataclass
class SequenceItem:
    """SequenceItem."""
    chain_id: str
    entity_type: str  # protein, ligand, dna, rna
    sequence: Optional[str] = None
    smiles: Optional[str] = None
    ccd: Optional[str] = None
    msa: Optional[str] = None
    modifications: List[Dict] = field(default_factory=list)


@dataclass
class Constraint:
    """Constraint."""
    constraint_type: str  # bond, pocket, contact
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Template:
    """Template."""
    path: str                                          
    file_type: str = "pdb"                             
    chain_ids: Optional[List[str]] = None              
    template_ids: Optional[List[str]] = None           
    force: bool = False                                
    threshold: float = 3.0                             


@dataclass
class DREAMConfig:
    """DREAMConfig."""
    
    version: int = 1
    sequences: List[SequenceItem] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    templates: List[Template] = field(default_factory=list)
    
    
    receptor_chains: List[str] = field(default_factory=list)
    binder_chains: List[str] = field(default_factory=list)
    design_type: str = "binder"  # binder, nanobody, fab, minibinder
    
    
    creativity: float = 0.0
    num_design_rounds: int = 1
    samples_per_round: int = 1
    
    
    protected_regions: ProtectedRegions = field(default_factory=ProtectedRegions)
    
    
    hotspot_indices: List[int] = field(default_factory=list)
    
    
    
    
    scaling_residues: List[str] = field(default_factory=list)
    
    
    
    
    
    
    scaling_residues_only: bool = False
    
    
    cdr_regions: Dict[str, str] = field(default_factory=dict)
    
    
    rg_loss: LossConfig = field(default_factory=LossConfig)
    helix_loss: LossConfig = field(default_factory=LossConfig)
    hotspot_loss: LossConfig = field(default_factory=LossConfig)
    distogram_penalty: LossConfig = field(default_factory=LossConfig)
    
    
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    
    
    msa: MSAConfig = field(default_factory=MSAConfig)
    
    
    output: OutputConfig = field(default_factory=OutputConfig)
    
    
    raw_boltz_schema: Dict = field(default_factory=dict)
    
    def get_receptor_msa(self) -> Optional[str]:
        """get receptor msa."""
        for seq in self.sequences:
            if seq.chain_id in self.receptor_chains and seq.msa:
                return seq.msa
        return None
    
    def get_binder_sequence(self) -> Optional[str]:
        """get binder sequence."""
        for seq in self.sequences:
            if seq.chain_id in self.binder_chains and seq.sequence:
                return seq.sequence
        return None
    
    def get_receptor_sequence(self) -> Optional[str]:
        """get receptor sequence."""
        for seq in self.sequences:
            if seq.chain_id in self.receptor_chains and seq.sequence:
                return seq.sequence
        return None
    
    def get_ligands(self) -> List[SequenceItem]:
        """get ligands."""
        return [seq for seq in self.sequences if seq.entity_type == "ligand"]
    
    def get_modifications(self) -> List[Tuple[str, int, str]]:
        """get modifications."""
        mods = []
        for seq in self.sequences:
            for mod in seq.modifications:
                mods.append((seq.chain_id, mod['position'], mod['ccd']))
        return mods
    
    def get_protected_modification_positions(self) -> List[Tuple[str, int]]:
        """get protected modification positions."""
        protected = []
        for seq in self.sequences:
            for mod in seq.modifications:
                if mod['ccd'] in self.protected_regions.protected_modifications:
                    protected.append((seq.chain_id, mod['position']))
        return protected
    
    def parse_cdr_regions(self) -> List[Tuple[int, int]]:
        """parse cdr regions."""
        regions = []
        for region_name, region_str in self.cdr_regions.items():
            if ':' in region_str:
                start, end = region_str.split(':')
                regions.append((int(start), int(end)))
            elif '-' in region_str:
                start, end = region_str.split('-')
                regions.append((int(start), int(end)))
        return regions
    
    def get_first_template(self) -> Optional['Template']:
        """get first template."""
        if self.templates:
            return self.templates[0]
        return None
    
    def to_cli_args(self) -> Dict[str, Any]:
        """to cli args."""
        args = {
            
            'receptor_chain': self.receptor_chains[0] if self.receptor_chains else None,
            'binder_template': self.get_binder_sequence(),
            
            
            'creativity': self.creativity,
            'num_design_rounds': self.num_design_rounds,
            'samples_per_round': self.samples_per_round,
            
            
            'hotspot_indices': ','.join(map(str, self.hotspot_indices)) if self.hotspot_indices else None,
            
            
            'use_rg_loss': self.rg_loss.enabled,
            'use_helix_loss': self.helix_loss.enabled,
            'use_hotspot_loss': self.hotspot_loss.enabled,
            'rg_weight': self.rg_loss.weight,
            'helix_weight': self.helix_loss.weight,
            'hotspot_weight': self.hotspot_loss.weight,
            'use_distogram_penalty': self.distogram_penalty.enabled,
            'distogram_penalty_weight': self.distogram_penalty.weight,
            'distogram_penalty_steps': self.distogram_penalty.steps,
            
            
            'num_steps': self.optimization.num_steps,
            'lr': self.optimization.learning_rate,
            'early_filter_threshold': self.optimization.early_filter_threshold,
            'max_regeneration_attempts': self.optimization.max_regeneration_attempts,
            
            # MSA
            'max_msa_seqs': self.msa.max_seqs,
            'receptor_msa': self.get_receptor_msa(),
            'fake_msa': self.msa.fake_msa,
            
            # Template
            'template_file': self.templates[0].path if self.templates else None,
            'template_chain_id': self.templates[0].chain_ids[0] if self.templates and self.templates[0].chain_ids else None,
            'template_force': self.templates[0].force if self.templates else False,
            'template_threshold': self.templates[0].threshold if self.templates else 3.0,
            
            
            'output_dir': self.output.dir,
        }
        
        
        if self.design_type == 'nanobody':
            args['nanobody'] = True
        elif self.design_type == 'fab':
            args['fab'] = True
        
        
        if self.cdr_regions:
            cdr_str = ','.join([f"{v}" for v in self.cdr_regions.values()])
            args['cdr_regions'] = cdr_str
        
        return args


def resolve_msa_path(msa: Optional[str], yaml_path: Path) -> Optional[str]:
    """Resolve an MSA path relative to the YAML file when that file exists."""
    if not isinstance(msa, str) or msa.strip() in ("", "empty"):
        return msa
    path = Path(msa)
    if path.is_absolute():
        return str(path)
    beside = (yaml_path.parent / path).resolve()
    if beside.is_file():
        return str(beside)
    return msa


def parse_sequence_item(item: Dict) -> SequenceItem:
    """parse sequence item."""
    entity_type = next(iter(item.keys())).lower()
    data = item[entity_type]
    
    chain_ids = data.get('id', [])
    if isinstance(chain_ids, str):
        chain_ids = [chain_ids]
    
    
    chain_id = chain_ids[0] if chain_ids else None
    
    return SequenceItem(
        chain_id=chain_id,
        entity_type=entity_type,
        sequence=data.get('sequence'),
        smiles=data.get('smiles'),
        ccd=data.get('ccd'),
        msa=data.get('msa'),
        modifications=data.get('modifications', []),
    )


def parse_constraint(item: Dict) -> Constraint:
    """parse constraint."""
    constraint_type = next(iter(item.keys())).lower()
    return Constraint(
        constraint_type=constraint_type,
        data=item[constraint_type],
    )


def parse_template(item: Dict) -> Template:
    """parse template."""
    
    if 'cif' in item:
        file_type = "cif"
        path = item['cif']
    elif 'pdb' in item:
        file_type = "pdb"
        path = item['pdb']
    else:
        file_type = "pdb"
        path = item.get('path', '')
    
    
    chain_ids = item.get('chain_id') or item.get('ids')
    if isinstance(chain_ids, str):
        chain_ids = [chain_ids]
    
    
    template_ids = item.get('template_id')
    if isinstance(template_ids, str):
        template_ids = [template_ids]
    
    return Template(
        path=path,
        file_type=file_type,
        chain_ids=chain_ids,
        template_ids=template_ids,
        force=item.get('force', False),
        threshold=item.get('threshold', 3.0),
    )


def parse_loss_config(data: Optional[Dict]) -> LossConfig:
    """parse loss config."""
    if data is None:
        return LossConfig()
    return LossConfig(
        enabled=data.get('enabled', False),
        weight=data.get('weight', 1.0),
        steps=data.get('steps'),
    )


def parse_dream_config(path: Union[str, Path]) -> DREAMConfig:
    """parse dream config."""
    path = Path(path)
    
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    config = DREAMConfig()
    config.raw_boltz_schema = data
    
    
    config.version = data.get('version', 1)
    
    
    for item in data.get('sequences', []):
        seq = parse_sequence_item(item)
        seq.msa = resolve_msa_path(seq.msa, path)
        config.sequences.append(seq)
    
    
    for item in data.get('constraints', []):
        config.constraints.append(parse_constraint(item))
    
    
    for item in data.get('templates', []):
        config.templates.append(parse_template(item))
    
    
    dream = data.get('dream', {})
    
    config.receptor_chains = dream.get('receptor_chains', [])
    config.binder_chains = dream.get('binder_chains', [])
    config.design_type = dream.get('design_type', 'binder')
    
    config.creativity = dream.get('creativity', 0.0)
    config.num_design_rounds = dream.get('num_design_rounds', 1)
    config.samples_per_round = dream.get('samples_per_round', 1)
    
    
    protected = dream.get('protected_regions', {})
    config.protected_regions = ProtectedRegions(
        protected_modifications=protected.get('protected_modifications', []),
        protected_ligands=protected.get('protected_ligands', []),
        protected_residues=protected.get('protected_residues', []),
    )
    
    
    config.hotspot_indices = dream.get('hotspot_indices', [])
    
    
    
    config.scaling_residues = dream.get('scaling_residues', [])
    
    
    config.scaling_residues_only = dream.get('scaling_residues_only', False)
    
    
    config.cdr_regions = dream.get('cdr_regions', {})
    
    
    losses = dream.get('losses', {})
    config.rg_loss = parse_loss_config(losses.get('rg_loss'))
    config.helix_loss = parse_loss_config(losses.get('helix_loss'))
    config.hotspot_loss = parse_loss_config(losses.get('hotspot_loss'))
    config.distogram_penalty = parse_loss_config(losses.get('distogram_penalty'))
    
    
    opt = dream.get('optimization', {})
    config.optimization = OptimizationConfig(
        num_steps=opt.get('num_steps', 200),
        learning_rate=opt.get('learning_rate', 0.2),
        early_filter_threshold=opt.get('early_filter_threshold'),
        max_regeneration_attempts=opt.get('max_regeneration_attempts', 10),
    )
    
    
    msa = dream.get('msa', {})
    config.msa = MSAConfig(
        max_seqs=msa.get('max_seqs', 2560),
        fake_msa=msa.get('fake_msa', False),
    )
    
    
    output = dream.get('output', {})
    config.output = OutputConfig(
        dir=output.get('dir', './output'),
        save_initial=output.get('save_initial', True),
        save_trajectory=output.get('save_trajectory', False),
    )
    
    return config


def validate_config(config: DREAMConfig) -> List[str]:
    """validate config."""
    errors = []
    
    
    if not config.sequences:
        errors.append('configfilerequiredcontainsasequence')
    
    if not config.receptor_chains:
        errors.append('required receptor_chains')
    
    if not config.binder_chains:
        errors.append('required binder_chains')
    
    
    chain_ids = {seq.chain_id for seq in config.sequences}
    for rc in config.receptor_chains:
        if rc not in chain_ids:
            errors.append(f"receptor_chain '{rc}' in sequences innotto")
    for bc in config.binder_chains:
        if bc not in chain_ids:
            errors.append(f"binder_chain '{bc}' in sequences innotto")
    
    
    if not 0 <= config.creativity <= 1:
        errors.append(f"creativity requiredin [0, 1] ,current: {config.creativity}")
    
    
    valid_types = {'binder', 'nanobody', 'fab', 'minibinder'}
    if config.design_type not in valid_types:
        errors.append(f"design_type required {valid_types} ,current: {config.design_type}")
    
    return errors


if __name__ == "__main__":
    
    import sys
    
    if len(sys.argv) < 2:
        print('use: python config_parser.py <config.yaml>')
        sys.exit(1)
    
    config = parse_dream_config(sys.argv[1])
    errors = validate_config(config)
    
    if errors:
        print(' configfailed:')
        for err in errors:
            print(f"  - {err}")
    else:
        print(' config')
        print(f"  Receptor chains: {config.receptor_chains}")
        print(f"  Binder chains: {config.binder_chains}")
        print(f"  Design type: {config.design_type}")
        print(f"  Creativity: {config.creativity}")
        print(f"  Design rounds: {config.num_design_rounds}")
        print(f"  Samples per round: {config.samples_per_round}")
        print(f"  Protected modifications: {config.protected_regions.protected_modifications}")
        print(f"  Protected ligands: {config.protected_regions.protected_ligands}")

