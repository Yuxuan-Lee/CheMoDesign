# DREAM-Boltz2 harness

This program is still under construction.

## Contributors

1. [Yuxuan Li](https://github.com/Yuxuan-Lee)
2. [Fred Jing](https://github.com/Fred-Jing)
3. [Yeyu Su](https://github.com/YeyuSu)
4. [Cursor Agent](https://github.com/cursor)

CheMoDesign backbone generation and sequence-conditioned refinement on frozen Boltz-2 (v2.2.0). Model weights stay frozen. Pair entries are scaled as `z_ij -> (1 - alpha) z_ij`. Receptor-receptor pairs and functional anchors (hotspots, and every pair that touches an atom-level chemical component) are left unscaled and are not updated. A geometric loss (radius of gyration, helix content, hotspot contacts) is backpropagated through one denoising step.

`alpha` is the YAML field `dream.creativity`.

## Layout

```
dream_boltz2/
  cli/design.py          dream: optimize s,z and sample backbones
  cli/wake.py            wake: LigandMPNN, fake paired MSA, template, noise sigma=20
  cli/screen.py          compactness, secondary structure, clash, hotspot filters
  cli/sequences.py       LigandMPNN sequence design
  cli/filter_designs.py  keep top sequences by confidence
  cli/af3_json.py        AlphaFold 3 input JSON
  cli/cluster_tm.py      complete-linkage binder clustering (TM >= 0.6)
  model/                 diffusion and Pairformer wrappers
  features/              receptor-binder features, ligands, covalent bonds
  losses/                Rg, helix, hotspot
  msa/                   synthetic paired MSA
  covalent/              deferred ligand insertion (FSY, 3-NT, CMN)
  sequence/              LigandMPNN runner (ligand_mpnn, soluble_mpnn, protein_mpnn)
examples/               campaign YAML files used in the manuscript
examples/msa/           receptor a3m files, referenced from each YAML
examples/ligands/       custom CCD pickles (KACU, K3ME, PTYU, TYS, TET2, BPUAA, FSYH, NITRO, SULF, GUANG)
```

## Setup

Install Boltz 2.2.0 (Python 3.10–3.12), then download the weights and molecule library. The data pack also contains the noncanonical ligands used in `examples/` (BPUAA, FSYH, GUANG, K3ME, KACU, NITRO, PTYU, SULF, TET2, TYS). LigandMPNN and its three checkpoints are already in `third_party/LigandMPNN`.

```bash
pip install "boltz==2.2.0" prody mdtraj
pip install -e .
python -m dream_boltz2.cli.setup_data
```

`setup_data` downloads into this repository, and the program reads those paths:

```
third_party/boltz/boltz2_conf.ckpt
third_party/boltz/mols/
third_party/LigandMPNN/model_params/
examples/ligands/
examples/msa/
```

Example ligands are copied into `third_party/boltz/mols/` after download. Boltz-2 does not need `ccd.pkl`. `BOLTZ_SRC` is only for a Boltz checkout that is not installed. `--model_type protein_mpnn` or `soluble_mpnn` selects the weights already in `third_party/LigandMPNN`.

Backbone screening assigns helix, sheet, and loop with MDTraj's DSSP (the same assignment used for the manuscript filters). MDTraj 1.10 and later carry that DSSP in Python, so a separate `dssp` or `mkdssp` binary is not required.

MSA paths in the YAML files are relative to the YAML (`msa/<target>.a3m`). The parser resolves them from the file location, so the examples run from this directory without the original cluster paths. Custom ligands in `examples/ligands/` are loaded ahead of the Boltz molecule cache.

## Pipeline

One campaign command runs the manuscript order: dream, screen, wake (sigma 20), screen again, then a final LigandMPNN pass and confidence filter. A YAML with `covalent_mode: deferred` adds the second covalent wake and masks the attachment glycine as X. `--skip-wake` is the thrombin path (screen, then LigandMPNN).

```bash
export PYTHONPATH=/path/to/DREAM-Boltz2-harness:${PYTHONPATH}

./campaign.sh --config examples/pd_l1.yaml
./campaign.sh --config examples/trop2_fluorosulfate.yaml --num-samples 2 --top-n 2
./campaign.sh --config examples/thrombin_sulfotyrosine.yaml --skip-wake
./campaign.sh --config examples/pd_l1.yaml --from wake --to screen_wake --dry-run
```

Defaults follow the shared campaign scripts: screen Rg 14, loop 40, sheet 95, 3 contacts, clash 0.1; wake warm-start sigma 20; final LigandMPNN 50 sequences at T=0.2, keep 4. Override any of them on the command line. `python -m dream_boltz2.cli.campaign --help` lists the stages.

The same steps can still be called one at a time (`cli.design`, `cli.screen`, `cli.wake`, `cli.sequences`). Waking also samples a sequence internally so the relaxation is sequence-conditioned. The final LigandMPNN stage designs sequences on the structures that passed the last screen.

## Examples

`examples/` holds the five-target benchmark inputs (`pd_l1.yaml`, `il2ra.yaml`, `il7ra.yaml`, `trop2.yaml`, `b7h3.yaml`), the three PTM recovery inputs (`brd4_acetyllysine.yaml`, `cbx8_trimethyllysine.yaml`, `src_phosphotyrosine.yaml`), and the prospective campaigns (`thrombin_sulfotyrosine.yaml`, `il7ra_tetrazine.yaml`, `zspa1_pbpa.yaml`, `trop2_fluorosulfate.yaml`, `her2_nitrotyrosine.yaml`, `caix_sulfonamide.yaml`, `pd_l1_cmn.yaml`).
