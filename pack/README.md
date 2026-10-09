---
license: mit
---

# CheMoDesign data pack

Files a CheMoDesign checkout needs besides the source repository.

- `boltz2_conf.ckpt` and `mols.tar` are the Boltz-2 confidence weights and the Boltz chemical-component library. Both are redistributed under the MIT license. Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro. See `LICENSE.boltz`.
- `custom_ligands/` holds the noncanonical components used by the manuscript examples: BPUAA, FSYH, GUANG, K3ME, KACU, NITRO, PTYU, SULF, TET2, and TYS. `SULF` here is the designed sulfonamide, not the standard CCD sulfate.

The Boltz-2 affinity checkpoint is not included. CheMoDesign does not call it.

Download this pack with:

```bash
python -m dream_boltz2.cli.setup_data
```
