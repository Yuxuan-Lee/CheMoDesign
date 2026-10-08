"""Cache token embeddings from the Boltz-2 input embedder.

Soft-token chemistry mixing is not part of the paper design path.
Calling the mixing helpers raises NotImplementedError.
"""

from __future__ import annotations


def install_embedder_capture(embedder) -> None:
    """Cache the atom-attention token vector on the embedder during forward."""
    if embedder is None or getattr(embedder, "_dream_a_capture_installed", False):
        return
    orig_attn = embedder.atom_attention_encoder.forward

    def _attn_fwd(*args, **kwargs):
        out = orig_attn(*args, **kwargs)
        embedder._captured_a = out[0]
        return out

    embedder.atom_attention_encoder.forward = _attn_fwd
    embedder._dream_a_capture_installed = True
    embedder._captured_a = None


def _unavailable(name: str):
    raise NotImplementedError(
        f"{name} is not included in this release. "
        "Paper campaigns use CCD ligands and covalent bonds, not soft-token mixing."
    )


def build_candidate_token_basis(*args, **kwargs):
    _unavailable("build_candidate_token_basis")


def mix_restype_modified(*args, **kwargs):
    _unavailable("mix_restype_modified")


def compute_token_a(*args, **kwargs):
    _unavailable("compute_token_a")


def mix_candidate_chemistry(*args, **kwargs):
    _unavailable("mix_candidate_chemistry")


def mix_delta_chemistry(*args, **kwargs):
    _unavailable("mix_delta_chemistry")
