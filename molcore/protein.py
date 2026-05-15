"""
molcore.protein — Protein sequence handling with ESM-2 embeddings.

Provides :class:`ProteinSeq` — an immutable protein sequence that can be
embedded with Meta's ESM-2 protein language model and converted to PyG graphs.

Requires ``pip install molcore[bio]`` for embedding (transformers + torch).
FASTA parsing and sequence inspection work without extra dependencies.

Usage::

    from molcore.protein import ProteinSeq

    # Single sequence
    p = ProteinSeq.from_sequence("MKTLLILAVLCLGFAQAS", name="signal_peptide")
    emb = p.embed()                         # (320,) ESM-2 mean-pooled
    emb_full = p.embed(pooling="none")      # (L, 320) per-residue

    # From FASTA file
    seqs = ProteinSeq.from_fasta("proteins.fasta")

    # Batch embedding
    embeddings = ProteinSeq.embed_batch([p.sequence for p in seqs])  # (N, 320)

    # Joint protein–ligand dataset (requires PyTDC)
    pairs = ProteinSeq.from_bindingdb("Kd", target="EGFR")
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from typing import Optional

# Canonical single-letter amino acid codes
_AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$", re.IGNORECASE)

# Default ESM-2 model — 8 M params, 320-dim, fast and lightweight
_DEFAULT_ESM_MODEL = "facebook/esm2_t6_8M_UR50D"

# Larger alternatives (uncomment to use):
# _DEFAULT_ESM_MODEL = "facebook/esm2_t12_35M_UR50D"   # 35 M, 480-dim
# _DEFAULT_ESM_MODEL = "facebook/esm2_t30_150M_UR50D"  # 150 M, 640-dim
# _DEFAULT_ESM_MODEL = "facebook/esm2_t33_650M_UR50D"  # 650 M, 1280-dim


@dataclass(frozen=True)
class ProteinSeq:
    """
    Immutable protein sequence.

    Parameters
    ----------
    sequence : str
        Amino acid sequence in single-letter IUPAC notation (A-Z, upper-case).
    name : str
        Identifier (e.g. UniProt accession, gene name). Optional.
    """
    sequence: str
    name: str = ""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_sequence(cls, sequence: str, name: str = "") -> "ProteinSeq":
        """
        Create from a raw amino acid string.

        Raises :class:`ValueError` if the sequence is empty or contains
        non-standard characters.
        """
        seq = sequence.strip().upper()
        if not seq:
            raise ValueError("Protein sequence is empty")
        if len(seq) > 100_000:
            raise ValueError(
                f"Sequence length {len(seq):,} exceeds 100 000 — "
                "possible resource-exhaustion input"
            )
        if not _AA_PATTERN.match(seq):
            bad = set(seq) - set("ACDEFGHIKLMNPQRSTVWY")
            raise ValueError(
                f"Non-standard amino acid characters: {sorted(bad)!r}. "
                "Use standard single-letter IUPAC codes."
            )
        return cls(sequence=seq, name=name)

    @classmethod
    def from_fasta(cls, path: "str | pathlib.Path") -> "list[ProteinSeq]":
        """
        Parse a FASTA file into a list of :class:`ProteinSeq` objects.

        Multi-record FASTA files are supported. Records with non-standard
        characters are skipped (a warning is printed).
        """
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(f"FASTA file not found: {p}")
        text = p.read_text(encoding="utf-8", errors="replace")
        return cls._parse_fasta_text(text)

    @classmethod
    def from_fasta_string(cls, text: str) -> "list[ProteinSeq]":
        """Parse FASTA-formatted text directly (useful for in-memory strings)."""
        return cls._parse_fasta_text(text)

    @classmethod
    def _parse_fasta_text(cls, text: str) -> "list[ProteinSeq]":
        records: list[ProteinSeq] = []
        name, parts = "", []
        for line in text.splitlines():
            line = line.rstrip()
            if line.startswith(">"):
                if name and parts:
                    _try_append(records, name, "".join(parts))
                name = line[1:].strip()
                parts = []
            elif line and not line.startswith(";"):
                parts.append(line)
        if name and parts:
            _try_append(records, name, "".join(parts))
        return records

    # ------------------------------------------------------------------
    # ESM-2 embedding
    # ------------------------------------------------------------------

    def embed(
        self,
        model: str = _DEFAULT_ESM_MODEL,
        pooling: str = "mean",
        device: Optional[str] = None,
    ) -> "torch.Tensor":
        """
        Compute an ESM-2 embedding for this sequence.

        Parameters
        ----------
        model : str
            HuggingFace model ID. Default is ``facebook/esm2_t6_8M_UR50D``
            (8 M params, 320-dim).
        pooling : str
            ``"mean"`` → (hidden_size,) averaged over residues.
            ``"cls"``  → (hidden_size,) CLS token only.
            ``"none"`` → (L, hidden_size) full per-residue tensor.
        device : str or None
            ``"cpu"``, ``"cuda"``, ``"mps"``, or None (auto-detect).

        Returns a float32 :class:`torch.Tensor`.

        Requires ``pip install molcore[bio]``.
        """
        return self.__class__.embed_batch(
            [self.sequence], model=model, pooling=pooling, device=device
        )[0]

    @staticmethod
    def embed_batch(
        sequences: "list[str]",
        model: str = _DEFAULT_ESM_MODEL,
        pooling: str = "mean",
        batch_size: int = 8,
        device: Optional[str] = None,
        max_length: int = 1024,
    ) -> "torch.Tensor":
        """
        Embed a list of sequences with ESM-2.

        Parameters
        ----------
        sequences : list[str]
            Amino acid strings (standard IUPAC single-letter codes).
        model : str
            HuggingFace ESM-2 model ID.
        pooling : str
            ``"mean"`` or ``"cls"`` → (N, hidden_size).
            ``"none"`` → list of (L_i, hidden_size) tensors (variable length).
        batch_size : int
            Number of sequences per forward pass.
        device : str or None
            Target device. Auto-detected if None.
        max_length : int
            Truncate sequences longer than this before tokenization.

        Returns :class:`torch.Tensor` of shape ``(N, hidden_size)`` for
        ``pooling in {"mean","cls"}``, or a list of tensors for ``"none"``.

        Requires ``pip install molcore[bio]``.
        """
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "ESM-2 embedding requires transformers: pip install molcore[bio]"
            )

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        tokenizer = AutoTokenizer.from_pretrained(model)
        esm = AutoModel.from_pretrained(model).to(device).eval()

        all_outputs: list = []

        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i : i + batch_size]
                encoded = tokenizer(
                    batch_seqs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(device)

                out = esm(**encoded)
                hidden = out.last_hidden_state  # (B, L, H)
                attention_mask = encoded["attention_mask"]  # (B, L)

                if pooling == "mean":
                    # Mask padding tokens before averaging
                    mask = attention_mask.unsqueeze(-1).float()
                    summed = (hidden * mask).sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1e-9)
                    embs = (summed / counts).cpu()           # (B, H)
                    all_outputs.append(embs)

                elif pooling == "cls":
                    all_outputs.append(hidden[:, 0, :].cpu())  # (B, H)

                elif pooling == "none":
                    # Return each sequence's non-padding tokens separately
                    for b in range(hidden.size(0)):
                        seq_len = int(attention_mask[b].sum().item())
                        all_outputs.append(hidden[b, :seq_len, :].cpu())

                else:
                    raise ValueError(
                        f"pooling must be 'mean', 'cls', or 'none'; got {pooling!r}"
                    )

        if pooling in ("mean", "cls"):
            return torch.cat(all_outputs, dim=0)  # (N, H)
        else:
            return all_outputs  # list of (L_i, H)

    # ------------------------------------------------------------------
    # PyG graph (residue-level)
    # ------------------------------------------------------------------

    def to_pyg(self, add_sequence_edges: bool = True) -> "torch_geometric.data.Data":
        """
        Convert to a residue-level PyG graph.

        Node features (20-dim one-hot over standard amino acids).
        Edges: sequential adjacency (residue i → i±1).

        Requires torch_geometric (already a core molcore dependency).
        """
        import torch
        from torch_geometric.data import Data

        AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
        aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}

        seq = self.sequence
        n = len(seq)

        # Node features: 20-dim one-hot (unknown AAs → zero vector)
        x = torch.zeros((n, 20), dtype=torch.float32)
        for i, aa in enumerate(seq):
            if aa in aa_to_idx:
                x[i, aa_to_idx[aa]] = 1.0

        if add_sequence_edges and n > 1:
            src = list(range(n - 1)) + list(range(1, n))   # bidirectional
            dst = list(range(1, n))   + list(range(n - 1))
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(
            x=x,
            edge_index=edge_index,
            num_nodes=n,
            name=self.name,
            sequence=self.sequence,
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sequence)

    def __repr__(self) -> str:
        return f"ProteinSeq(name={self.name!r}, len={len(self)})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_append(records: list, name: str, raw_seq: str) -> None:
    seq = raw_seq.strip().upper()
    if not seq:
        return
    try:
        records.append(ProteinSeq.from_sequence(seq, name=name))
    except ValueError as exc:
        import warnings
        warnings.warn(f"Skipping sequence {name!r}: {exc}", stacklevel=3)
