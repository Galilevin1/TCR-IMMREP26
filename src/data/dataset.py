"""Dataset, feature normalisation, and the Leave-One-Peptide-Out splitter.

Leakage rules enforced here:
  * tcr_id / pos / neg / ambiguous are never features;
  * data_origin / vdjdb_score are excluded for now (metadata, not biology);
  * structural-feature normalisation statistics are fitted on TRAIN rows only.

The held-out peptide's *sequence* is a legitimate inference-time input; only its
labels and validation statistics must stay out of training.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..models.sequence_encoder import encode_sequence

FORBIDDEN_FEATURES = ["tcr_id", "pos", "neg", "ambiguous", "target"]
EXCLUDED_METADATA = ["data_origin", "vdjdb_score"]

MAX_CDR3_LEN = 25
MAX_PEPTIDE_LEN = 15


class TCRpMHCDataset(Dataset):
    def __init__(self, df, hla_vocab, structural=None, structural_columns=None,
                 max_cdr3_len=MAX_CDR3_LEN, max_peptide_len=MAX_PEPTIDE_LEN):
        self.df = df.reset_index(drop=True)
        self.hla_vocab = hla_vocab
        self.structural_columns = structural_columns or []
        self.max_cdr3_len = max_cdr3_len
        self.max_peptide_len = max_peptide_len

        if self.structural_columns:
            if structural is None:
                raise ValueError("structural_columns given but no structural frame")
            mat = (structural.reindex(self.df.tcr_id)[self.structural_columns]
                   .to_numpy(dtype=np.float32))
            self.structural = torch.from_numpy(mat)
        else:
            self.structural = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        item = {
            "cdr3a": torch.tensor(
                encode_sequence(row.cdr3_alpha_aa, self.max_cdr3_len), dtype=torch.long),
            "cdr3b": torch.tensor(
                encode_sequence(row.cdr3_beta_aa, self.max_cdr3_len), dtype=torch.long),
            "peptide": torch.tensor(
                encode_sequence(row.epitope_aa, self.max_peptide_len), dtype=torch.long),
            "hla": torch.tensor(self.hla_vocab[row.hla_long], dtype=torch.long),
            "target": torch.tensor(float(row.target), dtype=torch.float32),
        }
        if self.structural is not None:
            item["structure"] = self.structural[i]
        return item


def build_hla_vocab(df):
    """Built from HLA strings only - an identifier, never a label."""
    return {h: i for i, h in enumerate(sorted(df.hla_long.unique()))}


class StructuralNormalizer:
    """Standardises structural features using TRAIN statistics only."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.columns_ = None

    def fit(self, frame, columns):
        self.columns_ = list(columns)
        sub = frame[self.columns_].astype(float)
        self.mean_ = sub.mean()
        # guard constant columns; they contribute nothing but must not produce NaN
        self.std_ = sub.std().replace(0.0, 1.0).fillna(1.0)
        return self

    def transform(self, frame):
        sub = frame[self.columns_].astype(float)
        out = (sub - self.mean_) / self.std_
        return out.fillna(0.0)

    def fit_transform(self, frame, columns):
        return self.fit(frame, columns).transform(frame)


def usable_structural_columns(frame, min_unique=2):
    """Drop all-NaN (AlphaFold-only) and constant columns."""
    cols = [c for c in frame.columns if c not in FORBIDDEN_FEATURES]
    keep = []
    for c in cols:
        s = pd.to_numeric(frame[c], errors="coerce")
        if s.notna().any() and s.nunique(dropna=True) >= min_unique:
            keep.append(c)
    return keep


def leave_one_peptide_out_splits(df):
    """Yield (held_out_peptide, train_idx, val_idx).

    train: epitope_aa != held_out_peptide
    val:   epitope_aa == held_out_peptide
    """
    for peptide in sorted(df.epitope_aa.unique()):
        val = df.index[df.epitope_aa == peptide]
        train = df.index[df.epitope_aa != peptide]
        if len(train) and len(val):
            yield peptide, train, val
