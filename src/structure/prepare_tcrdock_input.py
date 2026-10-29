"""Map the supervised dataframe onto TCRDock's 10-column input contract.

The contract (verified against external/TCRdock/setup_for_alphafold.py and the
reference tables in external/TCRdock/tcrdock/db/):

    organism mhc_class mhc peptide va ja cdr3a vb jb cdr3b

Only one biological identifier needs rewriting: TCRDock's gene tables spell the
TRAV/DV genes with a slash (TRAV14/DV4*01) while the TRAV_IMGT column uses an
underscore (TRAV14_DV4*01). Every other identifier is passed through unchanged.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TCRDOCK_DIR = ROOT / "external" / "TCRdock"

ORGANISM = "human"
MHC_CLASS = 1

# dataframe column -> TCRDock column
COLUMN_MAP = {
    "hla_long": "mhc",
    "epitope_aa": "peptide",
    "TRAV_IMGT": "va",
    "TRAJ_IMGT": "ja",
    "cdr3_alpha_aa": "cdr3a",
    "TRBV_IMGT": "vb",
    "TRBJ_IMGT": "jb",
    "cdr3_beta_aa": "cdr3b",
}

TCRDOCK_COLUMNS = ["organism", "mhc_class", "mhc", "peptide",
                   "va", "ja", "cdr3a", "vb", "jb", "cdr3b"]

# Columns that must never be used as predictive features downstream.
LABEL_COLUMNS = ["pos", "neg", "ambiguous"]
ID_COLUMNS = ["tcr_id"]
METADATA_COLUMNS = ["data_origin", "vdjdb_score"]


def to_tcrdock_gene(gene):
    """TRAV14_DV4*01 -> TRAV14/DV4*01. Everything else passes through."""
    if not isinstance(gene, str):
        return gene
    return gene.replace("_DV", "/DV")


def add_target(df):
    """target = 1 if pos else 0, with ambiguous rows dropped entirely."""
    out = df.loc[df["ambiguous"] == 0].copy()
    out["target"] = (out["pos"] == 1).astype(int)
    return out


def build_tcrdock_frame(df):
    """Return a frame with TCRDock's 10 columns plus tcr_id (join key) and target."""
    out = pd.DataFrame(index=df.index)
    out["tcr_id"] = df["tcr_id"]
    out["organism"] = ORGANISM
    out["mhc_class"] = MHC_CLASS
    for src, dst in COLUMN_MAP.items():
        out[dst] = df[src]
    for col in ("va", "ja", "vb", "jb"):
        out[col] = out[col].map(to_tcrdock_gene)
    if "target" in df.columns:
        out["target"] = df["target"]
    return out


def validate(frame, verbose=True):
    """Validate gene names using TCRDock's own checker. Returns a boolean mask."""
    sys.path.insert(0, str(TCRDOCK_DIR))
    from tcrdock.sequtil import genes_ok_for_modeling

    ok = []
    for _, row in frame.iterrows():
        ok.append(genes_ok_for_modeling(
            row.organism, row.va, row.ja, row.vb, row.jb, verbose=verbose))
    return pd.Series(ok, index=frame.index)


def select_pilot(df, n_pos=3, n_neg=3, peptide=None, hla=None, seed=0):
    """Pick a tiny balanced pilot, preferably from one peptide/HLA combination."""
    d = add_target(df)
    if peptide is None:
        combo = (d.groupby(["epitope_aa", "hla_long"])
                  .target.agg(["size", "sum"])
                  .sort_values("size", ascending=False))
        peptide, hla = combo.index[0]
    sub = d[(d.epitope_aa == peptide) & (d.hla_long == hla)]
    pos = sub[sub.target == 1].sample(n=n_pos, random_state=seed)
    neg = sub[sub.target == 0].sample(n=n_neg, random_state=seed)
    return pd.concat([pos, neg]).sort_values("target", ascending=False)
