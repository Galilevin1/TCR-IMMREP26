"""Audit the supervised dataframe against TCRDock's accepted input identifiers.

Runs inside the isolated envs/tcrdock environment. Read-only: reports what would
need to change, does not rewrite any biological identifier in place.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "TCRdock"))

import pandas as pd

from tcrdock.sequtil import all_genes, both_structure_alignments

ORGANISM = "human"

# TCRDock writes the TRAV/DV genes with a slash; the IMGT columns here use '_'.
def to_tcrdock_gene(g):
    if not isinstance(g, str):
        return g
    return g.replace("_DV", "/DV")


def main():
    df = pd.read_csv(ROOT / "data" / "raw" / "test_supervised.csv")
    print(f"rows={len(df)}  columns={len(df.columns)}")

    known = set(all_genes[ORGANISM])
    aligned = {v for (o, v) in both_structure_alignments.index if o == ORGANISM}

    for col, kind in [("TRAV_IMGT", "V"), ("TRAJ_IMGT", "J"),
                      ("TRBV_IMGT", "V"), ("TRBJ_IMGT", "J")]:
        raw = df[col].dropna().unique()
        mapped = {g: to_tcrdock_gene(g) for g in raw}
        changed = {k: v for k, v in mapped.items() if k != v}
        bad_raw = sorted(g for g in raw if g not in known)
        bad_mapped = sorted(v for v in mapped.values() if v not in known)
        print(f"\n--- {col}: {len(raw)} distinct")
        print(f"    renamed by '_DV'->'/DV': {len(changed)} {sorted(changed.items())[:3]}")
        print(f"    unrecognized BEFORE mapping: {len(bad_raw)} {bad_raw[:8]}")
        print(f"    unrecognized AFTER  mapping: {len(bad_mapped)} {bad_mapped[:8]}")
        if kind == "V":
            no_align = sorted(v for v in mapped.values()
                              if v in known and v not in aligned)
            print(f"    recognized but NO structure alignment: {len(no_align)} {no_align[:8]}")

    # MHC: TCRDock's pMHC template db keys on the 'A*02:01' style string.
    tmpl = pd.read_table(ROOT / "external" / "TCRdock" / "tcrdock" / "db"
                         / "pmhc_templates_v2.tsv")
    tmpl_alleles = set(tmpl.loc[tmpl.organism == ORGANISM, "mhc_allele"])
    print("\n--- hla_long")
    for a, n in df.hla_long.value_counts().items():
        print(f"    {a!r:12} n={n:5}  in pmhc_template db: {a in tmpl_alleles}")

    # Peptide sanity for MHC class 1
    print("\n--- peptide lengths (mhc_class=1)")
    print(df.epitope_aa.str.len().value_counts().sort_index().to_string())

    # CDR3 conventions required by setup_for_alphafold
    for col in ["cdr3_alpha_aa", "cdr3_beta_aa"]:
        s = df[col].dropna()
        print(f"\n--- {col}: starts with C: {(s.str[0]=='C').sum()}/{len(s)}; "
              f"ends F/W: {s.str[-1].isin(list('FW')).sum()}/{len(s)}; "
              f"non-standard chars: {(~s.str.match('^[ACDEFGHIKLMNPQRSTVWY]+$')).sum()}")
        print(f"    end-char counts: {s.str[-1].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
