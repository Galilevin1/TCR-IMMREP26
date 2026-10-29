"""Steps 8-11: parse the pilot structures and write the structural feature table.

    envs/tcrdock/Scripts/python.exe scripts/extract_pilot_features.py

Operates on the TCRDock *template* structures produced on CPU by
setup_for_alphafold. When GPU-predicted structures exist, point --pdb-suffix at
them instead; nothing else changes.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.structure.extract_tcrdock_features import (
    extract_structural_features, template_quality_features)
from src.structure.run_tcrdock import parse_structures, save_feature_cache

SETUP_DIR = ROOT / "outputs" / "structures" / "pilot_setup"
PARSE_TSV = ROOT / "outputs" / "structures" / "pilot_parsed.tsv"
CACHE_NAME = "pilot_template_features"  # -> outputs/structural_features/<name>.csv


def main():
    targets = pd.read_table(SETUP_DIR / "targets.tsv")
    print(f"[1] {len(targets)} pilot targets from setup")

    # one representative template structure per target (template index 0)
    pdbs, keep = [], []
    for _, row in targets.iterrows():
        pdb = SETUP_DIR / f"{row.targetid}_0.pdb"
        if pdb.exists():
            pdbs.append(pdb)
            keep.append(row)
        else:
            print(f"    MISSING structure for {row.tcr_id}: {pdb.name}")
    targets = pd.DataFrame(keep)
    print(f"[2] parsing {len(pdbs)} structures with parse_tcr_pmhc_pdbfile.py (CPU)")
    parse_structures(pdbs, PARSE_TSV)

    parsed = pd.read_table(PARSE_TSV)
    print(f"[3] parsed {len(parsed)} structures -> docking geometry available")
    assert len(parsed) == len(targets), "parse dropped structures; aborting"

    rows = []
    for (_, tgt), (_, prow), pdb in zip(targets.iterrows(), parsed.iterrows(), pdbs):
        print(f"[4] features for {tgt.tcr_id}")
        feats = extract_structural_features(pdb, prow)
        feats.update(template_quality_features(
            SETUP_DIR / f"{tgt.targetid}_alignments.tsv"))
        feats["tcr_id"] = tgt.tcr_id
        feats["target"] = int(tgt.target)
        rows.append(feats)

    df = pd.DataFrame(rows).set_index("tcr_id")
    # keep target last for readability
    df = df[[c for c in df.columns if c != "target"] + ["target"]]

    cache_path = save_feature_cache(df, CACHE_NAME)
    print(f"[5] wrote {cache_path.relative_to(ROOT)}  shape={df.shape}")

    informative = [c for c in df.columns
                   if c != "target" and df[c].notna().any() and df[c].nunique() > 1]
    print(f"[6] {len(informative)} features vary across the pilot; "
          f"{df.isna().all().sum()} are all-NaN (AlphaFold-only)")
    with pd.option_context("display.width", 200, "display.max_columns", 12):
        print(df[informative[:10] + ["target"]].to_string())


if __name__ == "__main__":
    main()
