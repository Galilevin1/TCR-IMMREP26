"""Step 1-6: build and validate the TCRDock pilot input. Runs on CPU, no AlphaFold.

    envs/tcrdock/Scripts/python.exe -I scripts/run_tcrdock_pilot.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.structure.prepare_tcrdock_input import (
    TCRDOCK_COLUMNS, add_target, build_tcrdock_frame, select_pilot, validate)

RAW = ROOT / "data" / "raw" / "test_supervised.csv"
PILOT_TSV = ROOT / "data" / "tcrdock_pilot.tsv"


def main():
    print("[1] loading dataframe:", RAW.name)
    df = pd.read_csv(RAW)
    print(f"    {len(df)} rows")

    labelled = add_target(df)
    print(f"[2] dropped {len(df) - len(labelled)} ambiguous rows -> {len(labelled)} usable")
    print(f"    target: {labelled.target.sum()} pos / {(labelled.target == 0).sum()} neg")

    print("[3] selecting pilot (single peptide/HLA combo, balanced)")
    pilot = select_pilot(df, n_pos=3, n_neg=3)
    print(f"    peptide={pilot.epitope_aa.iloc[0]} hla={pilot.hla_long.iloc[0]} n={len(pilot)}")

    print("[4] mapping to TCRDock columns")
    frame = build_tcrdock_frame(pilot)

    print("[5] validating gene names with tcrdock.sequtil.genes_ok_for_modeling")
    ok = validate(frame)
    print(f"    {ok.sum()}/{len(ok)} rows pass")
    if not ok.all():
        print("    FAILED rows:", frame.loc[~ok, 'tcr_id'].tolist())
        frame = frame[ok]

    PILOT_TSV.parent.mkdir(parents=True, exist_ok=True)
    # tcr_id kept as the join key back to the original dataframe; setup_for_alphafold
    # ignores extra columns, so it is safe to carry it (and target) in the same file.
    frame[["tcr_id"] + TCRDOCK_COLUMNS + ["target"]].to_csv(
        PILOT_TSV, sep="\t", index=False)
    print(f"[6] wrote {PILOT_TSV.relative_to(ROOT)}")
    print(frame[["tcr_id", "mhc", "peptide", "va", "cdr3a", "vb", "cdr3b", "target"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
