"""Drive the TCRDock stages and cache every result against tcr_id.

Stages, and where each can run:

  1. input construction      CPU   src/structure/prepare_tcrdock_input.py
  2. setup_for_alphafold     CPU   templates + targets.tsv  (works, minutes)
  3. run_prediction.py       GPU   AlphaFold forward pass   (not feasible on CPU)
  4. parse + feature extract CPU   works on any ternary PDB

Stage 4 does not care whether the PDB came from stage 2 (template) or stage 3
(prediction), so the feature code is validated now and reused unchanged later.

Caching: structure prediction is never redone if outputs/structural_features/
already holds a row for that tcr_id. Retraining the EBM never re-runs TCRDock.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TCRDOCK_DIR = ROOT / "external" / "TCRdock"
PYTHON = ROOT / "envs" / "tcrdock" / "Scripts" / "python.exe"
FEATURE_DIR = ROOT / "outputs" / "structural_features"


def run_setup(targets_tsv, output_dir, new_docking=True):
    """Stage 2 - CPU. Builds template PDBs, alignments and targets.tsv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(PYTHON), "-W", "ignore", "setup_for_alphafold.py",
           "--targets_tsvfile", str(Path(targets_tsv).resolve()),
           "--output_dir", str(output_dir.resolve())]
    if new_docking:
        cmd.append("--new_docking")
    subprocess.run(cmd, cwd=TCRDOCK_DIR, check=True)
    return output_dir / "targets.tsv"


def parse_structures(pdbfiles, out_tsv, organism="human", mhc_class=1):
    """Stage 4a - CPU. Docking geometry + CDR/MHC annotation for each PDB."""
    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(PYTHON), "-W", "ignore", "parse_tcr_pmhc_pdbfile.py",
           "--pdbfiles", *[str(Path(p).resolve()) for p in pdbfiles],
           "--organism", organism, "--mhc_class", str(mhc_class),
           "--out_tsvfile", str(out_tsv.resolve())]
    subprocess.run(cmd, cwd=TCRDOCK_DIR, check=True)
    return out_tsv


def build_prediction_command(targets_tsv, outfile_prefix, data_dir,
                             model_names="model_2_ptm_ft"):
    """Stage 3 - GPU. Returned as a string; deliberately not executed on CPU."""
    return (f"python run_prediction.py --targets {targets_tsv} "
            f"--outfile_prefix {outfile_prefix} --model_names {model_names} "
            f"--data_dir {data_dir}")


def load_feature_cache(name):
    path = FEATURE_DIR / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path).set_index("tcr_id")
    return None


def save_feature_cache(df, name):
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FEATURE_DIR / f"{name}.csv"
    df.to_csv(path, index=True)
    return path
