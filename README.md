# TCR–pMHC structure + EBM proof of concept

Two isolated Python environments, deliberately kept apart:

| Environment | Python | Purpose |
|---|---|---|
| `.venv` (uv project) | 3.11 | pandas / torch — EBM training |
| `envs/tcrdock/` | 3.10 | TCRDock + BLAST — structural pipeline |

They are separated because TCRDock pins `biopython==1.79`, whose `Bio.SubsMat`
module was removed in 1.80; that pin forces Python ≤ 3.10 and must not constrain
the modelling environment.

## Reproducing the structural environment

```bash
git clone --depth 1 https://github.com/phbradley/TCRdock.git external/TCRdock
uv venv --python 3.10 envs/tcrdock
uv pip install --python envs/tcrdock/Scripts/python.exe \
    "biopython==1.79" "numpy<2" "pandas<2.2" scipy matplotlib
```

BLAST 2.11.0 is required even for the non-AlphaFold code paths. The official
`download_blast.py` only supports Linux and macOS, so on Windows fetch the
matching build into the directory name TCRdock expects:

```bash
curl -L -o blast.tar.gz \
  https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.11.0/ncbi-blast-2.11.0+-x64-win64.tar.gz
tar -xzf blast.tar.gz -C external/TCRdock   # creates ncbi-blast-2.11.0+/
```

### Local patch to the TCRdock clone

`tcrdock/blast.py` appended `2> /dev/null` to every BLAST command, which cmd.exe
cannot resolve, so all BLAST calls failed silently on Windows. One line now uses
`os.devnull` instead. This is the only modification to upstream code.

`external/` is git-ignored, so the patch is tracked separately and must be
re-applied to a fresh clone:

```bash
git -C external/TCRdock apply ../../patches/0001-blast-devnull-windows.patch
```

## Raw data is not tracked

`data/raw/` is deliberately git-ignored: it holds the IMMREP26 / Kaggle
competition data, which is redistributed under the competition's own terms and
should not live in this repository. **A fresh clone cannot run anything until
these four files are restored manually** from the IMMREP26 / Kaggle data source:

```
data/raw/test_supervised.csv      # labelled set the pilot and EBM use
data/raw/test.csv                 # unlabelled evaluation set
data/raw/External_data.csv        # additional external TCR data
data/raw/tcrdist_predictions.csv  # TCRdist baseline predictions
```

Only `test_supervised.csv` is needed to reproduce the pilot end to end.

## Pipeline stages

| Stage | Where it runs | Status |
|---|---|---|
| Input construction and gene validation | CPU | works |
| `setup_for_alphafold.py` (templates, alignments) | CPU | works, ~1 min for 6 targets |
| `run_prediction.py` (AlphaFold forward pass) | **GPU** | not attempted locally |
| `parse_tcr_pmhc_pdbfile.py` + feature extraction | CPU | works |

Feature extraction does not care whether the PDB came from the template stage or
from an AlphaFold prediction, so the same code serves both.

## Running

```bash
envs/tcrdock/Scripts/python.exe -I scripts/audit_tcrdock_compat.py   # compatibility audit
envs/tcrdock/Scripts/python.exe -I scripts/run_tcrdock_pilot.py      # build data/tcrdock_pilot.tsv
envs/tcrdock/Scripts/python.exe    scripts/extract_pilot_features.py # structural features
uv run python scripts/run_ebm_pilot.py                               # Model A and Model B
```

Structural features are cached in `outputs/structural_features/` keyed by
`tcr_id`; retraining the EBM never re-runs TCRDock.

## GPU step, when a machine is available

```bash
cd external/TCRdock
python run_prediction.py \
    --targets ../../outputs/structures/pilot_setup/targets.tsv \
    --outfile_prefix ../../outputs/structures/pilot_pred \
    --model_names model_2_ptm_ft \
    --data_dir $ALPHAFOLD_DATA_DIR
```

This additionally requires the AlphaFold parameter set and a JAX environment,
neither of which is installed here.
