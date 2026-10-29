"""Derive numerical structural features from a TCR:pMHC ternary structure.

Every feature here is computed from quantities TCRDock actually produces:

  * the docking geometry written by parse_tcr_pmhc_pdbfile.py (6 floats + 2 bools,
    documented in the TCRdock README);
  * the chain boundaries, CDR loop spans and MHC core positions in the
    `tcrdock_info` JSON written by the same script;
  * heavy-atom coordinates read back from the PDB file itself.

Nothing is invented: if a quantity needs an AlphaFold run (PAE, pLDDT), it lives in
`ALPHAFOLD_ONLY_FEATURES` and is emitted as NaN until predicted structures exist.

The same code path consumes a TCRDock *template* structure (CPU, available now) and
an AlphaFold *predicted* structure (GPU, later) - only the input PDB changes.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONTACT_CUTOFF = 4.5  # Angstroms, heavy-atom

# 0-indexed order of the 8 loops in tcrdock_info['tcr_cdrs']
CDR_NAMES = ["cdr1a", "cdr2a", "cdr25a", "cdr3a", "cdr1b", "cdr2b", "cdr25b", "cdr3b"]

DOCKING_GEOMETRY_FEATURES = [
    "dg_d", "dg_torsion", "dg_tcr_unit_y", "dg_tcr_unit_z",
    "dg_mhc_unit_y", "dg_mhc_unit_z",
    "dg_tcr_unit_x_is_negative", "dg_mhc_unit_x_is_negative",
]

# Require a real AlphaFold forward pass; NaN from template-only runs.
ALPHAFOLD_ONLY_FEATURES = ["af_pmhc_tcr_pae", "af_plddt_mean", "af_plddt_cdr3a",
                           "af_plddt_cdr3b"]


def read_residue_coords(pdbfile):
    """Heavy-atom coords per residue, in file order (matches the chainseq order)."""
    residues, keys = [], []
    current, current_key = None, None
    for line in Path(pdbfile).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        element = line[76:78].strip() or line[12:16].strip()[:1]
        if element == "H":
            continue
        key = (line[21], line[22:27])  # chain, resseq+icode
        if key != current_key:
            current = []
            residues.append(current)
            keys.append(key)
            current_key = key
        current.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return [np.asarray(r, dtype=float) for r in residues], keys


def _contact_stats(coords, idx_a, idx_b, cutoff=CONTACT_CUTOFF):
    """(n residue-residue contacts, n atom-atom contacts, min heavy-atom distance)."""
    n_res, n_atom, best = 0, 0, np.inf
    for i in idx_a:
        ai = coords[i]
        if ai.size == 0:
            continue
        for j in idx_b:
            bj = coords[j]
            if bj.size == 0:
                continue
            d = np.linalg.norm(ai[:, None, :] - bj[None, :, :], axis=-1)
            m = d.min()
            if m < best:
                best = m
            hits = int((d < cutoff).sum())
            if hits:
                n_res += 1
                n_atom += hits
    return n_res, n_atom, (float(best) if np.isfinite(best) else np.nan)


def extract_structural_features(pdbfile, parsed_row):
    """Return a flat {feature_name: number} dict for one ternary structure.

    parsed_row: one row of the TSV written by parse_tcr_pmhc_pdbfile.py.
    """
    info = json.loads(parsed_row["tcrdock_info"])
    if not info.get("valid", False):
        raise ValueError(f"TCRDock marked the parse invalid: {pdbfile}")

    feats = {}

    # --- docking geometry, straight from TCRDock -------------------------------
    for name, col in zip(DOCKING_GEOMETRY_FEATURES,
                         ["d", "torsion", "tcr_unit_y", "tcr_unit_z",
                          "mhc_unit_y", "mhc_unit_z",
                          "tcr_unit_x_is_negative", "mhc_unit_x_is_negative"]):
        feats[name] = float(parsed_row[col])

    # torsion is an angle in [0, 2pi): give the model a continuous encoding too
    feats["dg_torsion_sin"] = float(np.sin(parsed_row["torsion"]))
    feats["dg_torsion_cos"] = float(np.cos(parsed_row["torsion"]))

    # --- chain layout ----------------------------------------------------------
    chain_lens = [len(s) for s in parsed_row["chainseq"].split("/")]
    assert len(chain_lens) == 4, f"expected MHC/pep/TCRA/TCRB, got {len(chain_lens)}"
    mhc_len, pep_len, tcra_len, _ = chain_lens
    mhc_idx = range(0, mhc_len)
    pep_idx = range(mhc_len, mhc_len + pep_len)
    tcr_idx = range(mhc_len + pep_len, sum(chain_lens))

    coords, _ = read_residue_coords(pdbfile)
    if len(coords) != sum(chain_lens):
        raise ValueError(f"{pdbfile}: {len(coords)} residues parsed, "
                         f"chainseq implies {sum(chain_lens)}")

    feats["peptide_length"] = float(pep_len)

    # --- CDR loop <-> peptide contacts ----------------------------------------
    cdrs = info["tcr_cdrs"]
    for name, (start, stop) in zip(CDR_NAMES, cdrs):
        loop = range(start, stop + 1)
        n_res, n_atom, mind = _contact_stats(coords, loop, pep_idx)
        feats[f"n_res_contacts_{name}_pep"] = float(n_res)
        feats[f"n_atom_contacts_{name}_pep"] = float(n_atom)
        feats[f"mindist_{name}_pep"] = mind
        feats[f"{name}_length"] = float(stop - start + 1)

    # --- whole-TCR interfaces --------------------------------------------------
    n_res, n_atom, mind = _contact_stats(coords, tcr_idx, pep_idx)
    feats["n_res_contacts_tcr_pep"] = float(n_res)
    feats["n_atom_contacts_tcr_pep"] = float(n_atom)
    feats["mindist_tcr_pep"] = mind

    n_res, n_atom, mind = _contact_stats(coords, tcr_idx, mhc_idx)
    feats["n_res_contacts_tcr_mhc"] = float(n_res)
    feats["n_atom_contacts_tcr_mhc"] = float(n_atom)
    feats["mindist_tcr_mhc"] = mind

    total = feats["n_atom_contacts_tcr_pep"] + feats["n_atom_contacts_tcr_mhc"]
    feats["n_atom_contacts_interface"] = total
    feats["frac_interface_on_peptide"] = (
        feats["n_atom_contacts_tcr_pep"] / total if total > 0 else np.nan)

    # how much of the peptide the TCR actually touches
    touched = sum(
        1 for p in pep_idx
        if _contact_stats(coords, [p], tcr_idx)[1] > 0)
    feats["n_peptide_residues_contacted"] = float(touched)
    feats["frac_peptide_residues_contacted"] = touched / pep_len

    # --- not available without a GPU AlphaFold run -----------------------------
    for name in ALPHAFOLD_ONLY_FEATURES:
        feats[name] = np.nan

    return feats


def template_quality_features(alignments_tsv):
    """Template-similarity numbers produced by setup_for_alphafold (CPU stage).

    These describe how good the structural templates were, averaged over runs.
    """
    al = pd.read_table(alignments_tsv)
    out = {}
    for col in ["overall_idents", "pmhc_idents", "tcra_idents", "tcrb_idents"]:
        if col in al.columns:
            out[f"tmpl_{col}_mean"] = float(al[col].mean())
            out[f"tmpl_{col}_max"] = float(al[col].max())
    out["tmpl_n_templates"] = float(len(al))
    return out
