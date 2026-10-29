"""Steps 12-14: engineering sanity check for Model A and Model B.

    uv run python scripts/run_ebm_pilot.py

This is a PIPELINE TEST on 6 rows, not model validation. It checks that the
forward pass runs, the loss falls, gradients reach every parameter group,
predictions stay finite, and structural features actually flow into Model B.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import (StructuralNormalizer, TCRpMHCDataset,
                              build_hla_vocab, leave_one_peptide_out_splits,
                              usable_structural_columns)
from src.models.ebm import TCRpMHCEnergyModel

RAW = ROOT / "data" / "raw" / "test_supervised.csv"
PILOT_TSV = ROOT / "data" / "tcrdock_pilot.tsv"
FEATURES = ROOT / "outputs" / "structural_features" / "pilot_template_features.csv"
MODEL_DIR = ROOT / "outputs" / "models"

SEED = 0


def gradient_report(model):
    """Which parameter groups received a finite, non-zero gradient."""
    groups = {}
    for name, p in model.named_parameters():
        head = name.split(".")[0]
        g = p.grad
        ok = g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
        groups[head] = groups.get(head, True) and bool(ok)
    return groups


def train_one(model, loader, epochs=150, lr=3e-3, use_structure=False, tag=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    history, grads = [], None

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in loader:
            opt.zero_grad()
            kwargs = {"structure": batch["structure"]} if use_structure else {}
            energy = model(batch["cdr3a"], batch["cdr3b"],
                           batch["peptide"], batch["hla"], **kwargs)
            logit = -energy                       # low energy = positive
            loss = loss_fn(logit, batch["target"])
            loss.backward()
            if epoch == 0:
                grads = gradient_report(model)
            opt.step()
            total += loss.item() * len(batch["target"])
        history.append(total / len(loader.dataset))
        if epoch % 30 == 0 or epoch == epochs - 1:
            print(f"    [{tag}] epoch {epoch:3d}  loss {history[-1]:.4f}")
    return history, grads


def evaluate(model, loader, use_structure=False):
    model.eval()
    energies, probs, targets = [], [], []
    with torch.no_grad():
        for batch in loader:
            kwargs = {"structure": batch["structure"]} if use_structure else {}
            e = model(batch["cdr3a"], batch["cdr3b"],
                      batch["peptide"], batch["hla"], **kwargs)
            energies.append(e)
            probs.append(torch.sigmoid(-e))
            targets.append(batch["target"])
    return (torch.cat(energies), torch.cat(probs), torch.cat(targets))


def check(label, condition):
    print(f"    {'PASS' if condition else 'FAIL'}  {label}")
    return condition


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    pilot = pd.read_csv(PILOT_TSV, sep="\t")
    feats = pd.read_csv(FEATURES).set_index("tcr_id")
    print(f"[1] pilot rows={len(pilot)}  structural feature table={feats.shape}")

    # the pilot TSV already carries epitope/hla under TCRDock names
    pilot = pilot.rename(columns={"peptide": "epitope_aa", "mhc": "hla_long",
                                  "cdr3a": "cdr3_alpha_aa", "cdr3b": "cdr3_beta_aa"})

    struct_cols = usable_structural_columns(feats)
    print(f"[2] usable structural features: {len(struct_cols)} "
          f"(of {feats.shape[1]}; dropped all-NaN and constant)")

    # Normalisation fitted on TRAIN rows only. With 6 rows the "train" set is the
    # whole pilot; the API is the one the LOPO loop will use.
    normalizer = StructuralNormalizer().fit(feats.loc[pilot.tcr_id], struct_cols)
    normed = normalizer.transform(feats)
    print(f"[3] normaliser fitted on {len(pilot)} train rows "
          f"(mean~0: {float(normed[struct_cols].mean().abs().max()):.2e})")

    hla_vocab = build_hla_vocab(pilot)
    ds_seq = TCRpMHCDataset(pilot, hla_vocab)
    ds_hyb = TCRpMHCDataset(pilot, hla_vocab, structural=normed,
                            structural_columns=struct_cols)
    loader_seq = DataLoader(ds_seq, batch_size=len(pilot), shuffle=True)
    loader_hyb = DataLoader(ds_hyb, batch_size=len(pilot), shuffle=True)
    # evaluation must not shuffle: predictions and targets have to stay aligned
    eval_seq = DataLoader(ds_seq, batch_size=len(pilot), shuffle=False)
    eval_hyb = DataLoader(ds_hyb, batch_size=len(pilot), shuffle=False)

    results = {}
    ok = True

    print("\n[4] Model A - sequence only")
    model_a = TCRpMHCEnergyModel(n_hla=len(hla_vocab), use_structure=False)
    print(f"    parameters: {sum(p.numel() for p in model_a.parameters()):,}")
    hist_a, grads_a = train_one(model_a, loader_seq, use_structure=False, tag="A")
    e_a, p_a, y_a = evaluate(model_a, eval_seq)
    results["A"] = (hist_a, e_a, p_a, y_a)

    print("\n[5] Model B - sequence + structure")
    model_b = TCRpMHCEnergyModel(n_hla=len(hla_vocab), use_structure=True,
                                 n_structural_features=len(struct_cols))
    print(f"    parameters: {sum(p.numel() for p in model_b.parameters()):,}")
    hist_b, grads_b = train_one(model_b, loader_hyb, use_structure=True, tag="B")
    e_b, p_b, y_b = evaluate(model_b, eval_hyb, use_structure=True)
    results["B"] = (hist_b, e_b, p_b, y_b)

    print("\n[6] engineering checks")
    for tag, (hist, energy, prob, y) in results.items():
        print(f"  Model {tag}:")
        ok &= check(f"forward pass produced {len(energy)} energies", len(energy) == len(pilot))
        ok &= check("energy is a single scalar per example", energy.dim() == 1)
        ok &= check("all predictions finite", bool(torch.isfinite(prob).all()))
        ok &= check("probabilities in [0,1]", bool(((prob >= 0) & (prob <= 1)).all()))
        # after deliberately overfitting 6 rows the sigmoid saturates; expected here,
        # but a reason to use weight decay / early stopping on the real dataset.
        sat = int(((prob <= 1e-7) | (prob >= 1 - 1e-7)).sum())
        print(f"    NOTE  {sat}/{len(prob)} probabilities saturated "
              f"(|energy| max {energy.abs().max():.1f})")
        ok &= check(f"loss decreased ({hist[0]:.4f} -> {hist[-1]:.4f})", hist[-1] < hist[0])
        ok &= check("can overfit the tiny pilot (loss < 0.05)", hist[-1] < 0.05)
        acc = ((prob > 0.5).float() == y).float().mean().item()
        ok &= check(f"fits pilot labels (accuracy {acc:.2f})", acc == 1.0)
        # low energy must correspond to positives
        mean_pos = energy[y == 1].mean().item()
        mean_neg = energy[y == 0].mean().item()
        ok &= check(f"E(pos)={mean_pos:+.3f} < E(neg)={mean_neg:+.3f}", mean_pos < mean_neg)

    print("\n  gradient flow (epoch 0):")
    for tag, grads in [("A", grads_a), ("B", grads_b)]:
        missing = [k for k, v in grads.items() if not v]
        ok &= check(f"Model {tag}: all {len(grads)} parameter groups got gradients"
                    + (f"; MISSING {missing}" if missing else ""), not missing)

    print("\n[7] structural features genuinely affect Model B")
    batch = next(iter(DataLoader(ds_hyb, batch_size=len(pilot))))
    model_b.eval()
    with torch.no_grad():
        base = model_b(batch["cdr3a"], batch["cdr3b"], batch["peptide"],
                       batch["hla"], structure=batch["structure"])
        perturbed = model_b(batch["cdr3a"], batch["cdr3b"], batch["peptide"],
                            batch["hla"], structure=torch.zeros_like(batch["structure"]))
    delta = (base - perturbed).abs().max().item()
    ok &= check(f"zeroing structure changes the energy (max |delta| = {delta:.4f})",
                delta > 1e-6)

    print("\n[8] Leave-One-Peptide-Out splitter (on the FULL dataframe)")
    full = pd.read_csv(RAW)
    full = full[full.ambiguous == 0].copy()
    full["target"] = (full.pos == 1).astype(int)
    n_splits = 0
    for peptide, tr, va in leave_one_peptide_out_splits(full):
        n_splits += 1
        overlap = set(full.loc[tr, "epitope_aa"]) & {peptide}
        assert not overlap, "peptide leaked into training fold"
        print(f"    hold out {peptide:12} train={len(tr):5} val={len(va):4} "
              f"val_pos={int(full.loc[va,'target'].sum()):4}")
    ok &= check(f"{n_splits} clean LOPO folds, no peptide overlap", n_splits == 5)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model_a.state_dict(), MODEL_DIR / "pilot_model_a_seq.pt")
    torch.save(model_b.state_dict(), MODEL_DIR / "pilot_model_b_hybrid.pt")
    print(f"\n[9] saved both models to {MODEL_DIR.relative_to(ROOT)}")

    print("\n" + ("ALL ENGINEERING CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
