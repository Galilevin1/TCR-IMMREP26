"""Energy-based TCR:pMHC compatibility model.

The model scores a (TCR, pMHC) *pair* rather than one flat feature vector: the
two sides are encoded separately, projected into a shared space, and combined
through an explicit interaction (product and absolute difference).

Convention:
    low energy  = compatible  = positive
    logit       = -energy
    loss        = BCEWithLogitsLoss(-energy, target)
"""
import torch
import torch.nn as nn

from .sequence_encoder import SequenceEncoder


class TCRpMHCEnergyModel(nn.Module):
    """Model A (use_structure=False) and Model B (use_structure=True)."""

    def __init__(self, n_hla, shared_dim=32, n_structural_features=0,
                 use_structure=False, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.use_structure = use_structure
        if use_structure and n_structural_features == 0:
            raise ValueError("use_structure=True needs n_structural_features > 0")

        self.alpha_encoder = SequenceEncoder(out_dim=shared_dim)
        self.beta_encoder = SequenceEncoder(out_dim=shared_dim)
        self.peptide_encoder = SequenceEncoder(out_dim=shared_dim)
        self.hla_embedding = nn.Embedding(n_hla, shared_dim)

        self.tcr_projection = nn.Sequential(
            nn.Linear(2 * shared_dim, shared_dim), nn.ReLU(),
            nn.Linear(shared_dim, shared_dim))
        self.pmhc_projection = nn.Sequential(
            nn.Linear(2 * shared_dim, shared_dim), nn.ReLU(),
            nn.Linear(shared_dim, shared_dim))

        energy_in = 4 * shared_dim  # z_tcr, z_pmhc, product, abs-difference
        if use_structure:
            energy_in += n_structural_features

        self.energy_mlp = nn.Sequential(
            nn.Linear(energy_in, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1))

    def encode_tcr(self, cdr3a, cdr3b):
        return self.tcr_projection(
            torch.cat([self.alpha_encoder(cdr3a), self.beta_encoder(cdr3b)], dim=1))

    def encode_pmhc(self, peptide, hla):
        return self.pmhc_projection(
            torch.cat([self.peptide_encoder(peptide), self.hla_embedding(hla)], dim=1))

    def forward(self, cdr3a, cdr3b, peptide, hla, structure=None):
        """Returns energy, shape (B,). Lower = more compatible."""
        z_tcr = self.encode_tcr(cdr3a, cdr3b)
        z_pmhc = self.encode_pmhc(peptide, hla)

        sequence_interaction = torch.cat(
            [z_tcr, z_pmhc, z_tcr * z_pmhc, (z_tcr - z_pmhc).abs()], dim=1)

        if self.use_structure:
            if structure is None:
                raise ValueError("structural features required but not supplied")
            energy_input = torch.cat([sequence_interaction, structure], dim=1)
        else:
            energy_input = sequence_interaction

        return self.energy_mlp(energy_input).squeeze(-1)

    def logits(self, *args, **kwargs):
        """logit = -energy, so sigmoid(logit) = P(positive)."""
        return -self.forward(*args, **kwargs)
