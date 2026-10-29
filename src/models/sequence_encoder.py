"""Small amino-acid sequence encoders: embedding -> 1D CNN -> pooling."""
import torch
import torch.nn as nn

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAD_IDX = 0
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
VOCAB_SIZE = len(AMINO_ACIDS) + 1  # +1 for padding


def encode_sequence(seq, max_len):
    """Amino-acid string -> fixed-length index list (0 = pad)."""
    idx = [AA_TO_IDX.get(aa, PAD_IDX) for aa in seq[:max_len]]
    return idx + [PAD_IDX] * (max_len - len(idx))


class SequenceEncoder(nn.Module):
    """Embedding + 1D CNN + masked mean/max pooling. Deliberately small."""

    def __init__(self, embed_dim=16, hidden_dim=32, out_dim=32, kernel_size=3):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_IDX)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )
        # mean and max pooled halves are concatenated
        self.project = nn.Linear(2 * hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, tokens):
        mask = (tokens != PAD_IDX).float().unsqueeze(1)  # B,1,L
        h = self.conv(self.embed(tokens).transpose(1, 2)) * mask  # B,H,L
        denom = mask.sum(dim=2).clamp(min=1.0)
        mean = h.sum(dim=2) / denom
        mx = h.masked_fill(mask == 0, float("-inf")).max(dim=2).values
        mx = torch.nan_to_num(mx, neginf=0.0)
        return self.project(torch.cat([mean, mx], dim=1))
