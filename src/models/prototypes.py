from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PrototypeLosses:
    l_c: torch.Tensor
    l_e: torch.Tensor
    l_d: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.l_c + self.l_e + self.l_d


class PrototypeModule(nn.Module):
    """TimeXL prototype losses on segment embeddings (single pool Z, regression)."""

    def __init__(self, n_prototypes: int, d_model: int, d_min: float) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, d_model) * 0.02)
        self.d_min = d_min

    def forward(self, segments: torch.Tensor) -> tuple[torch.Tensor, PrototypeLosses]:
        # segments: [B, S, D] — treat all segments in batch as Z for loss
        z = segments.reshape(-1, segments.size(-1))
        p = self.prototypes
        dist_zp = torch.cdist(z, p, p=2).pow(2)
        l_c = dist_zp.min(dim=1).values.mean()
        dist_pz = torch.cdist(p, z, p=2).pow(2)
        l_e = dist_pz.min(dim=1).values.mean()
        pp = torch.cdist(p, p, p=2).pow(2)
        mask = torch.triu(torch.ones_like(pp, dtype=torch.bool), diagonal=1)
        hinge = F.relu(self.d_min - pp[mask])
        l_d = hinge.mean() if hinge.numel() else torch.zeros((), device=segments.device)
        losses = PrototypeLosses(l_c=l_c, l_e=l_e, l_d=l_d)

        sim = -dist_zp
        weights = torch.softmax(sim, dim=-1)
        proto_mix = weights @ p
        proto_mix = proto_mix.view(segments.size(0), segments.size(1), -1)
        return proto_mix, losses

    def project(self, bank: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest train segment embedding for each prototype (H3 projection).

        Returns indices and **squared** Euclidean distances.
        """
        z = bank
        p = self.prototypes.detach()
        dist = torch.cdist(p, z, p=2).pow(2)
        idx = dist.argmin(dim=1)
        return idx, dist.gather(1, idx.unsqueeze(1)).squeeze(1)

    def pairwise_min_dist(self) -> torch.Tensor:
        p = self.prototypes.detach()
        if p.size(0) < 2:
            return p.new_zeros(())
        d = torch.cdist(p, p, p=2)
        mask = torch.triu(torch.ones_like(d, dtype=torch.bool), diagonal=1)
        return d[mask].min()

    def nn_dist_mean(self, bank: torch.Tensor) -> torch.Tensor:
        _idx, dist_sq = self.project(bank)
        return dist_sq.sqrt().mean()

    def init_from_bank(self, bank: torch.Tensor, random_state: int = 0) -> None:
        """k-means++ seeding from a pre-injection segment bank (actual data points)."""
        from sklearn.cluster import kmeans_plusplus

        z = bank.detach().reshape(-1, bank.size(-1))
        n = int(self.prototypes.size(0))
        if z.size(0) < n:
            raise ValueError(f"bank has {z.size(0)} rows; need >= n_prototypes={n}")
        x = z.cpu().numpy()
        centers, _idx = kmeans_plusplus(x, n_clusters=n, random_state=random_state)
        with torch.no_grad():
            self.prototypes.copy_(
                torch.from_numpy(centers).to(
                    device=self.prototypes.device, dtype=self.prototypes.dtype
                )
            )
