"""FP04 shared recurrent observed-cell predictor.

Only routing/input representation differs between arms.  Preprocessing (observed
RMS scaling, clipping, interventions and literal PCFI) belongs to the caller.
See IMPLEMENTATION_METHOD.md for exact statistics and normalization conventions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

# Required import order for this Windows research environment.
import sklearn  # noqa: F401
import torch
from torch import Tensor, nn


def _mlp(inputs: int, width: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, width), nn.GELU(), nn.Linear(width, outputs))


class ObservedCellPFN(nn.Module):
    """Class-equivariant graph/column recurrent predictor with shared parameters.

    ``forward`` returns unnormalized logits [nodes, supplied classes].  No labels
    are clamped at output and no softmax is applied within a class chunk.
    """

    def __init__(self, width: int = 32, rank: int = 4, rounds: int = 4) -> None:
        super().__init__()
        if width < 1 or rank < 1 or rounds < 0:
            raise ValueError("width/rank must be positive and rounds nonnegative")
        self.width = int(width)
        self.rank = int(rank)
        self.rounds = int(rounds)
        self.residual = 0.25

        self.value_mlp = nn.Sequential(_mlp(7, width, rank), nn.Tanh())
        # 0 = unknown; 1 = this class's context; 2 = another class's context.
        self.seed_embedding = nn.Embedding(3, width)
        self.node_mlp = _mlp(7, width, width)
        self.graph_mlp = _mlp(2 * width, width, width)
        self.graph_norm = nn.LayerNorm(width)
        self.P = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(rank))
        self.column_mlp = _mlp(width + 3, width, width)
        self.column_norm = nn.LayerNorm(width)
        self.V = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(rank))
        self.read_mlp = _mlp(2 * width, width, width)
        self.read_norm = nn.LayerNorm(width)
        # A class-shared final scalar bias cancels from the class softmax and
        # is unidentifiable under CE, so do not introduce that parameter.
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1, bias=False))

    @staticmethod
    def _observed_statistics(x: Tensor, observed: Tensor, degree: Tensor) -> tuple[Tensor, Tensor]:
        """Return node base [N,5] and column statistics [F,3], visible only."""
        nodes, features = x.shape
        mask = observed.to(dtype=x.dtype)
        row_count = mask.sum(dim=1)
        col_count = mask.sum(dim=0)
        row_denom = row_count.clamp_min(1)
        col_denom = col_count.clamp_min(1)
        row_mean = x.sum(dim=1) / row_denom
        row_second = x.square().sum(dim=1) / row_denom
        col_mean = x.sum(dim=0) / col_denom
        col_second = x.square().sum(dim=0) / col_denom
        node_base = torch.stack(
            (
                row_mean.tanh(),
                row_second.tanh(),
                row_count.log1p() / math.log1p(max(features, 1)),
                row_count / max(features, 1),
                degree.log1p() / math.log1p(max(nodes, 1)),
            ),
            dim=-1,
        )
        column_stats = torch.stack(
            (
                col_mean.tanh(),
                col_second.tanh(),
                col_count.log1p() / math.log1p(max(nodes, 1)),
            ),
            dim=-1,
        )
        return node_base, column_stats

    @staticmethod
    def _basis_inputs(value: Tensor, observed: Tensor, column_stats: Tensor) -> Tensor:
        """Seven basis inputs; column_stats is broadcastable to value[...,3]."""
        value = value.clamp(-10.0, 10.0)
        return torch.cat(
            (
                (value / 10).unsqueeze(-1),
                (value.abs() / 10).unsqueeze(-1),
                (value.square() / 100).unsqueeze(-1),
                observed.to(dtype=value.dtype).unsqueeze(-1),
                column_stats.expand(*value.shape, 3),
            ),
            dim=-1,
        )

    def _initial_state(
        self,
        context: Tensor,
        node_base: Tensor,
        class_counts: Tensor,
        class_ids: Tensor,
    ) -> Tensor:
        nodes = context.numel()
        channels = class_ids.numel()
        positive = context.unsqueeze(0) == class_ids.unsqueeze(1)
        known = context.unsqueeze(0) >= 0
        seeds = torch.where(known, torch.where(positive, 1, 2), 0)
        total = class_counts.sum()
        support = (class_counts[class_ids] / total.clamp_min(1)).view(channels, 1, 1)
        context_fraction = (total / nodes).expand(channels, nodes, 1)
        stats = torch.cat(
            (node_base.unsqueeze(0).expand(channels, -1, -1), support.expand(-1, nodes, -1), context_fraction),
            dim=-1,
        )
        return self.seed_embedding(seeds) + self.node_mlp(stats)

    def _graph_update(self, h: Tensor, src: Tensor, dst: Tensor, degree: Tensor) -> Tensor:
        neighbor_sum = torch.zeros_like(h)
        neighbor_sum.index_add_(1, dst, h[:, src, :])
        neighbor_mean = neighbor_sum / degree.clamp_min(1).view(1, -1, 1)
        return self.graph_norm(h + self.residual * self.graph_mlp(torch.cat((h, neighbor_mean), dim=-1)))

    def _make_feature_chunks(
        self,
        x: Tensor,
        values: Tensor,
        observed: Tensor,
        column_stats: Tensor,
        arm: str,
        feature_chunk: int,
        sparse: bool,
    ) -> list[dict[str, Any]]:
        """Cache scalar bases, never per-cell width-dimensional hidden states."""
        chunks: list[dict[str, Any]] = []
        for start in range(0, x.shape[1], feature_chunk):
            stop = min(start + feature_chunk, x.shape[1])
            mask = observed[:, start:stop]
            stats = column_stats[start:stop]
            if arm == "observed" and sparse:
                rows, columns = mask.nonzero(as_tuple=True)
                value = values[rows, start + columns]
                inputs = self._basis_inputs(value, mask[rows, columns], stats[columns])
                basis = self.value_mlp(inputs)
                chunks.append({"start": start, "stop": stop, "rows": rows, "columns": columns, "basis": basis})
            else:
                value = values[:, start:stop]
                inputs = self._basis_inputs(value, mask, stats.unsqueeze(0))
                basis = self.value_mlp(inputs)
                if arm == "observed":
                    basis = basis * mask.unsqueeze(-1)
                chunks.append({"start": start, "stop": stop, "basis": basis})
        return chunks

    def _column_read(
        self,
        h: Tensor,
        chunks: list[dict[str, Any]],
        column_stats: Tensor,
        row_counts: Tensor,
        column_counts: Tensor,
    ) -> Tensor:
        projected = torch.stack([layer(h) for layer in self.P], dim=2)  # [C,N,R,D]
        read_sum = torch.zeros_like(h)
        for chunk in chunks:
            start, stop, basis = chunk["start"], chunk["stop"], chunk["basis"]
            if "rows" in chunk:
                rows, columns = chunk["rows"], chunk["columns"]
                u = h.new_zeros((h.shape[0], stop - start, self.width))
                for r in range(self.rank):
                    contribution = projected[:, rows, r, :] * basis[:, r].view(1, -1, 1)
                    u.index_add_(1, columns, contribution)
            else:
                u = torch.einsum("nfr,cnrd->cfd", basis, projected)
            u = u / column_counts[start:stop].clamp_min(1).view(1, -1, 1)
            stats = column_stats[start:stop].unsqueeze(0).expand(h.shape[0], -1, -1)
            z = self.column_norm(self.column_mlp(torch.cat((u, stats), dim=-1)))
            if "rows" in chunk:
                for r in range(self.rank):
                    contribution = self.V[r](z)[:, columns, :] * basis[:, r].view(1, -1, 1)
                    read_sum.index_add_(1, rows, contribution)
            else:
                projected_z = torch.stack([layer(z) for layer in self.V], dim=2)
                read_sum = read_sum + torch.einsum("nfr,cfrd->cnd", basis, projected_z)
        return read_sum / row_counts.clamp_min(1).view(1, -1, 1)

    def forward(
        self,
        batch: Mapping[str, Any],
        arm: str = "observed",
        class_chunk: int = 1,
        feature_chunk: int = 256,
        sparse: bool = True,
        disable_columns: bool = False,
    ) -> Tensor:
        if arm not in ("observed", "mask", "pcfi"):
            raise ValueError("arm must be observed, mask or pcfi")
        if class_chunk < 1 or feature_chunk < 1:
            raise ValueError("class_chunk and feature_chunk must be positive")
        parameter = self.seed_embedding.weight
        x = torch.as_tensor(batch["x"], device=parameter.device, dtype=parameter.dtype)
        observed = torch.as_tensor(batch["observed"], device=x.device, dtype=torch.bool)
        if x.ndim != 2 or observed.shape != x.shape or x.shape[0] < 1:
            raise ValueError("x and observed must have the same [N,F] shape, N>=1")
        # Sanitize before squares, moments and every learned operation.  Hidden
        # NaNs/huge sentinels therefore never enter the observed or mask arm.
        x = torch.where(observed, x, torch.zeros_like(x))
        nodes, features = x.shape
        context = torch.as_tensor(batch["context"], device=x.device, dtype=torch.long)
        src = torch.as_tensor(batch["src"], device=x.device, dtype=torch.long)
        dst = torch.as_tensor(batch["dst"], device=x.device, dtype=torch.long)
        classes = int(batch["classes"])
        if context.shape != (nodes,) or src.ndim != 1 or src.shape != dst.shape or classes < 1:
            raise ValueError("invalid context, directed edge vectors or classes")
        degree = torch.bincount(dst, minlength=nodes).to(dtype=x.dtype)
        node_base, column_stats = self._observed_statistics(x, observed, degree)
        class_counts = torch.bincount(context[context >= 0], minlength=classes).to(dtype=x.dtype)
        if class_counts.numel() != classes:
            raise ValueError("context contains a label outside the supplied vocabulary")

        values = x
        if arm == "pcfi":
            values = torch.as_tensor(batch["pcfi"], device=x.device, dtype=x.dtype)
            if values.shape != x.shape:
                raise ValueError("pcfi must have shape [N,F]")
            # Clamp known values even if a caller's interpolation roundoff differs.
            values = torch.where(observed, x, values)
        if arm == "observed":
            row_counts = observed.sum(dim=1).to(dtype=x.dtype)
            column_counts = observed.sum(dim=0).to(dtype=x.dtype)
        else:
            row_counts = x.new_full((nodes,), features)
            column_counts = x.new_full((features,), nodes)
        chunks = [] if disable_columns else self._make_feature_chunks(
            x, values, observed, column_stats, arm, feature_chunk, sparse
        )
        logits: list[Tensor] = []
        for start in range(0, classes, class_chunk):
            class_ids = torch.arange(start, min(start + class_chunk, classes), device=x.device)
            h = self._initial_state(context, node_base, class_counts, class_ids)
            for _ in range(self.rounds):
                h = self._graph_update(h, src, dst, degree)
                read = torch.zeros_like(h) if disable_columns else self._column_read(
                    h, chunks, column_stats, row_counts, column_counts
                )
                h = self.read_norm(h + self.residual * self.read_mlp(torch.cat((h, read), dim=-1)))
            logits.append(self.head(h).squeeze(-1).transpose(0, 1))
        return torch.cat(logits, dim=1)


__all__ = ["ObservedCellPFN"]
