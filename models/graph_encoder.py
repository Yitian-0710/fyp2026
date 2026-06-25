"""
models/graph_encoder.py

Dense-adjacency GNN encoder for Graph-DQN V1.

This file intentionally does NOT depend on PyTorch Geometric. It uses dense
adjacency matrices returned by CascadingMitigationEnv and the normalized
adjacency tensor prepared by utils/graph_obs_utils.py.

Expected input batch format:
    batch = {
        "x": torch.Tensor,                # [B, N, F_node]
        "adj": torch.Tensor,              # [B, N, N]
        "adj_norm": torch.Tensor,         # [B, N, N], optional but preferred
        "global_features": torch.Tensor,  # [B, F_global], optional for encoder
        "action_mask": torch.Tensor,      # [B, A], not used here
    }

Main output:
    node_emb:  [B, N, hidden_dim]
    graph_emb: [B, hidden_dim]

Graph-DQN V1 purpose:
    Replace the vector-DQN flatten representation with a structure-aware graph
    representation. The encoder performs message passing with A_norm @ X so that
    node embeddings are influenced by neighboring nodes.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


PoolingType = Literal["mean", "sum", "max", "mean_max"]


class DenseGCNLayer(nn.Module):
    """
    One dense GCN-style message passing layer.

    Formula:
        H' = A_norm @ H @ W

    where A_norm is usually D^{-1/2}(A+I)D^{-1/2}.

    Parameters
    ----------
    in_dim:
        Input feature dimension.
    out_dim:
        Output feature dimension.
    use_bias:
        Whether to use bias in the linear transformation.
    activation:
        Activation name. Supported: "relu", "gelu", "tanh", "none".
    dropout:
        Dropout probability after activation.
    layer_norm:
        Whether to apply LayerNorm after linear/message passing.
    residual:
        Whether to add residual connection when dimensions match.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bias: bool = True,
        activation: str = "relu",
        dropout: float = 0.0,
        layer_norm: bool = True,
        residual: bool = True,
    ):
        super().__init__()

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.activation = activation.lower()
        self.dropout = float(dropout)
        self.use_residual = bool(residual and self.in_dim == self.out_dim)

        self.linear = nn.Linear(self.in_dim, self.out_dim, bias=use_bias)
        self.norm = nn.LayerNorm(self.out_dim) if layer_norm else nn.Identity()
        self.dropout_layer = nn.Dropout(self.dropout)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "tanh":
            return torch.tanh(x)
        if self.activation in {"none", "identity", "linear"}:
            return x

        raise ValueError(
            f"Unsupported activation={self.activation}. "
            "Expected 'relu', 'gelu', 'tanh', or 'none'."
        )

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Node features or hidden states, shape [B, N, F_in].
        adj_norm:
            Normalized adjacency matrix, shape [B, N, N].

        Returns
        -------
        torch.Tensor
            Updated node states, shape [B, N, F_out].
        """
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B,N,F], got {tuple(x.shape)}")
        if adj_norm.ndim != 3:
            raise ValueError(
                f"adj_norm must have shape [B,N,N], got {tuple(adj_norm.shape)}"
            )
        if x.shape[0] != adj_norm.shape[0] or x.shape[1] != adj_norm.shape[1]:
            raise ValueError(
                "x and adj_norm have inconsistent batch/node dimensions: "
                f"x={tuple(x.shape)}, adj_norm={tuple(adj_norm.shape)}"
            )

        residual = x

        # Message passing first: aggregate neighbor states.
        h = torch.bmm(adj_norm, x)  # [B, N, F_in]
        h = self.linear(h)          # [B, N, F_out]
        h = self.norm(h)
        h = self._activate(h)
        h = self.dropout_layer(h)

        if self.use_residual:
            h = h + residual

        return h


class GraphReadout(nn.Module):
    """
    Convert node embeddings to graph embeddings.

    Supported pooling:
        - mean
        - sum
        - max
        - mean_max: concatenate mean and max, then project to hidden_dim

    For Graph-DQN V1, mean pooling is usually stable enough.
    """

    def __init__(self, hidden_dim: int, pooling: PoolingType = "mean"):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.pooling = pooling

        if self.pooling == "mean_max":
            self.proj = nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.proj = nn.Identity()

    def forward(
        self,
        node_emb: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        node_emb:
            Node embeddings, shape [B, N, H].
        node_mask:
            Optional boolean mask, shape [B, N]. True means the node is included
            in readout. If None, all nodes are included.

        Returns
        -------
        torch.Tensor
            Graph embedding, shape [B, H].
        """
        if node_emb.ndim != 3:
            raise ValueError(
                f"node_emb must have shape [B,N,H], got {tuple(node_emb.shape)}"
            )

        if node_mask is None:
            if self.pooling == "mean":
                graph_emb = node_emb.mean(dim=1)
            elif self.pooling == "sum":
                graph_emb = node_emb.sum(dim=1)
            elif self.pooling == "max":
                graph_emb = node_emb.max(dim=1).values
            elif self.pooling == "mean_max":
                mean_emb = node_emb.mean(dim=1)
                max_emb = node_emb.max(dim=1).values
                graph_emb = torch.cat([mean_emb, max_emb], dim=-1)
                graph_emb = self.proj(graph_emb)
            else:
                raise ValueError(
                    f"Unsupported pooling={self.pooling}. "
                    "Expected 'mean', 'sum', 'max', or 'mean_max'."
                )
            return graph_emb

        if node_mask.ndim != 2:
            raise ValueError(
                f"node_mask must have shape [B,N], got {tuple(node_mask.shape)}"
            )

        mask = node_mask.bool().unsqueeze(-1)  # [B, N, 1]
        masked_node_emb = node_emb.masked_fill(~mask, 0.0)
        count = mask.sum(dim=1).clamp(min=1).float()  # [B, 1]

        if self.pooling == "mean":
            return masked_node_emb.sum(dim=1) / count

        if self.pooling == "sum":
            return masked_node_emb.sum(dim=1)

        if self.pooling in {"max", "mean_max"}:
            neg_inf = torch.finfo(node_emb.dtype).min
            max_input = node_emb.masked_fill(~mask, neg_inf)
            max_emb = max_input.max(dim=1).values
            max_emb = torch.where(torch.isfinite(max_emb), max_emb, torch.zeros_like(max_emb))

            if self.pooling == "max":
                return max_emb

            mean_emb = masked_node_emb.sum(dim=1) / count
            graph_emb = torch.cat([mean_emb, max_emb], dim=-1)
            return self.proj(graph_emb)

        raise ValueError(
            f"Unsupported pooling={self.pooling}. "
            "Expected 'mean', 'sum', 'max', or 'mean_max'."
        )


class DenseGraphEncoder(nn.Module):
    """
    Dense-adjacency GNN encoder for Graph-DQN.

    The encoder maps:
        node features + adjacency -> node embeddings + graph embedding

    Parameters
    ----------
    node_feature_dim:
        Dimension of node features from obs["node_features"].
    hidden_dim:
        Hidden dimension for node embeddings.
    num_layers:
        Number of GCN layers.
    global_feature_dim:
        Optional dimension of global graph features. If > 0 and
        use_global_features=True, global features are projected and fused into
        the graph embedding.
    pooling:
        Graph readout type.
    dropout:
        Dropout probability.
    activation:
        Activation for GCN layers.
    layer_norm:
        Whether to use layer norm in GCN layers.
    residual:
        Whether to use residual connections when possible.
    use_global_features:
        Whether to fuse obs["global_features"] into graph embedding.
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        global_feature_dim: int = 0,
        pooling: PoolingType = "mean",
        dropout: float = 0.0,
        activation: str = "relu",
        layer_norm: bool = True,
        residual: bool = True,
        use_global_features: bool = True,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        self.node_feature_dim = int(node_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.global_feature_dim = int(global_feature_dim)
        self.use_global_features = bool(use_global_features and global_feature_dim > 0)

        layers = []
        in_dim = self.node_feature_dim
        for _ in range(self.num_layers):
            layers.append(
                DenseGCNLayer(
                    in_dim=in_dim,
                    out_dim=self.hidden_dim,
                    activation=activation,
                    dropout=dropout,
                    layer_norm=layer_norm,
                    residual=residual,
                )
            )
            in_dim = self.hidden_dim

        self.layers = nn.ModuleList(layers)
        self.readout = GraphReadout(hidden_dim=self.hidden_dim, pooling=pooling)

        if self.use_global_features:
            self.global_proj = nn.Sequential(
                nn.Linear(self.global_feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.graph_fusion = nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.global_proj = None
            self.graph_fusion = None

    @staticmethod
    def normalize_adj(
        adj: torch.Tensor,
        add_self_loops: bool = True,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Fallback adjacency normalization.

        This mirrors utils.graph_obs_utils.normalize_adj so that the encoder can
        still work if batch does not contain 'adj_norm'.
        """
        if adj.ndim == 2:
            adj_work = adj.unsqueeze(0)
            squeeze_back = True
        elif adj.ndim == 3:
            adj_work = adj
            squeeze_back = False
        else:
            raise ValueError(f"adj must be [N,N] or [B,N,N], got {tuple(adj.shape)}")

        adj_work = adj_work.float()
        batch_size, num_nodes, _ = adj_work.shape

        if add_self_loops:
            eye = torch.eye(num_nodes, dtype=adj_work.dtype, device=adj_work.device)
            adj_work = adj_work + eye.unsqueeze(0).expand(batch_size, -1, -1)

        degree = adj_work.sum(dim=-1)
        degree_inv_sqrt = torch.pow(degree + eps, -0.5)
        adj_norm = degree_inv_sqrt.unsqueeze(-1) * adj_work * degree_inv_sqrt.unsqueeze(-2)

        if squeeze_back:
            return adj_norm.squeeze(0)
        return adj_norm

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        node_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a graph batch.

        Parameters
        ----------
        batch:
            Tensor batch from utils.graph_obs_utils.batch_graph_obs or
            obs_to_graph_tensor(add_batch_dim=True).
        node_mask:
            Optional boolean mask [B, N] for graph readout.

        Returns
        -------
        tuple
            node_emb:  [B, N, hidden_dim]
            graph_emb: [B, hidden_dim]
        """
        if "x" not in batch:
            raise KeyError("batch must contain key 'x'.")

        x = batch["x"].float()
        if x.ndim == 2:
            x = x.unsqueeze(0)

        if "adj_norm" in batch:
            adj_norm = batch["adj_norm"].float()
        elif "adj" in batch:
            adj_norm = self.normalize_adj(batch["adj"].float())
        else:
            raise KeyError("batch must contain either 'adj_norm' or 'adj'.")

        if adj_norm.ndim == 2:
            adj_norm = adj_norm.unsqueeze(0)

        h = x
        for layer in self.layers:
            h = layer(h, adj_norm)

        graph_emb = self.readout(h, node_mask=node_mask)

        if self.use_global_features and "global_features" in batch:
            global_features = batch["global_features"].float()
            if global_features.ndim == 1:
                global_features = global_features.unsqueeze(0)

            global_emb = self.global_proj(global_features)
            graph_emb = self.graph_fusion(torch.cat([graph_emb, global_emb], dim=-1))

        return h, graph_emb


# Backward-compatible alias.
GraphEncoder = DenseGraphEncoder
