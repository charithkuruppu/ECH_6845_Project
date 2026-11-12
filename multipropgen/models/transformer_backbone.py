"""Megalodon-inspired transformer backbone for MultiPropGen.

This module replaces the GVP message-passing stack inside the vector field. Everything around it
is unchanged: the atom/charge/time input embedding, the multi-property conditioning
(``properties_handle_method``), the CTMC flow matching of the categorical features, and the output
heads that predict coordinates, atom types, charges and bond orders.

The backbone is a stack of layers, and each layer is
    TransformerBlock         -> updates the invariant atom features h and bond features e
    EquivariantCoordUpdate   -> moves the atoms x using the features just computed

Ideas taken from Megalodon (Reidenbach et al. 2025, https://arxiv.org/abs/2505.18392):
  * atom, bond and geometric (distance) features are combined inside every transformer block;
  * multi-head attention learns how much each atom of the molecule contributes to each other atom;
  * attention-based feature updates alternate with geometric structure updates;
  * DiT-style adaptive layer norm (adaLN) conditions every block on the flow time;
  * the coordinate update has a cross-product term (as in Megalodon's XEGNN and the cross-product
    features of PropMolFlow's GVP), so the model is not forced to treat mirror images identically.

Attention runs over the edges of the batched DGL graph. MultiPropGen graphs are fully connected
within each molecule, so this is exact full attention inside each molecule, with no padding and no
attention between different molecules of the batch.
"""
import math

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from multipropgen.models.gvp import _norm_no_nan, _rbf


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of the flow time, shape (n_mols,) -> (n_mols, dim).

    t lies in [0, 1]; it is scaled by 1000 so the frequencies span that range the same way they
    span the 0..1000 integer steps used in DiT.
    """
    half = dim // 2
    dtype = t.dtype if t.is_floating_point() else torch.float32
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=dtype) / half)
    args = 1000.0 * t.to(dtype).unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def _mol_mean(x: torch.Tensor, node_batch_idx: torch.Tensor, n_mols: int) -> torch.Tensor:
    """Mean of a node quantity over each molecule, broadcast back to that molecule's nodes."""
    sums = torch.zeros((n_mols, x.shape[-1]), device=x.device, dtype=x.dtype).index_add_(0, node_batch_idx, x)
    counts = torch.bincount(node_batch_idx, minlength=n_mols).clamp(min=1).to(x.dtype).unsqueeze(-1)
    return (sums / counts)[node_batch_idx]


class SwiGLU(nn.Module):
    """Gated feed-forward network: W_out(SiLU(W_a x) * W_b x)."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.w_in = nn.Linear(dim, 2 * hidden)
        self.w_out = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(a) * b)


class TransformerBlock(nn.Module):
    """Updates atom features h (N, D) and bond features e (E, De); positions are not changed here.

    1. adaLN: per-molecule shift / scale / gate values computed from the conditioning vector.
    2. Each atom averages MLP([h_i, h_j, e_ij, rbf(d_ij)]) over its neighbours, so atom, bond and
       distance information are mixed before attention (as in Megalodon's DiTeBlock).
    3. Multi-head attention over the atoms of the same molecule. With ``pair_bias`` each head also
       gets a bias computed from the bond features and the distance of every atom pair.
    4. Gated residual updates of atoms and bonds, then SwiGLU feed-forward networks.
    """

    def __init__(self, n_node_feats: int, n_edge_feats: int, rbf_dim: int,
                 n_heads: int = 8, mlp_ratio: float = 4.0, pair_bias: bool = True):
        super().__init__()
        D, De = n_node_feats, n_edge_feats
        if D % n_heads != 0:
            raise ValueError(f'n_hidden_scalars ({D}) must be divisible by transformer n_heads ({n_heads})')
        self.n_heads = n_heads
        self.d_head = D // n_heads

        # adaLN modulation: shift/scale/gate for the attention and feed-forward sub-layers
        self.node_modulation = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D))
        self.edge_modulation = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * De))

        self.node_norm1 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.node_norm2 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.edge_norm1 = nn.LayerNorm(De, elementwise_affine=False, eps=1e-6)
        self.edge_norm2 = nn.LayerNorm(De, elementwise_affine=False, eps=1e-6)

        # atom + bond + distance mixing that feeds the attention
        self.mix = nn.Sequential(nn.Linear(2 * D + De + rbf_dim, D), nn.SiLU(), nn.Linear(D, D))

        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.q_norm = nn.LayerNorm(self.d_head)  # QK-norm keeps attention logits bounded
        self.k_norm = nn.LayerNorm(self.d_head)
        self.attn_out = nn.Linear(D, D)
        self.pair_bias = nn.Linear(De + rbf_dim, n_heads) if pair_bias else None

        self.node_ffn = SwiGLU(D, mlp_ratio)
        self.attn_to_edge = nn.Linear(D, De)
        self.edge_in = nn.Linear(De + rbf_dim, De)
        self.edge_ffn = SwiGLU(De, mlp_ratio)

        # adaLN-Zero (DiT): all gates start at 0, so every block starts out as the identity map
        for modulation in (self.node_modulation, self.edge_modulation):
            nn.init.zeros_(modulation[-1].weight)
            nn.init.zeros_(modulation[-1].bias)

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor, e: torch.Tensor, rbf: torch.Tensor,
                cond: torch.Tensor, node_batch_idx: torch.Tensor, edge_batch_idx: torch.Tensor):
        src, dst = g.edges()
        n_nodes = h.shape[0]

        (n_shift_a, n_scale_a, n_gate_a,
         n_shift_f, n_scale_f, n_gate_f) = self.node_modulation(cond)[node_batch_idx].chunk(6, dim=-1)
        (e_shift_a, e_scale_a, e_gate_a,
         e_shift_f, e_scale_f, e_gate_f) = self.edge_modulation(cond)[edge_batch_idx].chunk(6, dim=-1)

        h_mod = modulate(self.node_norm1(h), n_shift_a, n_scale_a)
        e_mod = modulate(self.edge_norm1(e), e_shift_a, e_scale_a)

        # combine atom, bond and geometric features: each atom averages messages from its neighbours
        messages = self.mix(torch.cat([h_mod[dst], h_mod[src], e_mod, rbf], dim=-1))
        h_mix = dgl.ops.copy_e_mean(g, messages)

        # multi-head attention between the atoms of each molecule (= the edges of the graph)
        q, k, v = self.qkv(h_mix).view(n_nodes, 3, self.n_heads, self.d_head).unbind(dim=1)
        q, k, v = self.q_norm(q), self.k_norm(k), v.contiguous()
        scores = dgl.ops.u_dot_v(g, k, q).squeeze(-1) / math.sqrt(self.d_head)  # (E, heads)
        if self.pair_bias is not None:
            scores = scores + self.pair_bias(torch.cat([e_mod, rbf], dim=-1))
        attn = dgl.ops.edge_softmax(g, scores)  # normalised over each atom's neighbours
        out = dgl.ops.u_mul_e_sum(g, v, attn.unsqueeze(-1))  # (N, heads, d_head)
        y = self.attn_out(out.reshape(n_nodes, -1))

        # gated residual updates of atoms and bonds
        h = h + n_gate_a * y
        e = e + e_gate_a * self.attn_to_edge(y[src] + y[dst])

        # feed-forward sub-layers (the bond one also sees the current distance)
        h = h + n_gate_f * self.node_ffn(modulate(self.node_norm2(h), n_shift_f, n_scale_f))
        e_in = self.edge_in(torch.cat([e, rbf], dim=-1))
        e = e + e_gate_f * self.edge_ffn(modulate(self.edge_norm2(e_in), e_shift_f, e_scale_f))
        return h, e


class EquivariantCoordUpdate(nn.Module):
    """Moves the atoms using the features just produced by the transformer block.

        x_i <- x_i + mean_j [ a_ij (x_i - x_j) / (|x_i - x_j| + 1)
                              + b_ij (x'_i cross x'_j) / (|x'_i cross x'_j| + 1) ]

    a_ij and b_ij are scalars predicted from (h_i, h_j, e_ij, rbf(d_ij)), and x' are coordinates
    centred on each molecule's centre. The first (EGNN) term is equivariant to rotations,
    translations and reflections. The cross-product term rotates with the molecule but flips sign
    under reflection, so mirror images (chirality) can be told apart.
    """

    def __init__(self, n_node_feats: int, n_edge_feats: int, rbf_dim: int,
                 use_cross_product: bool = True, clamp: float = 10.0):
        super().__init__()
        D = n_node_feats
        self.clamp = clamp
        self.node_norm = nn.LayerNorm(D)
        self.edge_norm = nn.LayerNorm(n_edge_feats)
        self.message = nn.Sequential(
            nn.Linear(2 * D + n_edge_feats + rbf_dim, D), nn.SiLU(),
            nn.Linear(D, D), nn.SiLU(),
        )
        self.radial_coef = nn.Linear(D, 1)
        self.cross_coef = nn.Linear(D, 1) if use_cross_product else None

        # start with tiny coordinate updates for stable early training (as in EGNN)
        for lin in (self.radial_coef, self.cross_coef):
            if lin is not None:
                nn.init.xavier_uniform_(lin.weight, gain=1e-3)
                nn.init.zeros_(lin.bias)

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor, e: torch.Tensor, x: torch.Tensor,
                rbf: torch.Tensor, node_batch_idx: torch.Tensor) -> torch.Tensor:
        src, dst = g.edges()
        hn, en = self.node_norm(h), self.edge_norm(e)
        m = self.message(torch.cat([hn[dst], hn[src], en, rbf], dim=-1))

        rel = x[dst] - x[src]
        rel = rel / (_norm_no_nan(rel, keepdims=True) + 1.0)
        update = rel * self.radial_coef(m).clamp(-self.clamp, self.clamp)

        if self.cross_coef is not None:
            xc = x - _mol_mean(x, node_batch_idx, g.batch_size)
            cross = torch.linalg.cross(xc[dst], xc[src], dim=-1)
            cross = cross / (_norm_no_nan(cross, keepdims=True) + 1.0)
            update = update + cross * self.cross_coef(m).clamp(-self.clamp, self.clamp)

        return x + dgl.ops.copy_e_mean(g, update)


class TransformerBackbone(nn.Module):
    """Stack of n_layers x (TransformerBlock -> EquivariantCoordUpdate).

    Input and output: atom features (N, D), bond features (E, De), positions (N, 3).
    """

    def __init__(self, n_node_feats: int, n_edge_feats: int, rbf_dim: int = 16, rbf_dmax: float = 20.0,
                 n_layers: int = 8, n_heads: int = 8, mlp_ratio: float = 4.0, pair_bias: bool = True,
                 use_cross_product: bool = True, coord_update_clamp: float = 10.0,
                 time_embedding_dim: int = 256):
        super().__init__()
        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax
        self.time_embedding_dim = time_embedding_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, n_node_feats), nn.SiLU(),
            nn.Linear(n_node_feats, n_node_feats),
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(n_node_feats, n_edge_feats, rbf_dim, n_heads=n_heads,
                             mlp_ratio=mlp_ratio, pair_bias=pair_bias)
            for _ in range(n_layers)
        ])
        self.coord_updates = nn.ModuleList([
            EquivariantCoordUpdate(n_node_feats, n_edge_feats, rbf_dim,
                                   use_cross_product=use_cross_product, clamp=coord_update_clamp)
            for _ in range(n_layers)
        ])
        self.final_node_norm = nn.LayerNorm(n_node_feats)
        self.final_edge_norm = nn.LayerNorm(n_edge_feats)

    def distance_features(self, g: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        src, dst = g.edges()
        d = _norm_no_nan(x[dst] - x[src])
        return _rbf(d, D_max=self.rbf_dmax, D_count=self.rbf_dim)

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor, e: torch.Tensor, x: torch.Tensor,
                t: torch.Tensor, node_batch_idx: torch.Tensor, edge_batch_idx: torch.Tensor,
                n_recycles: int = 1):
        """t is the flow time of each molecule, shape (n_mols,). Returns updated (h, e, x)."""
        cond = self.time_mlp(timestep_embedding(t, self.time_embedding_dim))  # (n_mols, D)
        for _ in range(n_recycles):
            for block, coord_update in zip(self.blocks, self.coord_updates):
                rbf = self.distance_features(g, x)  # distances from the current positions
                h, e = block(g, h, e, rbf, cond, node_batch_idx, edge_batch_idx)
                x = coord_update(g, h, e, x, rbf, node_batch_idx)
        return self.final_node_norm(h), self.final_edge_norm(e), x
