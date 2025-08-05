<style>
pre {
  overflow-x: auto;
  padding: 12px 16px;
  background-color: #f7f7f7;
  border: 1px solid #ddd;
  border-radius: 5px;
  font-family: SFMono-Regular, Consolas, Liberation Mono, Menlo, monospace;
  font-size: 14px;
  line-height: 1.5;
  white-space: pre;
  word-break: normal;
  tab-size: 4;
  margin-bottom: 1em;
}
</style>

# PointTriPE Demo

Below is the core code of our proposed PointTriPE model. The complete project will be released upon official paper acceptance.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.models.layers import DropPath
from utils.cutils import knn_edge_maxpooling
from pointnet2_ops import pointnet2_utils


def init_linear(layer, std=0.02):
    trunc_normal_(layer.weight, std=std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.)


def index_points(points, idx):
    B = points.shape[0]
    view_shape = list(idx.shape)
    C = points.shape[-1]
    if idx.dim() == 3:

        view_shape.append(1)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, C)
        pts_expanded = points.unsqueeze(1).expand(-1, view_shape[1], -1, -1)
        return torch.gather(pts_expanded, 2, idx_expanded)
    else:

        view_shape.append(C)
        batch_indices = torch.arange(B, device=points.device).view(B, *([1] * (idx.dim() - 1))).expand_as(idx)
        return points[batch_indices, idx, :]


def calc_pwd(x, eps=1e-6):
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    dist2 = x2 + x2.transpose(1, 2) - 2 * x @ x.transpose(1, 2)
    return dist2.clamp(min=eps)


class LFP(nn.Module):

    def __init__(self, in_dim, out_dim, bn_momentum, init=0.):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, momentum=bn_momentum)
        nn.init.constant_(self.bn.weight, init)

    def forward(self, x, knn):
        B, N, C = x.shape
        x = self.proj(x)
        x = knn_edge_maxpooling(x, knn, self.training)
        x = self.bn(x.view(B * N, -1)).view(B, N, -1)
        return x


class Mlp(nn.Module):
    def __init__(self, in_dim, mlp_ratio, bn_momentum, act, init=0.):
        super().__init__()
        hid_dim = round(in_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            act(),
            nn.Linear(hid_dim, in_dim, bias=False),
            nn.BatchNorm1d(in_dim, momentum=bn_momentum),
        )
        nn.init.constant_(self.mlp[-1].weight, init)

    def forward(self, x):
        B, N, C = x.shape
        return self.mlp(x.view(B * N, -1)).view(B, N, -1)


class MoECodebookPE(nn.Module):

    def __init__(self, K: int, D: int, E: int = 8, hidden: int = 32):
        super().__init__()
        self.E = E
        self.K = K
        self.D = D

        self.codebooks = nn.Parameter(torch.randn(E, K, D))

        self.gate = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, E)
        )

        self.expert_proj = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, K)
        )

    def forward(self, rel_pos: torch.Tensor) -> torch.Tensor:
        B, N, M, _ = rel_pos.shape
        rp = rel_pos.view(-1, 3)

        gate_logits = self.gate(rp)
        gate_w = F.softmax(gate_logits, dim=-1)

        expert_logits = self.expert_proj(rp)
        expert_w = F.softmax(expert_logits, dim=-1)

        expert_w_expand = expert_w
        cb = self.codebooks
        pe_experts = torch.einsum('bk,ekd->bed', expert_w_expand, cb)

        gate_w_expand = gate_w.unsqueeze(-1)
        pe = (gate_w_expand * pe_experts).sum(dim=1)

        pe = pe.view(B, N, M, self.D)
        return pe


class Q_KV_Attention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        n_bins = 16
        rpe_dim = 64
        self.dim = dim
        self.num_heads = num_heads

        self.rpe_net = MoECodebookPE(K=n_bins, D=rpe_dim)

        self.rpe_proj = nn.ModuleDict({
            'k': nn.Linear(rpe_dim, dim),
            'v': nn.Linear(rpe_dim, dim)
        })

        for m in self.rpe_proj.values():
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        self.lbr = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU()
        )

    def forward(self, xyz: torch.Tensor, x: torch.Tensor, knn: torch.Tensor):
        xs = index_points(x, knn)
        xyzs = index_points(xyz, knn)
        B, N, K, C = xs.shape

        rel_pos = xyzs - xyz.unsqueeze(2)
        pe = self.rpe_net(rel_pos)

        k_rpe = self.rpe_proj['k'](pe).view(B * N, K, C)
        v_rpe = self.rpe_proj['v'](pe).view(B * N, K, C)

        q = x.view(B * N, 1, C)
        k = xs.view(B * N, K, C) + k_rpe
        v = xs.view(B * N, K, C) + v_rpe

        out, _ = self.attn(q, k, v)
        out = self.lbr(out)
        out = out.view(B, N, C)
        return out


class MultiScaleAttentionPE(nn.Module):
    def __init__(self, embed_dim, num_heads=2):
        super().__init__()
        C = embed_dim

        self.mlp0 = nn.Sequential(
            nn.Linear(3, C),
            nn.ReLU(),
            nn.Linear(C, C)
        )

        self.cls2 = nn.Parameter(torch.zeros(1, 1, C))
        nn.init.trunc_normal_(self.cls2, std=0.02)

        self.attn2 = nn.MultiheadAttention(embed_dim=C, num_heads=num_heads, batch_first=True)
        self.proj2 = nn.Linear(2 * C, C)
        self.proj1 = nn.Linear(2 * C, C)
        self.proj0 = nn.Linear(2 * C, C)

    def forward(self, xyz0, xyz1, xyz2, pwd):
        B, N0, _ = xyz0.shape
        _, N1, _ = xyz1.shape
        _, N2, _ = xyz2.shape
        f0 = self.mlp0(xyz0)
        f1 = f0[:, :N1, :]
        f2 = f0[:, :N2, :]

        cls2 = self.cls2.expand(B, -1, -1)
        seq2 = torch.cat([f2, cls2], dim=1)

        attn_out2, _ = self.attn2(seq2, seq2, seq2)

        feat2_pe = torch.cat([attn_out2[:, N2:, :].expand(-1, N2, -1), attn_out2[:, :N2, :]], dim=-1)
        feat2_pe = self.proj2(feat2_pe)

        dist12, knn12 = pwd[:, :N1, :N2].topk(k=3, dim=-1, largest=False)
        f2_nei = index_points(feat2_pe, knn12)

        dist_recip = 1.0 / dist12
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weights12 = dist_recip / norm

        weights12 = weights12 / (weights12.sum(dim=-1, keepdim=True) + 1e-6)
        weights12 = weights12.unsqueeze(-1)
        f12 = (f2_nei * weights12).sum(dim=2)

        feat1_pe = torch.cat([f12, f1], dim=-1)
        feat1_pe = self.proj1(feat1_pe)

        dist01, knn01 = pwd[:, :N0, :N1].topk(k=3, dim=-1, largest=False)
        f1_nei = index_points(feat1_pe, knn01)

        dist_recip = 1.0 / dist01
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weights01 = dist_recip / norm

        weights01 = weights01 / (weights01.sum(dim=-1, keepdim=True) + 1e-6)
        weights01 = weights01.unsqueeze(-1)
        f01 = (f1_nei * weights01).sum(dim=2)

        feat0_pe = torch.cat([f01, f0], dim=-1)
        feat0_pe = self.proj0(feat0_pe)
        return feat2_pe, feat1_pe, feat0_pe


class Block(nn.Module):
    def __init__(self, dim, depth, drop_path, mlp_ratio, bn_momentum, act):
        super().__init__()
        self.depth = depth
        self.lfps = nn.ModuleList([Q_KV_Attention(dim, dim // 48) for _ in range(depth)])
        self.mlp = Mlp(dim, mlp_ratio, bn_momentum, act, init=0.2)
        self.mlps = nn.ModuleList([Mlp(dim, mlp_ratio, bn_momentum, act) for _ in range(depth // 2)])
        drop_rates = drop_path if isinstance(drop_path, list) else torch.linspace(0., drop_path, depth).tolist()
        self.drop_paths = nn.ModuleList([DropPath(dpr) for dpr in drop_rates])

        self.gpe = nn.Linear(64, dim, bias=False)

    def forward(self, xyz, x, knn, g_pos):
        x = x + self.gpe(g_pos)
        x = x + self.drop_paths[0](self.mlp(x))

        x = x + self.drop_paths[0](self.lfps[0](xyz, x, knn))
        x = x + self.drop_paths[0](self.mlps[0](x))

        x = x + self.drop_paths[1](self.lfps[1](xyz, x, knn))
        x = x + self.drop_paths[1](self.mlps[1](x))

        return x


class SetAbstraction(nn.Module):

    def __init__(self, args, depth):
        super().__init__()
        self.depth = depth
        self.n = args.ns[depth]
        self.dim = args.dims[depth]
        self.k = args.ks[depth]
        self.first = (depth == 0)

        nbr_in = 3
        nbr_hid = args.nbr_dims[0] if self.first else args.nbr_dims[1] // 2
        nbr_out = self.dim if self.first else args.nbr_dims[1]
        self.nbr_embed = nn.Sequential(
            nn.Linear(nbr_in, nbr_hid // 2, bias=False),
            nn.BatchNorm1d(nbr_hid // 2, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(nbr_hid // 2, nbr_hid, bias=False),
            nn.BatchNorm1d(nbr_hid, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(nbr_hid, nbr_out, bias=False)
        )
        self.nbr_bn = nn.BatchNorm1d(self.dim, momentum=args.bn_momentum)
        nn.init.constant_(self.nbr_bn.weight, 0.8 if self.first else 0.2)
        self.nbr_proj = nn.Identity() if self.first else nn.Linear(nbr_out, self.dim, bias=False)

        if not self.first:
            in_dim = args.dims[depth - 1]
            self.lfp = LFP(in_dim, self.dim, args.bn_momentum, init=0.3)
            self.skip = nn.Sequential(
                nn.Linear(in_dim, self.dim, bias=False),
                nn.BatchNorm1d(self.dim, momentum=args.bn_momentum)
            )

        self.block = Block(self.dim, args.depths[depth], args.drop_paths[depth], args.mlp_ratio, args.bn_momentum,
                           args.act)

    def forward(self, xyz, x, pwd, g_pos):
        B = xyz.size(0)

        _, knn = pwd[:, :self.n, :self.n].topk(k=self.k, dim=-1, largest=False)

        if self.depth != 0:
            x = x[:, :self.n].contiguous()
            B0, N0, C0 = x.shape
            x_skip = self.skip(x.view(B0 * N0, C0)).view(B0, N0, -1)
            x_prop = self.lfp(x, knn)
            x_new = x_skip + x_prop

        xyz_new = xyz[:, :self.n].contiguous()
        nbr = (index_points(xyz_new, knn) - xyz_new.unsqueeze(-2)).view(-1, 3)
        nbr = self.nbr_embed(nbr).view(B * self.n, self.k, -1).max(dim=1)[0]
        nbr = self.nbr_proj(nbr)
        nbr = self.nbr_bn(nbr).view(B, self.n, -1)
        x_new = nbr if self.first else nbr + x_new

        x_new = self.block(xyz_new, x_new, knn, g_pos)
        return xyz_new, x_new


class TriPE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sa0 = SetAbstraction(args, 0)
        self.sa1 = SetAbstraction(args, 1)
        self.sa2 = SetAbstraction(args, 2)
        self.proj = nn.Sequential(
            nn.BatchNorm1d(args.dims[-1], momentum=args.bn_momentum),
            nn.Linear(args.dims[-1], args.bottleneck),
            args.act()
        )

        self.head = nn.Sequential(
            nn.Linear(args.bottleneck, 512, bias=False),
            nn.BatchNorm1d(512, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256, momentum=args.bn_momentum),
            args.act(),
            nn.Dropout(.5),
            nn.Linear(256, args.num_classes)
        )

        self.apply(self._init_weights)

        self.pe0 = MultiScaleAttentionPE(64)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz):

        idx = pointnet2_utils.furthest_point_sample(xyz, 1024).long()
        xyz = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        pwd = calc_pwd(xyz)

        xyz0 = xyz[:, :1024].contiguous()
        xyz1 = xyz[:, :256].contiguous()
        xyz2 = xyz[:, :64].contiguous()
        xyz_pe2, xyz_pe1, xyz_pe0 = self.pe0(xyz0, xyz1, xyz2, pwd)

        xyz0, x0 = self.sa0(xyz, None, pwd, xyz_pe0)
        xyz1, x1 = self.sa1(xyz0, x0, pwd, xyz_pe1)
        xyz2, x2 = self.sa2(xyz1, x1, pwd, xyz_pe2)

        B, N, C = x2.shape
        x_flat = self.proj(x2.view(B * N, C)).view(B, N, -1)
        x_feat = x_flat.max(dim=1)[0]
        return self.head(x_feat)

