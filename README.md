Below is the core code of our proposed PointTriPE model. The complete project will be released upon official paper acceptance.


```python
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


