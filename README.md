一. Below is the core code of our proposed PointTriPE model. The complete project will be released upon official paper acceptance.
1. MSPE (Multi-Scale Positional Encoding) is designed to capture global contextual information at multiple geometric scales. It begins by applying an MLP to the raw 3D coordinates to obtain an initial positional embedding. Then, a hierarchical point cloud pyramid is constructed, over which lightweight self-attention layers are applied at each scale to model global structures. Cross-scale feature fusion is used to integrate information from different resolutions. Finally, the aggregated positional priors are injected back into the point features through projection layers. MSPE provides stable and hierarchical spatial cues that enhance the model’s ability to capture long-range dependencies and semantic structures, all while maintaining computational efficiency.
<pre> 
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
</pre>
2. LGE (Local Geometric Encoding) focuses on capturing fine-grained geometric patterns in local neighborhoods, such as edges, curvatures, and local structural variations. For each center point, LGE encodes the relative offsets $\Delta p$ between the point and its $k$ nearest neighbors using a shared lightweight MLP. The resulting features are aggregated via max pooling to obtain a geometry-aware local positional embedding. This module operates without relying on any rigid structural assumptions, is highly efficient, and significantly enhances the model’s sensitivity to boundary regions and geometric details. LGE serves to complement the global cues by injecting fine-scale spatial information into the point representations.
<pre> 
class LGE(nn.Module):  
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
    def forward(self, xyz, pwd):  
        B = xyz.size(0)  
        _, knn = pwd[:, :self.n, :self.n].topk(k=self.k, dim=-1, largest=False)  
        xyz_new = xyz[:, :self.n].contiguous()  
        nbr = (index_points(xyz_new, knn) - xyz_new.unsqueeze(-2)).view(-1, 3)  
        nbr = self.nbr_embed(nbr).view(B * self.n, self.k, -1).max(dim=1)[0]  
        nbr = self.nbr_proj(nbr)  
        nbr = self.nbr_bn(nbr).view(B, self.n, -1)  
        return xyz_new
</pre>
3.RPE (Relative Positional Encoding) is designed to model pairwise spatial relationships between points, bridging the mid-scale geometric semantics not captured by global or local encodings. Given the relative offset $\Delta p_{ij}$ between two points, RPE uses a gated Mixture-of-Experts (MoE) structure to produce a dynamic embedding. Specifically, a gating network assigns weights to multiple experts based on the offset, and each expert performs soft assignment over a directional codebook to encode fine-grained geometric variations. The two-stage soft mapping serves as a nonlinear kernel approximation, enabling the modeling of directional, asymmetric, and diverse spatial dependencies. Unlike static encodings, RPE modifies the Key and Value vectors in attention computation, allowing the attention mechanism to be geometry-aware at the pairwise level.   
<pre>
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
</pre>
二. Overview of our proposed network architecture. The left side illustrates a U-Net–style point transformer framework
used for semantic segmentation and classification. The right side shows the core attention-based feature extraction module
integrating our triadic positional encoding.
<img width="1224" height="478" alt="2025-08-05 19-03-01 的屏幕截图" src="https://github.com/user-attachments/assets/ac0055d8-9d67-4d3e-981b-48b21b350f47" />
三. Illustration of the proposed triadic positional encoding modules. From left to right: Multi-Scale Positional Encoding,
Local Geometric Encoding, and Relative Positional Encoding.
<img width="1194" height="420" alt="2025-08-05 19-03-08 的屏幕截图" src="https://github.com/user-attachments/assets/85aebfdf-4056-487c-b21d-6a6ed9c16c3c" />
四. Experiments
<img width="654" height="596" alt="2025-08-05 19-05-13 的屏幕截图" src="https://github.com/user-attachments/assets/1a1352ee-27c3-41b2-b151-f4f49c66e9c8" />
<img width="620" height="516" alt="2025-08-05 19-05-30 的屏幕截图" src="https://github.com/user-attachments/assets/2233871a-ad12-4773-be16-f161cd0290d4" />
<img width="650" height="508" alt="2025-08-05 19-05-34 的屏幕截图" src="https://github.com/user-attachments/assets/37187347-cef2-4207-8ca2-dc3c45486ff3" />
<img width="1362" height="344" alt="2025-08-05 19-05-55 的屏幕截图" src="https://github.com/user-attachments/assets/669b4f11-b78a-41ff-87ec-ed79143b8a16" />
<img width="660" height="442" alt="2025-08-05 19-06-08 的屏幕截图" src="https://github.com/user-attachments/assets/9d39485b-0be9-43e6-a9b3-3de005957cd3" />


