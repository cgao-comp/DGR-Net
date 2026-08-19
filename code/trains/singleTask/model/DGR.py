import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets import BertTextEncoder


class ComplementaryInformationExtractor(nn.Module):
    """Dissimilarity-aware attention over the modality-specific memory."""
    def __init__(self, query_dim, key_dim, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(key_dim, embed_dim)
        self.v_proj = nn.Linear(key_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # A signed coefficient is intentional: negative values attend to dissimilar cues.
        self.mode_factor = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory):
        batch, memory_len, _ = memory.shape
        q = self.q_proj(query).view(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(batch, memory_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(batch, memory_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = self.dropout(F.softmax(self.mode_factor * scores, dim=-1))
        retrieved = (weights @ v).transpose(1, 2).reshape(batch, -1)
        return self.out_proj(retrieved), weights


class RCR_Module(nn.Module):
    def __init__(self, fused_dim, private_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.extractor = ComplementaryInformationExtractor(
            fused_dim, private_dim, fused_dim, num_heads=num_heads, dropout=dropout
        )
        self.gate = nn.Sequential(nn.Linear(fused_dim * 2, fused_dim // 2), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(fused_dim // 2, 1), nn.Sigmoid())
        self.norm = nn.LayerNorm(fused_dim)
        # Stable early training, as specified in the manuscript.
        last = self.gate[-2]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, math.log(0.1 / 0.9))

    def forward(self, fused, private_features):
        memory = torch.cat(private_features, dim=2).transpose(1, 2)  # B, T_l+T_a+T_v, d
        retrieved, attention = self.extractor(fused, memory)
        gate = self.gate(torch.cat([fused, retrieved], dim=-1))
        return self.norm(fused + gate * retrieved), gate, attention


class HybridModalityGate(nn.Module):
    def __init__(self, dim, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.semantic_net = nn.Sequential(nn.Linear(3 * dim, 3 * dim // 2), nn.ReLU(), nn.Dropout(0.2),
                                          nn.Linear(3 * dim // 2, 3))
        # Softplus guarantees a non-negative uncertainty penalty while retaining gradients.
        self.uncertainty_scale_raw = nn.Parameter(torch.tensor(0.54132485))  # softplus ~= 1.0

    def forward(self, h_l, h_a, h_v, cls_l, cls_a, cls_v):
        semantic_scores = self.semantic_net(torch.cat([h_l, h_a, h_v], dim=-1))
        energies = torch.stack([
            -self.temperature * torch.logsumexp(cls_l / self.temperature, dim=-1),
            -self.temperature * torch.logsumexp(cls_a / self.temperature, dim=-1),
            -self.temperature * torch.logsumexp(cls_v / self.temperature, dim=-1),
        ], dim=-1)
        uncertainty_scale = F.softplus(self.uncertainty_scale_raw)
        weights = F.softmax(semantic_scores - uncertainty_scale * energies, dim=-1)
        return weights, energies, semantic_scores


class DGR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_bert = args.use_bert
        if self.use_bert:
            self.text_model = BertTextEncoder(args.use_finetune, args.transformers, args.pretrained)
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = args.dst_feature_dim_nheads[0]
        self.num_heads = args.dst_feature_dim_nheads[1]
        self.text_dropout, self.output_dropout = args.text_dropout, args.output_dropout
        self.num_classes = 7

        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, args.conv1d_kernel_size_l, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_l, args.conv1d_kernel_size_a, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_l, args.conv1d_kernel_size_v, bias=False)
        self.enc_private_l, self.enc_private_a, self.enc_private_v = (nn.Conv1d(self.d_l, self.d_l, 1, bias=False) for _ in range(3))
        # One encoder shared by all modalities, matching Eq. 2.
        self.enc_shared = nn.Conv1d(self.d_l, self.d_l, 1, bias=False)
        self.decoder_l, self.decoder_a, self.decoder_v = (nn.Conv1d(2 * self.d_l, self.d_l, 1, bias=False) for _ in range(3))

        self.cross_attention = nn.MultiheadAttention(self.d_l, self.num_heads, dropout=args.attn_dropout, batch_first=True)
        self.context_norm = nn.ModuleList([nn.LayerNorm(self.d_l) for _ in range(3)])
        self.context_ffn = nn.ModuleList([nn.Sequential(nn.Linear(self.d_l, 4 * self.d_l), nn.ReLU(), nn.Dropout(args.relu_dropout), nn.Linear(4 * self.d_l, self.d_l)) for _ in range(3)])
        self.projectors = nn.ModuleList([nn.Linear(self.d_l, self.d_l) for _ in range(3)])
        self.reg_heads = nn.ModuleList([nn.Linear(self.d_l, 1) for _ in range(3)])
        self.cls_heads = nn.ModuleList([nn.Linear(self.d_l, self.num_classes) for _ in range(3)])
        self.modality_gate = HybridModalityGate(self.d_l, temperature=1.0)

        self.fusion_dim = 3 * self.d_l
        self.rcr_module = RCR_Module(
            self.fusion_dim, self.d_l, num_heads=self.num_heads, dropout=args.attn_dropout
        )
        self.contrastive_head = nn.Sequential(nn.Linear(self.fusion_dim, self.fusion_dim), nn.ReLU(), nn.Linear(self.fusion_dim, 128))
        self.proj1, self.proj2, self.out_layer = nn.Linear(self.fusion_dim, self.fusion_dim), nn.Linear(self.fusion_dim, self.fusion_dim), nn.Linear(self.fusion_dim, 1)

    def _contextualize(self, sequences):
        all_sequences = torch.cat(sequences, dim=1)
        result = []
        for seq, norm, ffn in zip(sequences, self.context_norm, self.context_ffn):
            attended, _ = self.cross_attention(seq, all_sequences, all_sequences, need_weights=False)
            context = norm(seq + attended)
            result.append(norm(context + ffn(context)).mean(dim=1))
        return result

    def forward(self, text, audio, video):
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a, x_v = audio.transpose(1, 2), video.transpose(1, 2)
        proj_l, proj_a, proj_v = self.proj_l(x_l), self.proj_a(x_a), self.proj_v(x_v)
        private = [self.enc_private_l(proj_l), self.enc_private_a(proj_a), self.enc_private_v(proj_v)]
        shared = [self.enc_shared(x) for x in (proj_l, proj_a, proj_v)]
        recon = [decoder(torch.cat([s, p], dim=1)) for decoder, s, p in zip((self.decoder_l, self.decoder_a, self.decoder_v), shared, private)]

        vectors = self._contextualize([s.transpose(1, 2) for s in shared])
        reg_logits = [head(v) for head, v in zip(self.reg_heads, vectors)]
        cls_logits = [head(v) for head, v in zip(self.cls_heads, vectors)]
        features = [torch.sigmoid(projector(v)) for projector, v in zip(self.projectors, vectors)]
        weights, energies, semantic_scores = self.modality_gate(*features, *cls_logits)
        w_l, w_a, w_v = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        fused = torch.cat([features[0] * w_l, features[1] * w_a, features[2] * w_v], dim=1)
        refined, retrieval_gate, retrieval_attention = self.rcr_module(fused, private)
        output = self.out_layer(self.proj2(F.dropout(F.relu(self.proj1(refined)), p=self.output_dropout, training=self.training)) + refined)
        return {
            'output_logit': output, 'reg_logits': reg_logits, 'cls_logits': cls_logits,
            'contrastive_feat': F.normalize(self.contrastive_head(refined), dim=1), 'gate_weights': weights,
            'retrieval_gate': retrieval_gate, 'retrieval_attention': retrieval_attention,
            'gate_info': {'semantic_scores': semantic_scores, 'energies': energies},
            'decomp_info': {'original': [proj_l, proj_a, proj_v], 'recon': recon, 'shared': shared, 'private': private}
        }
