import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from mamba_ssm import Mamba
from recbole.model.abstract_recommender import SequentialRecommender

__all__ = ["MAPLE"]


class MAPLE(SequentialRecommender):
    def __init__(self, config, dataset):
        super(MAPLE, self).__init__(config, dataset)

        self.d_model = config['hidden_size']
        self.d_state = config['mamba_d_state']
        self.d_conv = config['d_conv']
        self.expand = config['expand']
        self.num_layers = config["num_layers"]
        self.K_L = config['num_proto_long']
        self.K_S = config['num_proto_short']
        self.tau_proxy = config['tau_proxy']
        self.tau_nce = config['tau_nce']
        self.lambda_align = config['lambda_align']
        self.short_seq_len_ratio = config['short_seq_len_ratio']
        self.injection_weight = config['injection_weight']

        self.ema_alpha = config['ema_alpha']

        self.item_embedding = nn.Embedding(self.n_items, self.d_model, padding_idx=0)
        self.LayerNorm = nn.LayerNorm(self.d_model, eps=1e-12)
        self.dropout = nn.Dropout(config['dropout_prob'])
        self.ffn_dropout_prob = config['ffn_dropout_prob']

        self.loss_type = config['loss_type']

        self.ablation_mode = config['ablation_mode']

        self.h_L_norm = nn.LayerNorm(self.d_model, eps=1e-12, elementwise_affine=False)
        self.h_S_norm = nn.LayerNorm(self.d_model, eps=1e-12, elementwise_affine=True)

        self.long_encoders = nn.ModuleList([
            MambaLayer(
                d_model=self.d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                num_layers=self.num_layers,
                ffn_dropout=self.ffn_dropout_prob
            ) for _ in range(self.num_layers)
        ])
        self.short_encoder = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=False,
            batch_first=True
        )

        P_L_init = torch.empty(self.K_L, self.d_model)
        nn.init.normal_(P_L_init, mean=0.0, std=1.0)
        P_L_init = F.normalize(P_L_init, dim=-1) * (self.d_model ** 0.5)
        self.register_buffer('P_L', P_L_init)

        P_S_init = torch.empty(self.K_S, self.d_model)
        nn.init.normal_(P_S_init, mean=0.0, std=1.0)
        P_S_init = F.normalize(P_S_init, dim=-1) * (self.d_model ** 0.5)
        self.register_buffer('P_S', P_S_init)

        if self.loss_type == "CE":
            self.fusion = DynamicGatedFusion(self.d_model, self.ablation_mode)
            self.rec_loss = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("'loss_type' must be 'CE'!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)

    def _get_soft_proxy(self, h, P, tau):
        h_norm = F.normalize(h, dim=-1)
        P_norm = F.normalize(P, dim=-1)
        weights = F.softmax(torch.matmul(h_norm, P_norm.T) / tau, dim=-1)
        return torch.matmul(weights, P)

    def _enhance_and_fuse(self, h_L, h_S, detach_proxy=False):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not detach_proxy):
            proxy_L = self._get_soft_proxy(h_L, self.P_L, tau=self.tau_proxy)
            proxy_S = self._get_soft_proxy(h_S, self.P_S, tau=self.tau_proxy)

        h_L_enhanced = h_L + self.injection_weight * proxy_L
        h_S_enhanced = h_S + self.injection_weight * proxy_S
        return self.fusion(h_L_enhanced, h_S_enhanced)

    def forward(self, item_seq, item_seq_len):
        emb_seq = self.item_embedding(item_seq)

        long_out = self.LayerNorm(self.dropout(emb_seq))
        for i in range(self.num_layers):
            long_out = self.long_encoders[i](long_out)

        h_L = self.h_L_norm(self.gather_indexes(long_out, item_seq_len - 1))

        B, L = item_seq.shape
        seq_idx = torch.arange(L, device=item_seq.device).unsqueeze(0).expand(B, L)
        short_cutoff = (item_seq_len.float() * (1 - self.short_seq_len_ratio)).long()
        short_mask = (seq_idx >= short_cutoff.unsqueeze(1)) & (seq_idx < item_seq_len.unsqueeze(1))
        short_emb = self.LayerNorm(self.dropout(emb_seq * short_mask.unsqueeze(-1).float()))
        short_out, _ = self.short_encoder(short_emb)
        h_S = self.gather_indexes(short_out, item_seq_len - 1)
        h_S = self.h_S_norm(h_S)

        return h_L, h_S

    def _update_prototypes_ema(self, h, prototypes, alpha):
        with torch.no_grad():
            p_norm = F.normalize(prototypes, dim=-1)
            sim = torch.matmul(prototypes, p_norm.T)
            assign_idx = torch.argmax(sim, dim=-1)
            K = prototypes.shape[0]
            mask = F.one_hot(assign_idx, num_classes=K).float()
            counts = mask.sum(dim=0)

            valid_mask = counts > 0

            h_sum = torch.matmul(mask.T, h)

            h_mean = torch.zeros_like(prototypes)
            h_mean[valid_mask] = h_sum[valid_mask] / counts[valid_mask].unsqueeze(-1)

            h_mean_norm = F.normalize(h_mean, dim=-1) * (self.d_model ** 0.5)

            prototypes[valid_mask] = alpha * prototypes[valid_mask] + (1.0 - alpha) * h_mean_norm[valid_mask]

            prototypes.data = F.normalize(prototypes.data, dim=-1) * (self.d_model ** 0.5)

    def cross_intent_contrastive_loss(self, h_L, h_S, P_L, P_S, tau_proxy, tau_nce):

        def _compute_binary_infonce(h_norm, pos_proxy, neg_proxy, tau):
            pos_score = torch.sum(h_norm * pos_proxy, dim=-1, keepdim=True) / tau
            neg_score = torch.sum(h_norm * neg_proxy, dim=-1, keepdim=True) / tau

            logits = torch.cat([pos_score, neg_score], dim=-1)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=h_norm.device)

            return F.cross_entropy(logits, labels)

        h_L_norm = F.normalize(h_L, dim=-1)
        h_S_norm = F.normalize(h_S, dim=-1)
        P_L_norm = F.normalize(P_L, dim=-1)
        P_S_norm = F.normalize(P_S, dim=-1)

        with torch.no_grad():
            weight_L = F.softmax(torch.matmul(h_L_norm, P_L_norm.T) / tau_proxy, dim=-1)
            proxy_L = F.normalize(torch.matmul(weight_L, P_L_norm), dim=-1)

            weight_S = F.softmax(torch.matmul(h_S_norm, P_S_norm.T) / tau_proxy, dim=-1)
            proxy_S = F.normalize(torch.matmul(weight_S, P_S_norm), dim=-1)

        loss_cl_L = _compute_binary_infonce(h_L_norm, proxy_L, proxy_S, tau_nce)
        loss_cl_S = _compute_binary_infonce(h_S_norm, proxy_S, proxy_L, tau_nce)

        return loss_cl_L + loss_cl_S

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        h_L, h_S = self.forward(item_seq, item_seq_len)

        h_fused = self._enhance_and_fuse(h_L, h_S, detach_proxy=True)
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(h_fused, test_item_emb.T)
        loss_rec = self.rec_loss(logits, pos_items)

        loss_cl = self.cross_intent_contrastive_loss(h_L, h_S, self.P_L, self.P_S, tau_proxy=self.tau_proxy,
                                                    tau_nce=self.tau_nce)

        if self.training:
            self._update_prototypes_ema(h_L, self.P_L, alpha=self.ema_alpha)
            self._update_prototypes_ema(h_S, self.P_S, alpha=self.ema_alpha)

        total_loss = loss_rec + self.lambda_align * loss_cl
        if wandb.run is not None:
            wandb.log({
                "Train/loss_total": total_loss.item(),
                "Train/loss_rec": loss_rec.item(),
                "Train/loss_cl": loss_cl.item(),
            })

        return total_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        h_L, h_S = self.forward(item_seq, item_seq_len)

        h_final = self._enhance_and_fuse(h_L, h_S)

        test_emb = self.item_embedding(test_item)
        score = torch.sum(h_final * test_emb, dim=-1)

        return score

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        h_L, h_S = self.forward(item_seq, item_seq_len)

        h_fused = self._enhance_and_fuse(h_L, h_S)
        scores = torch.matmul(h_fused, self.item_embedding.weight.T)
        return scores


class MambaLayer(nn.Module):

    def __init__(self, d_model, d_state, d_conv, expand, num_layers, ffn_dropout):
        super(MambaLayer, self).__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.num_layers = num_layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, inner_size=4 * d_model, dropout=ffn_dropout)

    def forward(self, x):
        mamba_out = self.mamba(self.norm1(x))
        if self.num_layers == 1:
            h = mamba_out
        else:
            h = x + mamba_out
        out = h + self.ffn(self.norm2(h))
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class DynamicGatedFusion(nn.Module):

    def __init__(self, d_model, ablation_mode):
        super(DynamicGatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.ablation_mode = ablation_mode

    def forward(self, h_L, h_S):
        concat_feat = torch.cat([h_L, h_S], dim=-1)
        gate_weight = self.gate(concat_feat)
        h_final = h_L + gate_weight * h_S
        return h_final
