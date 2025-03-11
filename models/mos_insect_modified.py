import logging
import math
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from utils.inc_net import MOSNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, target2onehot
from models.mix import CrossModalFusion, StatePredictor


##########################
# Global Fusion Module
##########################
class GlobalFusion(nn.Module):
    """
    Fuses global pooled feature and raw (time-step aggregated) feature.
    Input: two features [B, D] each; output: [B, D]
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.LayerNorm(feat_dim)
        )
    def forward(self, global_feat, raw_feat):
        fused = torch.cat([global_feat, raw_feat], dim=1)
        return self.fc(fused)


##########################
# DynamicTemporalModel
##########################
class DynamicTemporalModel(nn.Module):
    def __init__(self, feat_dim=512, max_stages=5, num_classes=100, mem_size=10):
        super().__init__()
        self.feat_dim = feat_dim
        self.max_stages = max_stages
        self.num_classes = num_classes
        self.mem_size = mem_size

        # 使用 register_buffer 存储记忆库（形状: [num_classes, max_stages, mem_size, feat_dim]）
        self.register_buffer('memory_features', torch.zeros(num_classes, max_stages, mem_size, feat_dim))
        self.register_buffer('memory_counts', torch.zeros(num_classes, max_stages, dtype=torch.long))
        
        # 增强的关联注意力机制与LayerNorm
        self.stage_relation = nn.MultiheadAttention(feat_dim, 8, batch_first=True)
        self.stage_norm = nn.LayerNorm(feat_dim)
        
        # 改进的阶段编码器，输入为2维（相对位置、当前阶段）
        self.stage_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, feat_dim),
            nn.LayerNorm(feat_dim)
        )
        
        # 增强的时序聚合器（使用2层LSTM）
        self.aggregator = nn.LSTM(feat_dim, feat_dim, num_layers=2, batch_first=True, dropout=0.1)
        
        # 双头预测阶段（预测虫态）
        self.stage_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LayerNorm(feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, max_stages)
        )

    def forward(self, x, current_stage, class_ids, return_attn=False):
        """
        Args:
            x: [B, T, D] or [B, D]
            current_stage: [B] or [B, T]
            class_ids: [B]
        Returns:
            Dictionary with:
              'temporal_feat': fused global feature [B, D]
              'raw_temporal_feat': raw time-step features [B, T, D]
              'stage_logits': predicted stage logits [B, max_stages]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, D]
            current_stage = current_stage.unsqueeze(1)  # [B, 1]
        B, T, D = x.shape

        # 构建批量化的位置编码输入：使用时间步位置和当前阶段
        position_ids = torch.arange(T, device=x.device).expand(B, T)  # [B, T]
        pos_stage_input = torch.stack([position_ids.float(), current_stage.float()], dim=-1)  # [B, T, 2]
        pos_emb = self.stage_encoder(pos_stage_input.view(-1, 2)).view(B, T, D)  # [B, T, D]
        
        # 加入位置编码
        x = x + pos_emb
        attn_out, attn_weights = self.stage_relation(x, x, x, need_weights=return_attn)
        x = self.stage_norm(x + attn_out)  # [B, T, D]
        
        # 未聚合的时序特征
        raw_temporal_feat = x  # [B, T, D]
        
        # 时序聚合：使用LSTM聚合
        _, (h_n, _) = self.aggregator(x)
        pooled_feat = h_n[-1]  # [B, D]

        # 融合全局池化特征和原始时序信息
        raw_mean = torch.mean(raw_temporal_feat, dim=1)  # [B, D]
        fused_global = GlobalFusion(D)(pooled_feat, raw_mean)  # [B, D]

        # 阶段预测
        stage_logits = self.stage_head(fused_global)  # [B, max_stages]

        # 更新记忆库（依然采用循环更新，可考虑后续向量化）
        with torch.no_grad():
            self._update_memory(raw_temporal_feat, class_ids, current_stage)
        
        outputs = {
            'temporal_feat': fused_global,           # 全局融合特征用于后续分类与状态预测
            'raw_temporal_feat': raw_temporal_feat,    # 用于原型池更新
            'stage_logits': stage_logits
        }
        if return_attn:
            outputs['attn_weights'] = attn_weights
        return outputs

    @torch.no_grad()
    def _update_memory(self, features, class_ids, stage_ids):
        """
        更新记忆库。此处仍采用循环遍历（未来可用scatter等操作向量化）。
        Args:
            features: [B, T, D]
            class_ids: [B]
            stage_ids: [B, T]
        """
        B, T, D = features.shape
        for b in range(B):
            cid = class_ids[b].item()
            for t in range(T):
                sid = stage_ids[b, t].item()
                cur_count = self.memory_counts[cid, sid].item()
                if cur_count < self.mem_size:
                    self.memory_features[cid, sid, cur_count] = features[b, t]
                    self.memory_counts[cid, sid] += 1
                else:
                    # FIFO更新：滚动后更新最后一个位置
                    self.memory_features[cid, sid] = torch.roll(self.memory_features[cid, sid], -1, dims=0)
                    self.memory_features[cid, sid, -1] = features[b, t]

    def get_stage_memory(self, class_id, stage_id):
        count = self.memory_counts[class_id, stage_id].item()
        if count == 0:
            return None
        return self.memory_features[class_id, stage_id, :count]


##########################
# InsectAwareProtoPool
##########################
class InsectAwareProtoPool(nn.Module):
    """
    支持虫态感知的多粒度原型池
    管理每个类别内不同虫态的原型，并支持共享虫态原型
    """
    def __init__(self, dim, config):
        super().__init__()
        self.config = config
        self.proto_dim = config.proto_dim

        # 类内多虫态原型：格式 {class_id: {stage: ParameterList}}
        self.class_protos = nn.ModuleDict()
        # 共享虫态原型：格式 {stage: Parameter} (每行一个共享原型)
        self.shared_stage_protos = nn.ParameterDict()
        # 动态路由网络（用于多头匹配，暂不做额外修改）
        self.router = nn.Sequential(
            nn.Linear(config.embed_dim, config.router_hidden),
            nn.GELU(),
            nn.Linear(config.router_hidden, config.proto_heads)
        )
        for stage in range(config.max_stages):
            self.shared_stage_protos[str(stage)] = nn.Parameter(
                torch.randn(config.shared_proto_per_stage, config.embed_dim)
            )
        self.consistency_weight = nn.Parameter(torch.tensor(0.1))

    def init_class(self, class_id):
        self.class_protos[str(class_id)] = nn.ModuleDict({
            str(stage): nn.ParameterList() for stage in range(self.config.max_stages)
        })

    def has_prototype(self, class_id, stage):
        cid_str = str(int(class_id))
        stage_str = str(int(stage))
        return cid_str in self.class_protos and stage_str in self.class_protos[cid_str]

    def init_prototype(self, class_id, stage, init_vector):
        cid_str = str(int(class_id))
        stage_str = str(int(stage))
        if cid_str not in self.class_protos:
            self.init_class(cid_str)
        self.class_protos[cid_str][stage_str].append(nn.Parameter(init_vector.clone()))

    def update_prototype(self, class_id, stage, new_vector):
        cid_str = str(int(class_id))
        stage_str = str(int(stage))
        if not self.has_prototype(class_id, stage):
            self.init_prototype(class_id, stage, new_vector)
        else:
            protos = self.class_protos[cid_str][stage_str]
            if len(protos) < self.config.max_proto_per_class:
                protos.append(nn.Parameter(new_vector.clone()))
            else:
                stacked = torch.stack(list(protos))
                sims = F.cosine_similarity(new_vector.unsqueeze(0), stacked)
                idx = torch.argmin(sims).item()
                protos[idx].data.copy_(new_vector)

    def calculate_consistency(self):
        loss = 0
        for cid in self.class_protos.keys():
            stages = sorted(self.class_protos[cid].keys(), key=lambda x: int(x))
            for i in range(len(stages) - 1):
                curr_proto = torch.stack(list(self.class_protos[cid][stages[i]]))
                next_proto = torch.stack(list(self.class_protos[cid][stages[i+1]]))
                loss += F.mse_loss(curr_proto.mean(dim=0), next_proto.mean(dim=0))
        return self.consistency_weight * (loss / len(self.class_protos)) if self.class_protos else 0

    def forward(self, features, class_ids, stages):
        """
        Args:
            features: [B, D]
            class_ids: [B]
            stages: [B]
        Returns:
            Enhanced features: [B, D] = original + 0.5*(class proto mean + shared proto mean)
        """
        enhanced = []
        for feat, cid, stage in zip(features, class_ids, stages):
            class_proto = self._get_class_proto(cid, stage)
            shared_proto = self._get_shared_stage_proto(stage)
            enhanced_feat = feat + 0.5 * (class_proto + shared_proto)
            enhanced.append(enhanced_feat)
        return torch.stack(enhanced)

    def _get_class_proto(self, cid, stage):
        cid_str = str(int(cid.item()))
        stage_str = str(int(stage.item()))
        if cid_str not in self.class_protos:
            return torch.zeros(self.config.embed_dim, device=next(self.parameters()).device)
        protos = self.class_protos[cid_str][stage_str]
        if len(protos) == 0:
            return torch.zeros(self.config.embed_dim, device=next(self.parameters()).device)
        stacked = torch.stack(list(protos))
        return torch.mean(stacked, dim=0)

    def _get_shared_stage_proto(self, stage):
        stage_str = str(int(stage.item()))
        protos = self.shared_stage_protos[stage_str]
        return torch.mean(protos, dim=0)

    def update_protos(self, features, class_ids, stages):
        with torch.no_grad():
            for feat, cid, stage in zip(features, class_ids, stages):
                self._update_class_proto(feat, cid, stage, momentum=self.config.proto_momentum)
                self._update_shared_proto(feat, stage, momentum=self.config.shared_momentum)

    def _update_class_proto(self, feat, cid, stage, momentum=0.9):
        cid_str = str(int(cid.item()))
        stage_str = str(int(stage.item()))
        if cid_str not in self.class_protos:
            self.init_class(cid_str)
        protos = self.class_protos[cid_str][stage_str]
        if len(protos) < self.config.max_proto_per_class:
            protos.append(nn.Parameter(feat.clone()))
        else:
            stacked = torch.stack(list(protos))
            sims = F.cosine_similarity(feat.unsqueeze(0), stacked)
            idx = torch.argmin(sims).item()
            new_proto = momentum * protos[idx].data + (1 - momentum) * feat
            protos[idx].data.copy_(new_proto)

    def _update_shared_proto(self, feat, stage, momentum=0.8):
        stage_str = str(int(stage.item()))
        protos = self.shared_stage_protos[stage_str]
        sims = F.cosine_similarity(feat.unsqueeze(0), protos)
        idx = torch.argmax(sims).item()
        new_proto = momentum * protos[idx].data + (1 - momentum) * feat
        protos[idx].data.copy_(new_proto)


##########################
# StageAwareLoss
##########################
class StageAwareLoss(nn.Module):
    def __init__(self, max_stages):
        super().__init__()
        self.transition = nn.Parameter(
            torch.eye(max_stages) + torch.randn(max_stages, max_stages) * 0.1
        )
    
    def forward(self, pred_stages, true_stages):
        valid_trans = self.transition[true_stages]  # [B, max_stages]
        valid_trans = F.softmax(valid_trans, dim=-1)
        return F.kl_div(F.log_softmax(pred_stages, dim=-1), valid_trans, reduction='batchmean')


##########################
# EnhancedPositionEncoder
##########################
class EnhancedPositionEncoder(nn.Module):
    def __init__(self, max_stages, feat_dim):
        super().__init__()
        self.max_stages = max_stages
        self.feat_dim = feat_dim
        pe = self._create_sinusoidal(max_stages, feat_dim)
        self.register_buffer('base_pe', pe)
        self.residual = nn.Embedding(max_stages, feat_dim)
        nn.init.normal_(self.residual.weight, std=0.02)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def _create_sinusoidal(self, max_stages, feat_dim):
        position = torch.arange(0, max_stages).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feat_dim, 2) * (-math.log(10000.0)/feat_dim))
        pe = torch.zeros(max_stages, feat_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def expand_max_stages(self, new_max):
        old_pe = self.base_pe.unsqueeze(0).unsqueeze(0)
        new_pe = F.interpolate(old_pe, size=(new_max, self.feat_dim), mode='linear', align_corners=False)
        self.base_pe = new_pe.squeeze()
        self.max_stages = new_max

    def forward(self, stage_labels):
        base = F.embedding(stage_labels, self.base_pe)  # [B, T, D]
        residual = self.residual(stage_labels)
        return base * self.scale + residual


##########################
# DynamicCausalAttention
##########################
class DynamicCausalAttention(nn.Module):
    def __init__(self, feat_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.qkv = nn.Linear(feat_dim, 3 * feat_dim)
        self.proj = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale_factor = nn.Parameter(torch.tensor(1.0 / math.sqrt(self.head_dim)))

    def _create_masks(self, padding_mask, seq_len):
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            padding_mask = padding_mask.expand(-1, self.num_heads, seq_len, -1)
            causal_mask = causal_mask | padding_mask
        return causal_mask.to(self.qkv.weight.device)

    def forward(self, x, padding_mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale_factor
        attn = attn.masked_fill(self._create_masks(padding_mask, T), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(x)


##########################
# SafeCoreManager
##########################
class SafeCoreManager(nn.Module):
    def __init__(self, num_classes, num_stages, feat_dim, momentum=0.99):
        super().__init__()
        self.momentum = momentum
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.register_buffer('prototypes', torch.randn(num_classes, num_stages, feat_dim) * 0.01)
        self.register_buffer('counts', torch.zeros(num_classes, num_stages))

    def expand_classes(self, new_num_classes):
        if new_num_classes <= self.num_classes:
            return
        new_protos = torch.randn(new_num_classes, self.num_stages, self.prototypes.size(-1)) * 0.01
        new_protos[:self.num_classes] = self.prototypes
        self.prototypes = new_protos
        self.num_classes = new_num_classes

    def update(self, features, class_ids, stage_ids):
        unique_c, unique_s = torch.unique(class_ids), torch.unique(stage_ids)
        for c in unique_c:
            for s in unique_s:
                mask = (class_ids == c) & (stage_ids == s)
                if not mask.any():
                    continue
                feat_subset = features[mask]
                current_proto = self.prototypes[c, s]
                new_proto = current_proto * self.momentum + (1 - self.momentum) * feat_subset.mean(dim=0)
                self.prototypes[c, s] = new_proto.detach()
                self.counts[c, s] += mask.sum().item()


##########################
# RobustStageVAE
##########################
class RobustStageVAE(nn.Module):
    def __init__(self, feat_dim, num_stages, latent_dim=128):
        super().__init__()
        self.num_stages = num_stages
        self.latent_dim = latent_dim
        self.stage_encoder = nn.Embedding(num_stages, latent_dim)
        nn.init.uniform_(self.stage_encoder.weight, -0.1, 0.1)
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim + latent_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + latent_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, feat_dim)
        )
        # Fix kl_weight at initialization
        self.kl_weight = nn.Parameter(torch.tensor(0.5))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, stage_labels):
        stage_emb = self.stage_encoder(stage_labels)
        x_cond = torch.cat([x, stage_emb], dim=1)
        params = self.encoder(x_cond)
        mu, logvar = params.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        z_cond = torch.cat([z, stage_emb], dim=1)
        recon = self.decoder(z_cond)
        return recon, mu, logvar

    def sample(self, stage_labels, num_samples=1):
        if num_samples > 1:
            stage_labels = stage_labels.repeat_interleave(num_samples)
        stage_emb = self.stage_encoder(stage_labels)
        z = torch.randn(len(stage_labels), self.latent_dim).to(stage_emb.device)
        z_cond = torch.cat([z, stage_emb], dim=1)
        return self.decoder(z_cond)


##########################
# TemporalAttentionPooler
##########################
class TemporalAttentionPooler(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(feat_dim))
        self.key = nn.Linear(feat_dim, feat_dim)
    def forward(self, x, padding_mask=None):
        attn = torch.einsum('btd,d->bt', x, self.query)
        attn = attn / math.sqrt(self.query.size(-1))
        if padding_mask is not None:
            attn = attn.masked_fill(padding_mask, float('-inf'))
        attn_weights = F.softmax(attn, dim=1)
        return torch.einsum('bt,btd->bd', attn_weights, x)


##########################
# InsectTemporalFramework
##########################
class InsectTemporalFramework(nn.Module):
    def __init__(self, num_stages=5, feat_dim=512, num_classes=20):
        super().__init__()
        self.num_stages = num_stages
        self.feat_dim = feat_dim
        
        # 时序位置编码模块
        self.temporal_pos_enc = EnhancedPositionEncoder(num_stages, feat_dim)
        # 视觉与位置融合模块
        self.fusion = CrossModalFusion(feat_dim)
        # 因果时序自注意力模块
        self.causal_attn = DynamicCausalAttention(feat_dim)
        # 状态预测模块
        self.state_predictor = StatePredictor(feat_dim, num_stages)
        # 核心向量管理器
        self.core_manager = SafeCoreManager(num_classes, num_stages, feat_dim)
        # VAE伪特征回放模块
        self.vae_generator = RobustStageVAE(feat_dim, num_stages)
        # 时序信息聚合
        self.pooler = TemporalAttentionPooler(feat_dim)
        # 新增融合模块，将全局池化特征与原始时序均值融合
        self.fusion_module = GlobalFusion(feat_dim)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, visual_feat, stage_labels, class_ids=None, padding_mask=None):
        """
        Args:
            visual_feat: [B, T, D]
            stage_labels: [B, T]
            class_ids: [B]
            padding_mask: [B, T]
        Returns:
            Dictionary with:
              'temporal_feat': fused global feature [B, D]
              'raw_temporal_feat': unpooled time-step features [B, T, D]
              'stage_logits': [B, num_stages]
              'vae_loss': scalar loss
        """
        B, T, D = visual_feat.shape
        # 位置编码注入
        pos_emb = self.temporal_pos_enc(stage_labels)  # [B, T, D]
        fused_feat = self.fusion(visual_feat, pos_emb)   # [B, T, D]
        # 因果时序建模
        temporal_feat = self.causal_attn(fused_feat, padding_mask)  # [B, T, D]
        # 保存原始未池化特征用于原型更新
        raw_temporal_feat = temporal_feat
        # 信息聚合：池化
        pooled_feat = self.pooler(temporal_feat, padding_mask)  # [B, D]
        # 计算原始时序均值
        raw_mean = torch.mean(temporal_feat, dim=1)  # [B, D]
        # 融合全局池化特征和原始均值
        fused_global = self.fusion_module(pooled_feat, raw_mean)  # [B, D]
        # 状态预测
        stage_logits = self.state_predictor(fused_global)
        # 核心向量管理更新
        if self.training and class_ids is not None:
            valid_mask = ~padding_mask if padding_mask is not None else torch.ones_like(stage_labels).bool()
            self.core_manager.update(
                temporal_feat[valid_mask],
                class_ids.repeat_interleave(T)[valid_mask.flatten()],
                stage_labels.flatten()[valid_mask.flatten()]
            )
        # VAE特征回放
        vae_loss = self.compute_vae_loss(temporal_feat, stage_labels, padding_mask)
        return {
            'temporal_feat': fused_global,
            'raw_temporal_feat': raw_temporal_feat,
            'stage_logits': stage_logits,
            'vae_loss': vae_loss
        }

    def compute_vae_loss(self, features, stage_labels, padding_mask):
        B, T, D = features.shape
        valid_mask = ~padding_mask if padding_mask is not None else torch.ones_like(stage_labels).bool()
        valid_feat = features[valid_mask]
        valid_stages = stage_labels[valid_mask]
        recon, mu, logvar = self.vae_generator(valid_feat, valid_stages)
        recon_loss = F.mse_loss(recon, valid_feat.detach())
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
        return recon_loss + self.vae_generator.kl_weight * kl_loss


##########################
# Learner
##########################
num_workers = 8
class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = MOSNet(args, True)
        self.cls_mean = dict()
        self.cls_cov = dict()
        self.cls2task = dict()
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.ca_lr = args["ca_lr"]
        self.crct_epochs = args["crct_epochs"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args.get("min_lr", None) is not None else 1e-8
        self.args = args
        self.ensemble = args["ensemble"]

        for n, p in self._network.backbone.named_parameters():
            if 'adapter' not in n and 'head' not in n:
                p.requires_grad = False

        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        # 模块初始化：利用 backbone 输出维度
        feat_dim = self._network.backbone.out_dim
        proto_config = type("Config", (), {})()
        proto_config.proto_dim = feat_dim
        proto_config.embed_dim = feat_dim
        proto_config.router_hidden = feat_dim // 2
        proto_config.proto_heads = 4
        proto_config.max_stages = args.get("max_stages", 3)
        proto_config.shared_proto_per_stage = 5
        proto_config.proto_momentum = 0.9
        proto_config.shared_momentum = 0.8
        proto_config.max_proto_per_class = 10

        self.proto_pool = InsectAwareProtoPool(dim=feat_dim, config=proto_config)
        self.stage_loss = StageAwareLoss(max_stages=proto_config.max_stages)

        self.temporal_framework = InsectTemporalFramework(
            num_stages=proto_config.max_stages,
            feat_dim=feat_dim,
            num_classes=args.get("num_classes", 100)
        )

        if total_params != total_trainable_params:
            for name, param in self._network.backbone.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))

    def replace_fc(self):
        model = self._network.to(self._device)
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader_for_protonet):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.forward_orig(data)['features']
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        class_list = np.unique(self.train_dataset.labels)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto
        return model

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        for i in range(self._known_classes, self._total_classes):
            self.cls2task[i] = self._cur_task
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.data_manager = data_manager
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test")
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.replace_fc()
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.backbone.to(self._device)
        optimizer = self.get_optimizer(self._network.backbone)
        scheduler = self.get_scheduler(optimizer)
        self._init_train(train_loader, test_loader, optimizer, scheduler)
        self._network.backbone.adapter_update()
        self._compute_mean(self._network.backbone)
        if self._cur_task > 0:
            self.classifer_align(self._network.backbone)

    def get_optimizer(self, model):
        base_params = [p for name, p in model.named_parameters() if 'adapter' in name and p.requires_grad]
        base_fc_params = [p for name, p in model.named_parameters() if 'adapter' not in name and p.requires_grad]
        base_params = {'params': base_params, 'lr': self.init_lr, 'weight_decay': self.weight_decay}
        base_fc_params = {'params': base_fc_params, 'lr': self.init_lr * 0.1, 'weight_decay': self.weight_decay}
        network_params = [base_params, base_fc_params]
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(network_params, momentum=0.9)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(network_params)
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(network_params)
        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None
        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets, stages) in enumerate(train_loader):  # stages added
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                stages = stages.to(self._device)
                if inputs.dim() == 5:  # [B, T, C, H, W] 时序模式
                    B, T = inputs.shape[:2]
                    spatial_feat = self._network.backbone(inputs.flatten(0, 1))["features"]
                    temporal_feat = spatial_feat.view(B, T, -1)
                    temporal_out = self.temporal_framework(
                        temporal_feat,
                        stage_labels=stages,
                        class_ids=targets
                    )
                    # Use raw temporal features for prototype enhancement
                    enhanced_feat = self.proto_pool(
                        temporal_out['raw_temporal_feat'].reshape(B * T, -1),
                        targets.repeat_interleave(T),
                        stages.flatten()
                    )
                    logits = self._network.fc(enhanced_feat)
                    ce_loss = F.cross_entropy(logits, targets.repeat_interleave(T))
                    stage_loss = self.stage_loss(
                        temporal_out['stage_logits'],
                        stages[:, -1]
                    )
                    vae_loss = temporal_out['vae_loss']
                    loss = 0.4 * ce_loss + 0.3 * stage_loss + 0.2 * vae_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, 1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum().item()
                total += targets.size(0)
            scheduler.step()
            train_acc = np.around(correct * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)
        logging.info(info)

    @torch.no_grad()
    def _compute_mean(self, model):
        model.eval()
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size * 3, shuffle=False, num_workers=4)
            vectors = []
            for _, _inputs, _targets in idx_loader:
                _vectors = model(_inputs.to(self._device), adapter_id=self._cur_task, train=True)["features"]
                vectors.append(_vectors)
            vectors = torch.cat(vectors, dim=0)
            if self.args["ca_storage_efficient_method"] == 'covariance':
                features_per_cls = vectors
                self.cls_mean[class_idx] = features_per_cls.mean(dim=0).to(self._device)
                self.cls_cov[class_idx] = torch.cov(features_per_cls.T) + (torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-4).to(self._device)
            elif self.args["ca_storage_efficient_method"] == 'variance':
                features_per_cls = vectors
                self.cls_mean[class_idx] = features_per_cls.mean(dim=0).to(self._device)
                self.cls_cov[class_idx] = torch.diag(torch.cov(features_per_cls.T) + (torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-4).to(self._device))
            elif self.args["ca_storage_efficient_method"] == 'multi-centroid':
                from sklearn.cluster import KMeans
                n_clusters = self.args["n_centroids"]
                features_per_cls = vectors.cpu().numpy()
                kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
                kmeans.fit(features_per_cls)
                cluster_lables = kmeans.labels_
                cluster_means = []
                cluster_vars = []
                for i in range(n_clusters):
                    cluster_data = features_per_cls[cluster_lables == i]
                    cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float64).to(self._device)
                    cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float64).to(self._device)
                    cluster_means.append(cluster_mean)
                    cluster_vars.append(cluster_var)
                self.cls_mean[class_idx] = cluster_means
                self.cls_cov[class_idx] = cluster_vars

    def classifer_align(self, model):
        model.train()
        run_epochs = self.crct_epochs
        param_list = [p for n, p in model.named_parameters() if p.requires_grad and 'adapter' not in n]
        network_params = [{'params': param_list, 'lr': self.ca_lr, 'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=self.ca_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)
        prog_bar = tqdm(range(run_epochs))
        for epoch in prog_bar:
            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.batch_size * 5
            if self.args["ca_storage_efficient_method"] in ['covariance', 'variance']:
                for class_idx in range(self._total_classes):
                    mean = self.cls_mean[class_idx].to(self._device)
                    cov = self.cls_cov[class_idx].to(self._device)
                    if self.args["ca_storage_efficient_method"] == 'variance':
                        cov = torch.diag(cov)
                    m = MultivariateNormal(mean.float(), cov.float())
                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                    sampled_data.append(sampled_data_single)
                    sampled_label.extend([class_idx] * num_sampled_pcls)
            elif self.args["ca_storage_efficient_method"] == 'multi-centroid':
                for class_idx in range(self._total_classes):
                    for cluster in range(len(self.cls_mean[class_idx])):
                        mean = self.cls_mean[class_idx][cluster]
                        var = self.cls_cov[class_idx][cluster]
                        if var.mean() == 0:
                            continue
                        m = MultivariateNormal(mean.float(), (torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)).float())
                        sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                        sampled_data.append(sampled_data_single)
                        sampled_label.extend([class_idx] * num_sampled_pcls)
            else:
                raise NotImplementedError
            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)
            if epoch == 0:
                print("sampled data shape: ", sampled_data.shape)
            inputs = sampled_data
            targets = sampled_label
            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]
            losses = 0.0
            correct, total = 0, 0
            for _iter in range(self._total_classes):
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                outputs = model(inp, fc_only=True)
                logits = outputs['logits'][:, :self._total_classes]
                loss = F.cross_entropy(logits, tgt)
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(tgt.expand_as(preds)).cpu().sum()
                total += len(tgt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss
            scheduler.step()
            ca_acc = np.round(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, CA_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.crct_epochs,
                losses / self._total_classes,
                ca_acc,
            )
            prog_bar.set_description(info)
        logging.info(info)

    def orth_loss(self, features, targets):
        if self.cls_mean:
            sample_mean = []
            for k, v in self.cls_mean.items():
                if isinstance(v, list):
                    sample_mean.extend(v)
                else:
                    sample_mean.append(v)
            sample_mean = torch.stack(sample_mean, dim=0).to(self._device, non_blocking=True)
            M = torch.cat([sample_mean, features], dim=0)
            sim = torch.matmul(M, M.t()) / 0.8
            loss = F.cross_entropy(sim, torch.arange(0, sim.shape[0]).long().to(self._device))
            return self.args["reg"] * loss
        else:
            sim = torch.matmul(features, features.t()) / 0.8
            loss = F.cross_entropy(sim, torch.arange(0, sim.shape[0]).long().to(self._device))
            return self.args["reg"] * loss

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        orig_y_pred = []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                orig_logits = self._network.forward_orig(inputs)["logits"][:, :self._total_classes]
                orig_preds = torch.max(orig_logits, dim=1)[1].cpu().numpy()
                orig_idx = torch.tensor([self.cls2task[v] for v in orig_preds], device=self._device)
                orig_y_pred.append(orig_preds)
                all_features = torch.zeros(len(inputs), self._cur_task + 1, self._network.backbone.out_dim, device=self._device)
                for t_id in range(self._cur_task + 1):
                    t_features = self._network.backbone(inputs, adapter_id=t_id, train=False)["features"]
                    all_features[:, t_id, :] = t_features
                final_logits = []
                MAX_ITER = 4
                for x_id in range(len(inputs)):
                    loop_num = 0
                    prev_adapter_idx = orig_idx[x_id]
                    while True:
                        loop_num += 1
                        cur_feature = all_features[x_id, prev_adapter_idx].unsqueeze(0)
                        cur_logits = self._network.backbone(cur_feature, fc_only=True)["logits"][:, :self._total_classes]
                        cur_pred = torch.max(cur_logits, dim=1)[1].cpu().numpy()
                        cur_adapter_idx = torch.tensor([self.cls2task[v] for v in cur_pred], device=self._device)[0]
                        if loop_num >= MAX_ITER or cur_adapter_idx == prev_adapter_idx:
                            break
                        else:
                            prev_adapter_idx = cur_adapter_idx
                    final_logits.append(cur_logits)
                final_logits = torch.cat(final_logits, dim=0).to(self._device)
                if self.ensemble:
                    final_logits = F.softmax(final_logits, dim=1)
                    orig_logits = F.softmax(orig_logits / (1 / (self._cur_task + 1)), dim=1)
                    outputs = final_logits + orig_logits
                else:
                    outputs = final_logits
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        orig_acc = (np.concatenate(orig_y_pred) == np.concatenate(y_true)).sum() * 100 / len(np.concatenate(y_true))
        logging.info("the accuracy of the original model:{}".format(np.around(orig_acc, 2)))
        return np.concatenate(y_pred), np.concatenate(y_true)
