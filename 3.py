import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Normal

class InsectTemporalFramework(nn.Module):
    def __init__(self, num_stages, feat_dim=512, num_classes=100):
        super().__init__()
        self.num_stages = num_stages
        self.feat_dim = feat_dim
        
        # 一、时序位置编码模块（改进插值策略）
        self.temporal_pos_enc = EnhancedPositionEncoder(num_stages, feat_dim)
        
        # 二、时序-视觉特征融合模块
        self.fusion = CrossModalFusion(feat_dim)
        
        # 三、因果时序自注意力模块（带填充处理）
        self.causal_attn = DynamicCausalAttention(feat_dim)
        
        # 四、时序状态预测与对比学习
        self.state_predictor = StatePredictor(feat_dim, num_stages)
        
        # 五、时序核心向量管理器（安全扩展）
        self.core_manager = SafeCoreManager(num_classes, num_stages, feat_dim)
        
        # VAE伪特征回放模块（条件注入改进）
        self.vae_generator = RobustStageVAE(feat_dim, num_stages)
        
        # 时序信息聚合
        self.pooler = TemporalAttentionPooler(feat_dim)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, visual_feat, stage_labels, class_ids=None, padding_mask=None):
        """
        visual_feat: [B, T, D] 时序视觉特征
        stage_labels: [B, T] 各时间步阶段标签
        padding_mask: [B, T] 填充位置掩码（True表示填充）
        """
        B, T, D = visual_feat.shape
        
        # 1. 位置编码注入
        pos_emb = self.temporal_pos_enc(stage_labels)  # [B, T, D]
        fused_feat = self.fusion(visual_feat, pos_emb)  # [B, T, D]
        
        # 2. 因果时序建模
        temporal_feat = self.causal_attn(fused_feat, padding_mask)  # [B, T, D]
        
        # 3. 信息聚合
        pooled_feat = self.pooler(temporal_feat, padding_mask)  # [B, D]
        
        # 4. 状态预测
        stage_logits = self.state_predictor(pooled_feat)
        
        # 5. 核心向量管理
        if self.training and class_ids is not None:
            valid_mask = ~padding_mask if padding_mask is not None else torch.ones_like(stage_labels).bool()
            self.core_manager.update(
                temporal_feat[valid_mask], 
                class_ids.repeat_interleave(T)[valid_mask.flatten()],
                stage_labels.flatten()[valid_mask.flatten()]
            )
        
        # 6. VAE特征回放
        vae_loss = self.compute_vae_loss(temporal_feat, stage_labels, padding_mask)
        
        return {
            'temporal_feat': pooled_feat,
            'stage_logits': stage_logits,
            'vae_loss': vae_loss
        }

    def compute_vae_loss(self, features, stage_labels, padding_mask):
        B, T, D = features.shape
        valid_mask = ~padding_mask if padding_mask is not None else torch.ones_like(stage_labels).bool()
        
        # 仅使用有效时间步
        valid_feat = features[valid_mask]
        valid_stages = stage_labels[valid_mask]
        
        recon, mu, logvar = self.vae_generator(valid_feat, valid_stages)
        recon_loss = F.mse_loss(recon, valid_feat.detach())
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B  # 按batch平均
        
        return recon_loss + 0.5 * kl_loss

# ===== 改进模块实现 =====

class EnhancedPositionEncoder(nn.Module):
    """结合正弦编码与可学习插值的位置编码"""
    def __init__(self, max_stages, feat_dim):
        super().__init__()
        self.max_stages = max_stages
        self.feat_dim = feat_dim
        
        # 正弦基编码
        pe = self._create_sinusoidal(max_stages, feat_dim)
        self.register_buffer('base_pe', pe)
        
        # 可学习残差
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
        # 一维时序插值
        old_pe = self.base_pe.unsqueeze(0).unsqueeze(0)  # [1, 1, L, D]
        new_pe = F.interpolate(old_pe, size=(new_max, self.feat_dim), 
                              mode='linear', align_corners=False)
        self.base_pe = new_pe.squeeze()
        self.max_stages = new_max

    def forward(self, stage_labels):
        base = F.embedding(stage_labels, self.base_pe)  # [B, T, D]
        residual = self.residual(stage_labels)          # [B, T, D]
        return base * self.scale + residual

class DynamicCausalAttention(nn.Module):
    """支持填充处理的动态因果注意力"""
    def __init__(self, feat_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        
        self.qkv = nn.Linear(feat_dim, 3*feat_dim)
        self.proj = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化缩放因子
        self.scale_factor = nn.Parameter(torch.tensor(1.0/math.sqrt(self.head_dim)))

    def _create_masks(self, padding_mask, seq_len):
        # 因果掩码
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        # 融合填充掩码
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            padding_mask = padding_mask.expand(-1, self.num_heads, seq_len, -1)
            causal_mask = causal_mask | padding_mask
            
        return causal_mask.to(self.qkv.weight.device)

    def forward(self, x, padding_mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, T, d]
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale_factor
        attn = attn.masked_fill(self._create_masks(padding_mask, T), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(x)

class SafeCoreManager(nn.Module):
    """安全的核心向量管理（带噪声初始化）"""
    def __init__(self, num_classes, num_stages, feat_dim, momentum=0.99):
        super().__init__()
        self.momentum = momentum
        self.num_classes = num_classes
        self.num_stages = num_stages
        
        # 带噪声的初始化
        self.register_buffer('prototypes', torch.randn(num_classes, num_stages, feat_dim) * 0.01)
        self.register_buffer('counts', torch.zeros(num_classes, num_stages))
        
    def expand_classes(self, new_num_classes):
        if new_num_classes <= self.num_classes:
            return
            
        new_protos = torch.randn(new_num_classes, self.num_stages, self.feat_dim) * 0.01
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
                
                # 动量更新
                new_proto = current_proto * self.momentum + \
                           (1 - self.momentum) * feat_subset.mean(dim=0)
                
                self.prototypes[c, s] = new_proto.detach()
                self.counts[c, s] += mask.sum().item()

class RobustStageVAE(nn.Module):
    """鲁棒的阶段感知VAE"""
    def __init__(self, feat_dim, num_stages, latent_dim=128):
        super().__init__()
        self.num_stages = num_stages
        self.latent_dim = latent_dim
        
        # 阶段条件编码
        self.stage_encoder = nn.Embedding(num_stages, latent_dim)
        nn.init.uniform_(self.stage_encoder.weight, -0.1, 0.1)
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim + latent_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, latent_dim*2)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + latent_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, feat_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

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

class TemporalAttentionPooler(nn.Module):
    """时序注意力池化"""
    def __init__(self, feat_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(feat_dim))
        self.key = nn.Linear(feat_dim, feat_dim)
        
    def forward(self, x, padding_mask=None):
        # x: [B, T, D]
        attn = torch.einsum('btd,d->bt', x, self.query)  # [B, T]
        attn = attn / math.sqrt(self.query.size(-1))
        
        if padding_mask is not None:
            attn = attn.masked_fill(padding_mask, float('-inf'))
            
        attn_weights = F.softmax(attn, dim=1)
        return torch.einsum('bt,btd->bd', attn_weights, x)

# ===== 辅助模块与训练框架 =====

class InsectTrainer:
    def __init__(self, model, num_classes, temp=0.1):
        self.model = model
        self.temp = temp
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
        self.class_centers = torch.zeros(num_classes, model.feat_dim)

    def compute_loss(self, outputs, labels, stage_labels):
        losses = {}
        
        # 阶段分类损失
        losses['stage'] = F.cross_entropy(outputs['stage_logits'], stage_labels)
        
        # 对比学习损失
        feat = outputs['temporal_feat']
        proto = self.model.core_manager.prototypes[labels, stage_labels]
        proto_feat = F.normalize(proto, dim=-1)
        logits = torch.einsum('bd,bd->b', feat, proto_feat) / self.temp
        losses['contrast'] = F.cross_entropy(logits.unsqueeze(0), torch.zeros(1).long().to(feat.device))
        
        # VAE损失
        losses['vae'] = outputs['vae_loss']
        
        # 总损失
        total_loss = 0.4*losses['stage'] + 0.4*losses['contrast'] + 0.2*losses['vae']
        return total_loss, losses

    def train_step(self, batch):
        images, labels, stages, padding_mask = batch
        self.optimizer.zero_grad()
        
        # 视觉特征提取
        with torch.no_grad():
            visual_feat = self.model.visual_encoder(images)  # 假设已有预训练编码器
            
        outputs = self.model(visual_feat, stages, labels, padding_mask)
        
        # 损失计算
        total_loss, loss_dict = self.compute_loss(outputs, labels, stages[:, -1])  # 取最后有效阶段
        
        # 反向传播
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss_dict
