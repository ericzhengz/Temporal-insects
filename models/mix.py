import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 时序-视觉特征融合模块 =====
class CrossModalFusion(nn.Module):
    """视觉特征与时序位置编码的深度融合模块"""
    def __init__(self, feat_dim):
        super().__init__()
        self.vis_transform = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
        ).to(device)  # 确保在 GPU 上
        self.pos_transform = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
        ).to(device)  # 确保在 GPU 上
        self.fusion_core = nn.Linear(feat_dim * 2, feat_dim).to(device)  # 确保在 GPU 上

    def forward(self, visual_feat, pos_emb):
        visual_feat = visual_feat.to(device)  # 确保在 GPU 上
        pos_emb = pos_emb.to(device)  # 确保在 GPU 上
        vis_proj = self.vis_transform(visual_feat)
        pos_proj = self.pos_transform(pos_emb)
        fused_feat = torch.cat([vis_proj, pos_proj], dim=-1)
        fused_feat = self.fusion_core(fused_feat)
        return fused_feat

# ===== 时序状态预测与对比学习模块 =====
class StatePredictor(nn.Module):
    """多任务预测模块：阶段分类 + 对比特征学习"""
    def __init__(self, feat_dim, num_stages, proj_dim=128):
        super().__init__()
        self.num_stages = num_stages
        
        # 阶段预测分支
        self.stage_predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//2),
            nn.LayerNorm(feat_dim//2),
            nn.GELU(),
            nn.Linear(feat_dim//2, num_stages)
        )
        
        # 对比学习投影头
        self.contrast_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
        # 温度系数
        self.tau = nn.Parameter(torch.tensor(0.1))
        
        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        # 阶段预测头初始化
        nn.init.xavier_normal_(self.stage_predictor[-1].weight)
        nn.init.constant_(self.stage_predictor[-1].bias, 0)
        
        # 对比头初始化
        for m in self.contrast_proj:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, temporal_feat):
        """
        输入:
            temporal_feat: [B, T, D] 时序特征
        返回:
            stage_logits: [B, T, num_stages] 各时间步阶段logits
            contrast_feat: [B*T, proj_dim] 对比学习特征
        """
        B, T, D = temporal_feat.shape
        
        # 阶段预测
        stage_logits = self.stage_predictor(temporal_feat)  # [B, T, S]
        
        # 对比特征投影
        contrast_feat = self.contrast_proj(temporal_feat.view(-1, D))  # [B*T, proj_dim]
        contrast_feat = F.normalize(contrast_feat, dim=-1)
        
        return stage_logits, contrast_feat

    def compute_contrast_loss(self, feat, labels, temperature=0.1):
        """
        对比损失计算
        输入:
            feat: [N, proj_dim] 特征向量(N=B*T)
            labels: [N] 阶段标签
        """
        # 相似度矩阵
        sim_matrix = torch.mm(feat, feat.T) / temperature  # [N, N]
        
        # 正样本掩码
        label_mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # [N, N]
        diag_mask = ~torch.eye(len(labels), dtype=torch.bool, device=feat.device)
        pos_mask = label_mask & diag_mask
        
        # 损失计算
        exp_sim = torch.exp(sim_matrix)
        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        neg_sum = (exp_sim * ~pos_mask).sum(dim=1)
        loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8)).mean()
        
        return loss