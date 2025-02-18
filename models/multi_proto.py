import torch
import torch.nn as nn
import torch.nn.functional as F

class InsectAwareProtoPool(nn.Module):
    """
    支持虫态感知的多粒度原型池
    管理每个类别内不同虫态的原型，并支持共享虫态原型
    """
    def __init__(self, dim, config):
        super().__init__()
        self.config = config
        self.proto_dim = config.proto_dim  # 原型投影维度（如果需要变换）

        # 类内多虫态原型：字典格式 {class_id: {stage: ParameterList of prototypes}}
        self.class_protos = nn.ModuleDict()
        # 共享虫态原型：字典格式 {stage: Parameter}，直接存储一个矩阵，每一行代表一个共享原型
        self.shared_stage_protos = nn.ParameterDict()

        # 动态路由网络，用于多头路由匹配
        self.router = nn.Sequential(
            nn.Linear(config.embed_dim, config.router_hidden),
            nn.GELU(),
            nn.Linear(config.router_hidden, config.proto_heads)
        )

        # 初始化共享虫态原型，每个阶段配置 config.shared_proto_per_stage 个原型
        for stage in range(config.max_stages):
            self.shared_stage_protos[str(stage)] = nn.Parameter(
                torch.randn(config.shared_proto_per_stage, config.embed_dim)
            )

    def init_class(self, class_id):
        """
        为新类别初始化虫态专属原型池，
        格式：{stage: ParameterList()}，每个阶段初始为空
        """
        self.class_protos[str(class_id)] = nn.ModuleDict({
            str(stage): nn.ParameterList() for stage in range(self.config.max_stages)
        })

    def forward(self, features, class_ids, stages):
        """
        输入:
          features: [B, D]（经过 Backbone 得到的特征）
          class_ids: [B]，每个样本对应的类别 ID（整型张量）
          stages: [B]，每个样本对应的虫态阶段（整型张量）
        返回:
          增强后的特征：[B, D] = 原始特征 + 0.5 * (类内原型均值 + 共享原型均值)
        """
        enhanced = []
        for feat, cid, stage in zip(features, class_ids, stages):
            # 获取该类别该阶段的原型（若无则默认为零向量）
            class_proto = self._get_class_proto(cid, stage)
            # 获取共享虫态原型均值
            shared_proto = self._get_shared_stage_proto(stage)
            # 原始特征加上两部分增强
            enhanced_feat = feat + 0.5 * (class_proto + shared_proto)
            enhanced.append(enhanced_feat)
        return torch.stack(enhanced)

    def update_protos(self, features, class_ids, stages):
        """
        双路径原型更新：对每个样本同时更新类内原型和共享虫态原型
        """
        with torch.no_grad():
            for feat, cid, stage in zip(features, class_ids, stages):
                self._update_class_proto(feat, cid, stage, momentum=self.config.proto_momentum)
                self._update_shared_proto(feat, stage, momentum=self.config.shared_momentum)

    def _get_class_proto(self, cid, stage):
        """
        返回类别 cid 在阶段 stage 下的原型均值。
        如果当前类别或阶段尚未初始化，则返回零向量
        """
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
        """
        返回共享虫态原型均值
        """
        stage_str = str(int(stage.item()))
        protos = self.shared_stage_protos[stage_str]
        return torch.mean(protos, dim=0)

    def _update_class_proto(self, feat, cid, stage, momentum=0.9):
        """
        更新类别 cid 下阶段 stage 的原型
        使用余弦相似度匹配，若原型数不足则直接添加，新样本与原型更新时采用动量更新
        注意：更新时使用 .data.copy_() 保持梯度流不干扰
        """
        cid_str = str(int(cid.item()))
        stage_str = str(int(stage.item()))
        # 若该类别尚未初始化，先初始化
        if cid_str not in self.class_protos:
            self.init_class(cid_str)
        protos = self.class_protos[cid_str][stage_str]
        if len(protos) < self.config.max_proto_per_class:
            protos.append(nn.Parameter(feat.clone()))
        else:
            # 计算每个已有原型与 feat 的余弦相似度
            stacked = torch.stack(list(protos))
            sims = F.cosine_similarity(feat.unsqueeze(0), stacked)
            # 此处选择余弦相似度最低的原型进行更新
            idx = torch.argmin(sims).item()
            # 动量更新：自适应更新系数可以考虑加入距离量化
            new_proto = momentum * protos[idx].data + (1 - momentum) * feat
            protos[idx].data.copy_(new_proto)

    def _update_shared_proto(self, feat, stage, momentum=0.8):
        """
        更新共享虫态原型：选择与 feat 余弦相似度最高的原型进行更新
        """
        stage_str = str(int(stage.item()))
        protos = self.shared_stage_protos[stage_str]
        sims = F.cosine_similarity(feat.unsqueeze(0), protos)
        idx = torch.argmax(sims).item()
        new_proto = momentum * protos[idx].data + (1 - momentum) * feat
        protos[idx].data.copy_(new_proto)

# 阶段转移一致性损失模块
class StageAwareLoss(nn.Module):
    def __init__(self, max_stages):
        super().__init__()
        # 使用可学习的转移矩阵作为先验
        self.transition = nn.Parameter(
            torch.eye(max_stages) + torch.randn(max_stages, max_stages) * 0.1
        )
    
    def forward(self, pred_stages, true_stages):
        """
        pred_stages: [B, max_stages]，模型预测的虫态分布
        true_stages: [B]，真实的虫态标签（整数）
        计算每个样本的 KL 散度损失
        """
        # 获取对应的真实转移先验 (例如：取出对应行)
        valid_trans = self.transition[true_stages]  # [B, max_stages]
        valid_trans = F.softmax(valid_trans, dim=-1)
        return F.kl_div(F.log_softmax(pred_stages, dim=-1), valid_trans, reduction='batchmean')
