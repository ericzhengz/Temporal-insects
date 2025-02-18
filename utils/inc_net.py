import copy
import logging
import torch
from torch import nn
from backbone.linears import SimpleLinear, SplitCosineLinear, CosineLinear, EaseCosineLinear, SimpleContinualLinear
from backbone.prompt import CodaPrompt
import timm
import torch.nn.functional as F

def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    # SimpleCIL or SimpleCIL w/ Finetune
    if name == "pretrained_vit_b16_224" or name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif name == "pretrained_vit_b16_224_in21k" or name == "vit_base_patch16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif '_mos' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "mos":
            from backbone import vit_mos
            from easydict import EasyDict
            from models.multi_proto import InsectAwareProtoPool, StageAwareLoss
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                _device = args["device"][0],
                adapter_momentum = args["adapter_momentum"],
                # VPT related
                vpt_on=False,
                vpt_num=0,
                
                #new new new
                proto_on=True,
                proto_dim=768,
                max_proto_per_class=3,
                max_stages=5,
                shared_proto_per_stage=2,
                router_hidden=256,
                proto_heads=4,
                
                # 虫态配置
                stage_loss_weight=0.3,
                proto_momentum=0.9,
                shared_momentum=0.8,
                # 虫态损失权重
                stage_loss_weight = 0.3,
                # 其他配置...
                embed_dim = 768
            )
            if name == "vit_base_patch16_224_mos":
                model = vit_mos.vit_base_patch16_224_mos(num_classes=args["nb_classes"],
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
            elif name == "vit_base_patch16_224_in21k_mos":
                model = vit_mos.vit_base_patch16_224_in21k_mos(num_classes=args["nb_classes"],
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property  # 只读，便于访问
    def feature_dim(self): # 返回bcakbone的输出特征维度
        return self.backbone.out_dim

    def extract_vector(self, x):  # 从输入x提取特征向量
        if self.model_type == 'cnn':
            self.backbone(x)['features']  # cnn从backbone返回的字典中提取features
        else:
            return self.backbone(x)   # ViT直接返回

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            """
            {
                'fmaps': [x_1, x_2, ..., x_n],
                'features': features
                'logits': logits
            }
            """
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):  # 替换原有，适应新的类别数量
        fc = self.generate_fc(self.feature_dim, nb_classes)  # 生成新的全连接层
        if self.fc is not None:  # 保留旧类别的weight和bias，复制到新的全连接层
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):  # 权重对齐
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)  
            out = self.fc(x["features"])  # 生成分类结果
            out.update(x)  # 更新（feature和logit）
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations
        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.backbone.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.backbone.last_conv.register_forward_hook(
            forward_hook
        )



from collections import defaultdict
from torch.distributions.multivariate_normal import MultivariateNormal
# ========== 动态时序模块 ==========

class DynamicTemporalModel(nn.Module):
    def __init__(self, feat_dim=512, max_stages=5):
        super().__init__()
        self.feat_dim = feat_dim
        self.max_stages = max_stages
        
        # 动态记忆库：{class_id: {stage: [prototype]}}
        self.memory_bank = defaultdict(lambda: defaultdict(list))
        
        # 关联注意力机制
        self.stage_relation = nn.MultiheadAttention(feat_dim, 8, batch_first=True)
        
        # 阶段编码器（动态生成位置编码）
        self.stage_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, feat_dim)
        )
        
        # 时序聚合器：采用 LSTM 聚合序列信息
        self.aggregator = nn.LSTM(feat_dim, feat_dim, batch_first=True)
        
        # 阶段预测头：输出各阶段的概率分布
        self.stage_head = nn.Linear(feat_dim, max_stages)

    def forward(self, x, current_stage, class_ids):
        """
        输入：
            x: Backbone 提取后的特征 [B, D]
            current_stage: 当前虫态标签 [B]（每个样本的阶段标签）
            class_ids: 类别ID [B]
        输出：
            temporal_feat: 聚合后的时序特征 [B, D]
            stage_logits: 阶段预测 logits [B, max_stages]
        """
        B = x.size(0)
        seq_features = []
        for bid in range(B):
            cid = class_ids[bid].item()
            stage = current_stage[bid].item()
            
            # 更新记忆库：限制每个类别每个阶段最多保存 10 个原型
            if len(self.memory_bank[cid][stage]) < 10:
                self.memory_bank[cid][stage].append(x[bid].detach())
            else:
                self.memory_bank[cid][stage].pop(0)
                self.memory_bank[cid][stage].append(x[bid].detach())
            
            # 抽取当前类别的历史记录（排除当前阶段），构成一个历史序列
            all_history = []
            for s in self.memory_bank[cid]:
                if s != stage:
                    all_history.extend(self.memory_bank[cid][s])
            if all_history:
                # 若历史记录不为空，取最近 max_stages 个原型构成序列
                hist_seq = torch.stack(all_history[-self.max_stages:])
            else:
                # 否则用当前样本自身构成序列
                hist_seq = x[bid].unsqueeze(0)
            
            # 处理历史序列，生成动态时序特征
            seq_feature = self._process_sequence(hist_seq, current_stage[bid])
            seq_features.append(seq_feature)
        
        temporal_feat = torch.stack(seq_features)  # [B, D]
        stage_logits = self.stage_head(temporal_feat)  # [B, max_stages]
        return temporal_feat, stage_logits

    def _process_sequence(self, hist_seq, current_stage):
        """
        对历史序列 hist_seq 进行位置编码、注意力交互和 LSTM 聚合，返回单个样本的聚合特征。
        hist_seq: [T, D]
        current_stage: 标量，当前阶段标签
        """
        T = hist_seq.size(0)
        # 生成动态位置编码：计算每个位置与当前阶段的差值
        stage_diff = torch.arange(T, dtype=torch.float32, device=hist_seq.device).view(-1, 1) - current_stage.float()
        # stage_diff: [T, 1] → 经过 stage_encoder 得到 [T, D]
        pos_emb = self.stage_encoder(stage_diff)  # [T, D]
        
        # 将位置编码加入历史序列：得到增强后的序列 [T, D]
        seq_input = hist_seq + pos_emb
        
        # 利用多头注意力机制对序列进行交互建模
        # 注意：MultiheadAttention 要求输入维度为 [B, T, D]，这里 B=1
        attn_out, _ = self.stage_relation(seq_input.unsqueeze(0), seq_input.unsqueeze(0), seq_input.unsqueeze(0))
        # 利用 LSTM 聚合时序信息
        _, (h_n, _) = self.aggregator(attn_out)  # h_n: [1, 1, D]
        return h_n.squeeze(0).squeeze(0)  # 返回 [D]

# ========== 改进后的 MOSNet ==========

class MOSNet(nn.Module):
    def __init__(self, args, use_temporal=True):
        """
        参数:
          args: 包含必要的配置参数，如 num_classes、max_stages 等
          use_temporal: 是否启用时序建模（针对虫态和时序信息）
        """
        super(MOSNet, self).__init__()
        self.args = args
        self.use_temporal = use_temporal
        
        # Backbone 特征提取网络（例如 ResNet、ViT 等），需提供属性 out_dim
        # 此处仅示例，实际请替换为具体网络
        self.backbone = ...  # 例如：ResNet、EfficientNet 等
        self.out_dim = getattr(self.backbone, 'out_dim', 512)
        
        # 分类头：将特征映射到类别数
        self.fc = nn.Linear(self.out_dim, args.get("num_classes", 100))
        
        if self.use_temporal:
            # 动态时序建模模块：根据配置中 max_stages 参数构建
            self.dynamic_temporal_model = DynamicTemporalModel(
                feat_dim=self.out_dim, 
                max_stages=args.get("max_stages", 5)
            )
            # 虫态原型池：存储各类别各阶段的原型，用于特征增强
            self.prototype_pool = nn.ModuleDict()  # 格式：{class_id: nn.ModuleDict({stage: Parameter})}

    def forward(self, x, stages=None, class_ids=None, padding_mask=None):
        """
        前向传播:
          当输入 x 为时序数据时，x.shape 为 [B, T, C, H, W]；
          否则，x.shape 为 [B, C, H, W]。
          
          stages: 若启用时序建模，提供每个样本各时刻的阶段标签，形状 [B, T] 或 [B]（若为单帧）
          class_ids: 每个样本对应的类别ID，形状 [B]
        """
        if self.use_temporal and x.dim() == 5:
            # 时序数据模式：输入 x 的形状为 [B, T, C, H, W]
            B, T = x.shape[:2]
            # 将时序数据合并 batch 与时刻维度送入 backbone
            x_flat = x.flatten(0, 1)  # [B*T, C, H, W]
            feats = self.backbone(x_flat)["features"]  # 假设输出形状为 [B*T, D]
            # 还原时序形状 [B, T, D]
            temporal_feats = feats.view(B, T, -1)
            # 此处可采用时序聚合（例如均值池化、注意力池化等），简单示例中使用均值
            pooled_feats = temporal_feats.mean(dim=1)  # [B, D]
            
            if stages is not None and class_ids is not None:
                # 若提供阶段信息，则取每个样本最后时刻的阶段作为当前阶段
                # 假设 stages shape 为 [B, T]，取最后一列
                current_stage = stages[:, -1]  # [B]
                # 动态时序建模：输入 pooled_feats [B, D]，当前阶段 [B]，类别 [B]
                temporal_out, stage_logits = self.dynamic_temporal_model(pooled_feats, current_stage, class_ids)
                # 利用虫态原型池对时序特征进行增强
                enhanced_feats = self.apply_prototype_augmentation(temporal_out, class_ids, current_stage)
                logits = self.fc(enhanced_feats)
                return {
                    "logits": logits, 
                    "stage_logits": stage_logits, 
                    "features": enhanced_feats
                }
            else:
                # 若未提供阶段或类别信息，则直接分类
                logits = self.fc(pooled_feats)
                return {
                    "logits": logits, 
                    "features": pooled_feats
                }
        else:
            # 单帧数据模式：输入 x 的形状为 [B, C, H, W]
            feats = self.backbone(x)["features"]
            logits = self.fc(feats)
            return {"logits": logits, "features": feats}

    def apply_prototype_augmentation(self, features, class_ids, stages):
        """
        对每个样本，根据类别和当前阶段，如果原型池中存在对应原型，
        则将原型按一定比例（例如 0.5）加到特征上，得到增强后的特征。
        输入:
            features: [B, D]
            class_ids: [B]
            stages: [B]
        输出:
            enhanced_features: [B, D]
        """
        enhanced = []
        for f, cid, stage in zip(features, class_ids, stages):
            cid_str = str(cid.item())
            stage_str = str(stage.item())
            if cid_str in self.prototype_pool and stage_str in self.prototype_pool[cid_str]:
                proto = self.prototype_pool[cid_str][stage_str]
                enhanced.append(f + 0.5 * proto)
            else:
                enhanced.append(f)
        return torch.stack(enhanced)

    def update_prototype(self, class_id, stage, new_feature):
        """
        更新指定类别和阶段的原型池。若对应原型不存在，则初始化；
        否则采用动量更新（例如 0.9 老值 + 0.1 新值）。
        """
        cid_str = str(class_id)
        stage_str = str(stage)
        if cid_str not in self.prototype_pool:
            self.prototype_pool[cid_str] = nn.ModuleDict()
        if stage_str not in self.prototype_pool[cid_str]:
            self.prototype_pool[cid_str][stage_str] = nn.Parameter(new_feature.clone())
        else:
            with torch.no_grad():
                updated = 0.9 * self.prototype_pool[cid_str][stage_str].data + 0.1 * new_feature
                self.prototype_pool[cid_str][stage_str].data.copy_(updated)
