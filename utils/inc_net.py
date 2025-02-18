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
            from utils.multi_proto import InsectAwareProtoPool, StageAwareLoss
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


class CosineIncrementalNet(BaseNet):  # 基于余弦相似度分类器
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num  ==  1:  # 第一个任务：直接迁移到新分类器
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:  # 后续任务：将旧fc1和fc2权重分别迁移到新分类器对应部分，再迁移sigma
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:  # 初始任务
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)  # 输出通过代理合并
        else:  # 增量任务
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
            )  # 两部分：旧任务类别数和新任务类别数

        return fc


from utils.multi_proto import InsectAwareProtoPool, StageAwareLoss
class MOSNet(nn.Module):
    def __init__(self, args, pretrained):
        super(MOSNet, self).__init__()
        # 获取预训练 backbone (假设 get_backbone 已实现)
        self.backbone = get_backbone(args, pretrained)
        self.backbone.out_dim = args["tuning_config"].embed_dim  # 设置输出维度
        self.fc = None  # 根据需要扩展分类头
        self._device = args["device"][0]
        self.config = args["tuning_config"]
    
        # 新增多原型虫态模块
        if self.config.proto_on:
            self.proto_pool = InsectAwareProtoPool(
                dim=self.backbone.out_dim,
                config=self.config
            )
            self.stage_loss = StageAwareLoss(max_stages=self.config.max_stages)
            self.stage_head = nn.Sequential(
                nn.Linear(self.backbone.out_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.config.max_stages)
            )

        # 原始分类头（此处可以根据需要扩展为双任务）
        self.head = nn.Linear(self.backbone.out_dim, args["num_classes"])

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc.requires_grad_(False)
        
        in_dim = self.backbone.out_dim
        self.head = nn.Linear(in_dim, nb_classes)
        
        # 同时扩展原型池
        if self.config.proto_on:
            for cid in range(len(self.proto_pool.class_protos), nb_classes):
                self.proto_pool.init_class(cid)
    
    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc
    
    def forward_orig(self, x):
        features = self.backbone(x, adapter_id=0)['features']
        
        res = dict()
        res['features'] = features
        res['logits'] = self.fc(features)['logits']
                
        return res
        
    def forward(self, x, adapter_id=-1, train=False, fc_only=False):
        # 通过 backbone 提取特征
        res = self.backbone(x, adapter_id, train, fc_only)
        # res 格式假设为 dict: {"features": tensor, "logits": tensor}
        if self.config.proto_on and not fc_only:
            # 需要提前设置当前 batch 的虫态标签
            if hasattr(self, '_current_labels'):
                # 当前 labels 格式: [B, 2]，第一列为类别，第二列为虫态
                class_ids = self._current_labels[:, 0]
                stages = self._current_labels[:, 1]
                # 利用多原型模块增强特征
                proto_enhanced = self.proto_pool(res["features"], class_ids, stages)
                res["features"] = res["features"] + 0.5 * proto_enhanced
                # 虫态预测
                res["stage_logits"] = self.stage_head(res["features"])
            else:
                # 若未设置，则不做原型增强
                res["stage_logits"] = self.stage_head(res["features"])
        # 分类预测
        res["logits"] = self.head(res["features"])
        return res

    def set_current_labels(self, labels):
        """
        注入当前 batch 的虫态标签
        labels: [B, 2] (第一列为类别ID，第二列为虫态ID)
        """
        self._current_labels = labels.long().to(self._device)

    def update_protos(self, features):
        """
        双路径原型更新，利用当前 batch 的虫态标签
        """
        if hasattr(self, '_current_labels'):
            class_ids = self._current_labels[:, 0]
            stages = self._current_labels[:, 1]
            self.proto_pool.update_protos(features.detach(), class_ids, stages)

    def get_loss(self, outputs, labels):
        """
        多任务损失计算：
          - 分类损失 (CE) 针对昆虫种类
          - 虫态转移损失 (StageAwareLoss) 针对虫态
        """
        cls_loss = F.cross_entropy(outputs['logits'], labels[:, 0])
        stage_loss = self.stage_loss(outputs['stage_logits'], labels[:, 1])
        return cls_loss + self.config.stage_loss_weight * stage_loss