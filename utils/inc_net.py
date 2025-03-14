import copy
import logging
import torch
from torch import nn
from backbone.linears import SimpleLinear, CosineLinear
import timm

def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    
    if '_mos' in name or '_mos_modified' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "mos" or args["model_name"] == "mos_modified":
            from backbone import vit_mos
            from easydict import EasyDict
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
            )
            if name == "vit_base_patch16_224_mos" or name == "vit_base_patch16_224_mos_modified":
                model = vit_mos.vit_base_patch16_224_mos(num_classes=args["nb_classes"],
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
            elif name == "vit_base_patch16_224_in21k_mos" or name == "vit_base_patch16_224_in21k_mos_modified":
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

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

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

class MOSNet(nn.Module):
    def __init__(self, args, pretrained):
        super(MOSNet, self).__init__()
        self.backbone = get_backbone(args, pretrained)
        self.backbone.out_dim = 768
        self.fc = None
        self._device = args["device"][0]

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
        res = self.backbone(x, adapter_id, train, fc_only)

        return res
