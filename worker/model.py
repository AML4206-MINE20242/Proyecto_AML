
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torchvision.models.densenet import DenseNet121_Weights
from torchvision.models.efficientnet import EfficientNet_B7_Weights
from torchvision.models.efficientnet import *
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models.vision_transformer import ViT_B_16_Weights


class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(weights = DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_ftrs, out_size)
        
    def forward(self, x):
        x = self.densenet121(x)
        return x

class EfficientNetV2(nn.Module):
    def __init__(self, out_size):
        super(EfficientNetV2, self).__init__()
        self.efficientnetv2 = torchvision.models.efficientnet_v2_m(weights =EfficientNet_V2_M_Weights.IMAGENET1K_V1)
       
        num_ftrs = self.efficientnetv2.classifier[1].in_features
        self.efficientnetv2.classifier[1] = nn.Linear(num_ftrs, out_size)
        
    def forward(self, x):
        x = self.efficientnetv2(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, out_size):
        super(VisionTransformer, self).__init__()
        self.vit = torchvision.models.vision_transformer.vit_b_16(weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        
        num_ftrs = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(num_ftrs, out_size, bias=True)
        
    def forward(self, x):
        x = self.vit(x)
        return x
    