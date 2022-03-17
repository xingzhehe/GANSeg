import torch
import math
from torch import nn
import torchvision
import torch.nn.functional as F
from torchvision import models


class Segmenter(nn.Module):
    def __init__(self, hyper_paras):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet50(num_classes=hyper_paras['n_classes'],
                                                            pretrained=False, progress=False)

    def forward(self, input_dict):
        return {'seg': self.model(input_dict['img'])['out']}
