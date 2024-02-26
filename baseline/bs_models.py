import numpy as np
import timm.models
import torch
import torchvision.models
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights, ViT_B_16_Weights, Swin_T_Weights


class mobilenet_v3(nn.Module):
    def __init__(self, num_classes1):
        """
        MobileNetV3
        :param num_classes1: output dimension
        """
        super(mobilenet_v3, self).__init__()
        self.backbone = torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_features = 576
        self.backbone.classifier = nn.Identity()
        self.head_label = nn.Linear(in_features, num_classes1)

    def forward(self, x):
        """
        forward pass of mobilenet_v3
        :param x: the provided input tensor
        :return: the current position
        """
        x = self.backbone(x)
        return self.head_label(x)


class resnet18(nn.Module):
    def __init__(self, num_classes1):
        """
        ResNet18
        :param num_classes1: output dimension
        """
        super(resnet18, self).__init__()
        self.backbone = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head_label = nn.Linear(in_features, num_classes1)

    def forward(self, x):
        """
        forward pass of resnet18
        :param x: the provided input tensor
        :return: the current position
        """
        x = self.backbone(x)
        return self.head_label(x)


class vit(nn.Module):
    def __init__(self, num_classes1):
        """
        Vision Transformer b-16
        :param num_classes1: output dimension
        """
        super(vit, self).__init__()
        self.backbone = torchvision.models.vit_b_16(ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = 384
        self.backbone.head = nn.Identity()
        self.head_label = nn.Linear(in_features, num_classes1)

    def forward(self, x):
        """
        forward pass of vit
        :param x: the provided input tensor
        :return: the current position
        """
        x = self.backbone(x)
        return self.head_label(x)


class swint(nn.Module):
    def __init__(self, num_classes1):
        """
        Swin Transformer tiny
        :param num_classes1: output dimension
        """
        super(swint, self).__init__()
        self.backbone = torchvision.models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        self.head_label = nn.Linear(in_features, num_classes1)

    def forward(self, x):
        """
        forward pass of swint
        :param x: the provided input tensor
        :return: the current position
        """
        x = self.backbone(x)
        return self.head_label(x)


class Extractor(nn.Module):
    def __init__(self, backbone):
        """
        Feature Extractor
        :param backbone: backbone of Feature Extractor (MobileNetV3 small)
        """
        super(Extractor, self).__init__()
        self.backbone = backbone
        self.backbone.classifier = nn.Identity()

    def forward(self, x):
        """
        forward pass of Extractor
        :param x: the provided input tensor
        :return: the visual semantic features of input
        """
        x = self.backbone(x)
        return x


class LSTM(nn.Module):
    def __init__(self, *, backbone, extractor_dim,
                 len, dim, num_layers,
                 num_classes1, num_classes2):
        """
        baseline (MobileNetV3 small + LSTM)
        :param backbone: backbone of Feature Extractor (MobileNetV3 small)
        :param extractor_dim: output dimension of Feature Extractor
        :param len: input sequence length of baseline
        :param dim: input dimension of LSTM
        :param num_layers: depth of LSTM
        :param num_classes1: output dimension of baseline
        :param num_classes2: output dimension of baseline
        """
        super().__init__()
        self.extractor = Extractor(backbone)
        self.extractor_dim = extractor_dim
        self.dim = dim
        self.len = len

        self.ang_linear = nn.Linear(2, dim)
        self.img_linear = nn.Linear(self.extractor_dim + dim, dim)

        self.lstm = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=num_layers, batch_first=True)

        self.mlp_head = nn.Linear(dim, 2 * dim)
        self.head_label = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes1)
        )
        self.head_target = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes1)
        )
        self.head_angle = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Hardtanh(),
            nn.Linear(dim, dim // 2),
            nn.Hardtanh(),
            nn.Linear(dim // 2, num_classes2)
        )

    def forward(self, img, ang):
        """
        forward pass of baseline
        :param img: input frame sequence
        :param ang: input angle sequence
        :return: the current position, the next position, the direction angle
        """
        # b,len,3,224,224->b*len,3,224,224->b*len,576->b,len,576
        img = self.extractor(img.view(-1, 3, 224, 224))
        img = img.view(-1, self.len, self.extractor_dim)

        # b,len,2->b,len,dim
        for i in range(1, self.len):
            ang[:, i, :] += ang[:, i - 1, :]
        ang = self.ang_linear(ang)

        # b,len,extractor_dim+dim->b,len,dim
        img = torch.cat((img, ang), dim=-1)
        img = self.img_linear(img)

        # b,len,dim->b,dim
        img, _ = self.lstm(img)
        img = img[:, -1, :]

        # b,dim->b,2*dim->2,b,dim
        img = self.mlp_head(img)
        ang = img[:, self.dim:]
        img = img[:, 0: self.dim]

        return self.head_label(img), self.head_target(img), self.head_angle(ang)
