import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Dict, Tuple

def get_syncbn():
    # return nn.BatchNorm2d
    return nn.SyncBatchNorm

class dec_deeplabv3_plus(nn.Module):
    def __init__(
        self,
        in_planes,
        num_classes=19,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 24, 36),
        low_conv_planes=48,
        use_boundary: bool = False,
        boundary_mid_ratio: float = 0.5,
        boundary_use_bn: bool = True,
        boundary_min_mid: int = 8,
        use_domain: bool = False,
        grl_lambda: float = 0.0,
    ):
        super(dec_deeplabv3_plus, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d

        self.use_boundary = use_boundary
        self.use_domain = use_domain

        self.low_conv = nn.Sequential(
            nn.Conv2d(256, low_conv_planes, kernel_size=1),
            norm_layer(low_conv_planes),
            nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(
            in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations
        )

        self.head = nn.Sequential(
            nn.Conv2d(self.aspp.get_outplanes(), 256, 1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(256+int(low_conv_planes), 256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

        if self.use_boundary:
            self.boundary_head = DynamicBoundaryHead(
                mid_ratio=boundary_mid_ratio, use_bn=boundary_use_bn, min_mid=boundary_min_mid
            )

        if self.use_domain:
            self.grl = GradientReversalLayer(lambda_=grl_lambda)
            dh_kwargs = dict(mid_ratio=0.5, min_mid=64, use_bn=True, p_drop=0.2)
            self.domain_head = DomainClassifier(**dh_kwargs)


    def forward(self, x):
        x1, x2, x3, x4 = x
        low_feat = self.low_conv(x1)
        h, w = low_feat.size()[-2:]

        aspp_out = self.aspp(x4)
        aspp_out = self.head(aspp_out)
        aspp_out = F.interpolate(
            aspp_out, size=(h, w), mode="bilinear", align_corners=True
        )

        aspp_out = torch.cat((low_feat, aspp_out), dim=1)

        classification_head = self.classifier(aspp_out)

        out = {"classification_head": classification_head}

        if self.use_boundary:
            boundary_head = self.boundary_head(aspp_out)
            out["boundary_head"] = boundary_head

        if self.use_domain:
            d_feat = self.grl(aspp_out)
            domain_head = self.domain_head(d_feat)  # [B,1]
            out["domain_head"] = domain_head

        return out


class Aux_Module(nn.Module):
    def __init__(self, in_planes, num_classes=19, sync_bn=False):
        super(Aux_Module, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.aux = nn.Sequential(
            nn.Conv2d(in_planes, 256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        res = self.aux(x)
        return res


class ASPP(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(
        self, in_planes, inner_planes=256, sync_bn=False, dilations=(12, 24, 36)
    ):
        super(ASPP, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[0],
                dilation=dilations[0],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[1],
                dilation=dilations[1],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[2],
                dilation=dilations[2],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )

        self.out_planes = (len(dilations) + 2) * inner_planes

    def get_outplanes(self):
        return self.out_planes

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(
            self.conv1(x), size=(h, w), mode="bilinear", align_corners=True
        )
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        aspp_out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        return aspp_out

class DynamicBoundaryHead(nn.Module):

    def __init__(self, mid_ratio: float = 0.5, use_bn: bool = True, min_mid: int = 8) -> None:
        super().__init__()
        self.mid_ratio = float(mid_ratio)
        self.use_bn = bool(use_bn)
        self.min_mid = int(min_mid)
        self.block = None  # type: Optional[nn.Sequential]

    def _build(self, c_in: int, device: torch.device, dtype: torch.dtype) -> None:
        c_mid = max(self.min_mid, int(c_in * self.mid_ratio))
        layers = [nn.Conv2d(c_in, c_mid, kernel_size=3, padding=1, bias=False)]
        if self.use_bn:
            layers += [nn.BatchNorm2d(c_mid)]
        layers += [nn.ReLU(inplace=True), nn.Conv2d(c_mid, 1, kernel_size=1, bias=True)]
        self.block = nn.Sequential(*layers)

        self.block.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.block is None:
            self._build(c_in=int(x.shape[1]), device=x.device, dtype=x.dtype)
        return self.block(x)

class GradientReversalFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def set_lambda(self, lambda_: float):
        self.lambda_ = float(lambda_)

    def forward(self, x, lambda_: float = None):
        return GradientReversalFn.apply(x, self.lambda_)

class DomainClassifier(nn.Module):

    def __init__(self, mid_ratio: float = 0.5, min_mid: int = 64, use_bn: bool = True, p_drop: float = 0.2):
        super().__init__()
        self.mid_ratio = float(mid_ratio)
        self.min_mid = int(min_mid)
        self.use_bn = bool(use_bn)
        self.p_drop = float(p_drop)
        self.backbone: Optional[nn.Sequential] = None
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def _build(self, c_in: int, device, dtype):
        c_mid = max(self.min_mid, int(c_in * self.mid_ratio))
        layers = [nn.Conv2d(c_in, c_mid, kernel_size=1, bias=False)]
        if self.use_bn:
            layers.append(nn.BatchNorm2d(c_mid))
        layers += [nn.ReLU(inplace=True), nn.Dropout2d(self.p_drop), nn.Conv2d(c_mid, c_mid, kernel_size=1, bias=False)]
        if self.use_bn:
            layers.append(nn.BatchNorm2d(c_mid))
        layers += [nn.ReLU(inplace=True), nn.Conv2d(c_mid, 1, kernel_size=1, bias=True)]
        self.backbone = nn.Sequential(*layers).to(device=device, dtype=dtype)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if self.backbone is None:
            self._build(c_in=int(feat.shape[1]), device=feat.device, dtype=feat.dtype)
        x = self.gap(feat)          # (N,C,1,1)
        x = self.backbone(x)        # (N,1,1,1)
        x = x.flatten(1)            # (N,1) logits
        return x

