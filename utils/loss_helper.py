# this script is changed from https://github.com/ZhenZHAO/AugSeg/blob/main/augseg/utils/loss_helper.py
import numpy as np
import scipy.ndimage as nd
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Sequence
from torch import Tensor
from typing import Optional, Tuple, Union

def get_criterion(cfg, type):
    cfg_criterion = cfg["criterion"]
    aux_weight = (
        cfg["net"]["aux_loss"]["loss_weight"]
        if cfg["net"].get("aux_loss", False)
        else 0
    )
    ignore_index = cfg["dataset"]["ignore_label"]
    num_classes = cfg['net']['num_classes']
    if type == "ohem":
        criterion = CriterionOhem(
            aux_weight, ignore_index=ignore_index, **cfg_criterion["kwargs"]
        )
    elif type == "focal":
        criterion = FocalLoss(
            gamma=2, reduction='mean', ignore_index=ignore_index,
        )
        criterion = LabelSmoothingCrossEntropy(ignore_index=ignore_index)
    elif type == "nl":
        criterion = NL(ignore_index=ignore_index)
    elif type == "ce":
        criterion = Criterion(
            aux_weight, ignore_index=ignore_index
        )

    elif type == "sup_hybrid_loss":
        criterion = HybridSCEDiceCriterion(
            aux_weight, ignore_index=ignore_index
        )
    elif type == "boundary_loss":
        criterion = BoundaryConstraint(
            band_radius=3, topk_ratio=0.2
        )

    return criterion

class NL(nn.Module):
    def __init__(self, ignore_index=255, use_weight=False):
        super(NL, self).__init__()
        self._ignore_index = ignore_index
        self.use_weight = use_weight
        if not use_weight:
            self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            weights = torch.FloatTensor(
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                ]
            ).cuda()
            self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
            self._criterion1 = nn.CrossEntropyLoss(
                ignore_index=ignore_index, weight=weights
            )

    def forward(self, preds, target, epoch):
        h, w = target.size(1), target.size(2)

        pred_h, pred_w = preds.size(2), preds.size(3)
        assert pred_h == h and pred_w == w
        loss = self._criterion(preds, target)
        return loss

class Criterion(nn.Module):
    def __init__(self, aux_weight, ignore_index=255, use_weight=False):
        super(Criterion, self).__init__()
        self._aux_weight = aux_weight
        self._ignore_index = ignore_index
        self.use_weight = use_weight
        if not use_weight:
            self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            weights = torch.FloatTensor(
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                ]
            ).cuda()
            self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
            self._criterion1 = nn.CrossEntropyLoss(
                ignore_index=ignore_index, weight=weights
            )

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred, aux_pred = preds
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)
            assert (
                len(preds) == 2
                and main_h == aux_h
                and main_w == aux_w
                and main_h == h
                and main_w == w
            )
            if self.use_weight:
                loss1 = self._criterion(main_pred, target) + self._criterion1(
                    main_pred, target
                )
            else:
                loss1 = self._criterion(main_pred, target)
            loss2 = self._criterion(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion(preds, target)
        return loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean',ignore_index=255):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target, epoch):
        c = output.size()[1]
        log_preds = F.log_softmax(output, dim=1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction,
                                                           ignore_index=self.ignore_index)

class LALoss(nn.Module):
    def __init__(
        self,
        cls_num_list,
        tau = 1.0
    ):
        super(LALoss, self).__init__()
        base_probs = cls_num_list / cls_num_list.sum()
        scaled_class_weights = tau * torch.log(base_probs + 1e-12)
        scaled_class_weights = scaled_class_weights.reshape(1, -1)  # [1,classnum]
        self.scaled_class_weights = scaled_class_weights.float().cuda()

class CriterionOhem(nn.Module):
    def __init__(
        self,
        aux_weight,
        thresh=0.7,
        min_kept=100000,
        ignore_index=255,
        use_weight=False,
    ):
        super(CriterionOhem, self).__init__()
        self._aux_weight = aux_weight
        self._criterion1 = OhemCrossEntropy2dTensor(
            ignore_index, thresh, min_kept, use_weight
        )
        self._criterion2 = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred, aux_pred = preds
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)
            assert (
                len(preds) == 2
                and main_h == aux_h
                and main_w == aux_w
                and main_h == h
                and main_w == w
            )

            loss1 = self._criterion1(main_pred, target)
            loss2 = self._criterion2(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion1(preds, target)
        return loss


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=255, thresh=0.7, min_kept=100000, factor=8):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.factor = factor
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def find_threshold(self, np_predict, np_target):
        # downsample 1/8
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0 / factor, 1.0 / factor), order=1)
        target = nd.zoom(np_target, (1.0, 1.0 / factor, 1.0 / factor), order=0)

        n, c, h, w = predict.shape
        min_kept = self.min_kept // (
            factor * factor
        )  # int(self.min_kept_ratio * n * h * w)

        input_label = target.ravel().astype(np.int32)
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if min_kept >= num_valid:
            threshold = 1.0
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                k_th = min(len(pred), min_kept) - 1
                new_array = np.partition(pred, k_th)
                new_threshold = new_array[k_th]
                if new_threshold > self.thresh:
                    threshold = new_threshold
        return threshold

    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape

        threshold = self.find_threshold(np_predict, np_target)

        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()

        if num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        new_target = (
            torch.from_numpy(input_label.reshape(target.size()))
            .long()
            .cuda(target.get_device())
        )

        return new_target

    def forward(self, predict, target, epoch, weight=None):
        """
        Args:
            predict:(n, c, h, w)
            target:(n, h, w)
            weight (Tensor, optional): a manual rescaling weight given to each class.
                                       If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad

        input_prob = F.softmax(predict, 1)
        target = self.generate_new_target(input_prob, target)
        return self.criterion(predict, target)

class OhemCrossEntropy2dTensor(nn.Module):
    """
    Ohem Cross Entropy Tensor Version
    """

    def __init__(
        self, ignore_index=255, thresh=0.7, min_kept=256, use_weight=False, reduce=False
    ):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [
                    0.8373,
                    0.918,
                    0.866,
                    1.0345,
                    1.0166,
                    0.9969,
                    0.9754,
                    1.0489,
                    0.8786,
                    1.0023,
                    0.9539,
                    0.9843,
                    1.1116,
                    0.9037,
                    1.0865,
                    1.0955,
                    1.0865,
                    1.1529,
                    1.0507,
                ]
            ).cuda()
            # weight = torch.FloatTensor(
            #    [0.4762, 0.5, 0.4762, 1.4286, 1.1111, 0.4762, 0.8333, 0.5, 0.5, 0.8333, 0.5263, 0.5882,
            #    1.4286, 0.5, 3.3333,5.0, 10.0, 2.5, 0.8333]).cuda()
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", weight=weight, ignore_index=ignore_index
            )
        elif reduce:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="none", ignore_index=ignore_index
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", ignore_index=ignore_index
            )

    def forward(self, pred, target, epoch):
        b, c, h, w = pred.size()
        target = target.view(-1)

        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()

        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
            # print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)

            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]

            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)

class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 2.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        pre = x
        target = y
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

def one_hot_mask(
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert target label map into one-hot encoding and generate a valid-pixel mask.

    Args:
        target (torch.Tensor): Ground truth tensor of shape [N, H, W], dtype=torch.long.
        num_classes (int): Number of classes.
        ignore_index (int): Label index to ignore in loss computation.

    Returns:
        y_onehot (torch.Tensor): One-hot encoded tensor of shape [N, C, H, W], dtype=torch.float32.
        valid_mask (torch.Tensor): Boolean mask of shape [N, 1, H, W], True for valid pixels.
    """
    valid_mask: torch.Tensor = (target != ignore_index).unsqueeze(1)
    temp: torch.Tensor = target.clone()
    temp[~valid_mask.squeeze(1)] = 0  # Set ignored pixels to 0 temporarily

    y_onehot: torch.Tensor = F.one_hot(temp, num_classes=num_classes).permute(0, 3, 1, 2).float()
    return y_onehot, valid_mask

class SymmetricCrossEntropy(nn.Module):
    """
    Symmetric Cross-Entropy (SCE) combines standard Cross-Entropy (CE) and Reverse Cross-Entropy (RCE).

    SCE = α * CE + β * RCE

    Attributes:
        alpha (float): Weight of the CE term.
        beta (float): Weight of the RCE term.
        ignore_index (int): Index of ignored class.
        label_smoothing (float): Label smoothing factor for numerical stability.
        class_weight (Optional[torch.Tensor]): Class weights for CE (optional).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        ignore_index: int = 255,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
        class_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.ignore_index: int = ignore_index
        self.label_smoothing: float = label_smoothing
        self.reduction: str = reduction

        if class_weight is not None:
            class_weight = torch.as_tensor(class_weight, dtype=torch.float32)
        self.register_buffer("class_weight", class_weight if class_weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SCE loss.

        Args:
            logits (torch.Tensor): Model outputs of shape [N, C, H, W].
            target (torch.Tensor): Ground truth labels of shape [N, H, W].

        Returns:
            torch.Tensor: Scalar SCE loss.
        """
        N, C, H, W = logits.shape

        # ----- Standard Cross-Entropy -----
        ce: torch.Tensor = F.cross_entropy(
            logits, target,
            ignore_index=self.ignore_index,
            weight=self.class_weight,
            reduction="none"
        )  # shape [N, H, W]

        valid_mask: torch.Tensor = (target != self.ignore_index)
        ce = ce[valid_mask]  # shape [M]

        if self.reduction == "mean":
            ce = ce.mean() if ce.numel() > 0 else logits.new_tensor(0.0)
        elif self.reduction == "sum":
            ce = ce.sum()

        # ----- Reverse Cross-Entropy -----
        p: torch.Tensor = F.softmax(logits, dim=1)
        y_onehot, valid_mask_4d = one_hot_mask(target, C, ignore_index=self.ignore_index)

        # Label smoothing for numerical stability
        eps: float = self.label_smoothing
        y_smooth: torch.Tensor = (1 - eps) * y_onehot + eps / C

        # Mask invalid pixels
        p_valid: torch.Tensor = p * valid_mask_4d
        y_valid: torch.Tensor = y_smooth * valid_mask_4d

        rce: torch.Tensor = -(p_valid * torch.log(torch.clamp(y_valid, min=1e-6))).sum(dim=1)
        denom: torch.Tensor = valid_mask_4d.sum().clamp(min=1)
        rce = rce.sum() / denom

        return self.alpha * ce + self.beta * rce


class DiceLoss(nn.Module):
    """
    Generalized multi-class Dice Loss.

    Attributes:
        ignore_index (int): Label index to ignore.
        smooth (float): Smoothing factor to avoid division by zero.
    """

    def __init__(self, ignore_index: int = 255, smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.ignore_index: int = ignore_index
        self.smooth: float = smooth
        self.reduction: str = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice Loss.

        Args:
            logits (torch.Tensor): Model outputs of shape [N, C, H, W].
            target (torch.Tensor): Ground truth labels of shape [N, H, W].

        Returns:
            torch.Tensor: Scalar Dice loss.
        """
        N, C, H, W = logits.shape
        probs: torch.Tensor = F.softmax(logits, dim=1)
        y_onehot, valid_mask = one_hot_mask(target, C, ignore_index=self.ignore_index)

        probs = probs * valid_mask
        y_onehot = y_onehot * valid_mask

        dims = (0, 2, 3)
        intersection: torch.Tensor = (probs * y_onehot).sum(dims)
        denominator: torch.Tensor = (probs * probs).sum(dims) + (y_onehot * y_onehot).sum(dims)

        dice_per_class: torch.Tensor = (2 * intersection + self.smooth) / (denominator + self.smooth)
        loss_per_class: torch.Tensor = 1.0 - dice_per_class

        if self.reduction == "mean":
            return loss_per_class.mean()
        elif self.reduction == "sum":
            return loss_per_class.sum()
        else:
            return loss_per_class

class HybridSCEDiceCriterion(nn.Module):
    """
    Hybrid loss combining Symmetric Cross-Entropy (SCE) and Dice Loss.

    Total Loss = w_sce * SCE + w_dice * Dice

    Attributes:
        aux_weight (float): Auxiliary loss weight.
        ignore_index (int): Label index to ignore.
        sce (SymmetricCrossEntropy): Instance of SCE loss.
        dice (DiceLoss): Instance of Dice loss.
    """

    def __init__(
        self,
        aux_weight: float = 0.0,
        ignore_index: int = 255,
        sce_alpha: float = 1.0,
        sce_beta: float = 1.0,
        label_smoothing: float = 0.1,
        class_weight: Optional[torch.Tensor] = None,
        dice_smooth: float = 1.0,
        w_sce: float = 1.0,
        w_dice: float = 1.0,
    ):
        super().__init__()
        self.aux_weight: float = aux_weight
        self.ignore_index: int = ignore_index
        self.w_sce: float = w_sce
        self.w_dice: float = w_dice

        self.sce: SymmetricCrossEntropy = SymmetricCrossEntropy(
            alpha=sce_alpha,
            beta=sce_beta,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction="mean",
            class_weight=class_weight,
        )
        self.dice: DiceLoss = DiceLoss(ignore_index=ignore_index, smooth=dice_smooth)

    def _compute_single_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the hybrid loss for a single prediction.

        Args:
            logits (torch.Tensor): Predicted logits [N, C, H, W].
            target (torch.Tensor): Ground truth labels [N, H, W].

        Returns:
            torch.Tensor: Scalar combined loss.
        """
        loss_sce: torch.Tensor = self.sce(logits, target)
        loss_dice: torch.Tensor = self.dice(logits, target)
        return self.w_sce * loss_sce + self.w_dice * loss_dice

    def forward(
        self,
        preds: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the hybrid criterion.

        Args:
            preds (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
                - Single tensor [N, C, H, W], or
                - Tuple of (main_pred, aux_pred) for auxiliary supervision.
            target (torch.Tensor): Ground truth tensor [N, H, W].

        Returns:
            torch.Tensor: Scalar hybrid loss.
        """
        if isinstance(preds, (list, tuple)):

            main_pred, aux_pred = preds

            # Resize predictions if spatial size mismatched
            if main_pred.shape[-2:] != target.shape[-2:]:
                main_pred = F.interpolate(main_pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
            if aux_pred.shape[-2:] != target.shape[-2:]:
                aux_pred = F.interpolate(aux_pred, size=target.shape[-2:], mode='bilinear', align_corners=False)

            main_loss: torch.Tensor = self._compute_single_loss(main_pred, target)
            aux_loss: torch.Tensor = self._compute_single_loss(aux_pred, target) if self.aux_weight > 0 else main_pred.new_tensor(0.0)
            return main_loss + self.aux_weight * aux_loss

        else:

            logits: torch.Tensor = preds
            if logits.shape[-2:] != target.shape[-2:]:
                logits = F.interpolate(logits, size=target.shape[-2:], mode='bilinear', align_corners=False)
            return self._compute_single_loss(logits, target)

class BoundaryConstraint(nn.Module):
    """
    Single constraint: within a tolerance band around the label boundary,
    the strong gradient of the original image is used as the target boundary;
    outside the band, the target is zero.
    Loss = BCEWithLogits(boundary_logits, target)
    """
    def __init__(self, band_radius: int = 3, topk_ratio: float = 0.2):
        """
        Args:
            band_radius (int): Radius (in pixels) of the tolerance band.
                Increase when pseudo-labels are less accurate (e.g., 3–5).
            topk_ratio (float): Ratio of top gradients used as strong edges (0.1–0.3).
        """
        super().__init__()
        self.r = band_radius
        self.topk_ratio = topk_ratio
        self.bce = nn.BCEWithLogitsLoss()

        # Sobel operators for edge detection
        self.register_buffer('kx', torch.tensor([[1, 0, -1],
                                                 [2, 0, -2],
                                                 [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3) / 8)
        self.register_buffer('ky', torch.tensor([[1, 2, 1],
                                                 [0, 0, 0],
                                                 [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3) / 8)

    @staticmethod
    def _dilate(x: torch.Tensor) -> torch.Tensor:
        """Morphological dilation using 3×3 max pooling."""
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def _erode(x: torch.Tensor) -> torch.Tensor:
        """Morphological erosion using negative max pooling."""
        return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)

    def _label_boundary_band(self, lab01: torch.Tensor) -> torch.Tensor:
        """
        Generate the tolerance band around the label boundary.

        Args:
            lab01 (Tensor): Binary label (B,1,H,W) with values {0,1,...}.
                Pixels > 0 are treated as foreground (multi-class can be simplified as foreground/background).
        Returns:
            band_mask (Tensor): Binary mask (B,1,H,W) in {0,1}.
        """
        # Approximate boundary via morphological gradient
        dil = self._dilate(lab01)
        ero = self._erode(lab01)
        boundary = (dil - ero).clamp(0, 1)

        # Expand the boundary by r pixels to form the tolerance band
        band = boundary
        for _ in range(self.r):
            band = self._dilate(band)
        return band

    def _image_strong_edges(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract the strong edges from the input image based on gradient magnitude.

        Args:
            image (Tensor): Input RGB image (B,3,H,W).
        Returns:
            edge_bin (Tensor): Binary edge map (B,1,H,W) with strong edges = 1.
        """
        gray = image.float().mean(dim=1, keepdim=True)

        # Ensure kernel and image tensors share the same device/dtype
        kx = self.kx.to(device=gray.device, dtype=gray.dtype)
        ky = self.ky.to(device=gray.device, dtype=gray.dtype)

        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        G = torch.sqrt(gx * gx + gy * gy + 1e-6)

        B, _, H, W = G.shape
        k = max(1, int(self.topk_ratio * H * W))
        tau = torch.topk(G.view(B, -1), k, dim=1).values[:, -1].view(B, 1, 1, 1)
        edge_bin = (G >= tau).float()
        return edge_bin

    def forward(self, boundary_logits: torch.Tensor, image: torch.Tensor, label: torch.Tensor):
        """
        Compute the boundary constraint loss.

        Args:
            boundary_logits (Tensor): Predicted boundary logits (B,1,H,W).
            image (Tensor): Original RGB input image (B,3,H,W).
            label (Tensor): Ground truth or pseudo-label mask (B,H,W).
        Returns:
            loss (Tensor): BCEWithLogits loss value.
            stats (dict): Dictionary with auxiliary statistics (e.g., positive ratio).
        """
        label = label.to(boundary_logits.device)
        image = image.to(boundary_logits.device).float()

        # (1) Tolerance band around label boundary
        lab01 = (label.unsqueeze(1) > 0).float()
        band = self._label_boundary_band(lab01)

        # (2) Strong edges from the image
        strong_edges = self._image_strong_edges(image)

        # (3) Target: inside band = strong edges, outside = 0
        target = band * strong_edges

        # Compute loss
        loss = self.bce(boundary_logits, target)
        return loss, {'pos_ratio': float(target.mean().item())}

def compute_unsupervised_loss_by_threshold(predict, target, logits, thresh=0.95, ep=0):

    batch_size, num_class, h, w = predict.shape
    thresh_mask = logits.ge(thresh).bool() * (target != 255).bool()
    target[~thresh_mask] = 255
    loss = F.cross_entropy(predict, target, ignore_index=255, reduction="none")
    # criterion = CycleFocalLoss(ignore_index=255, epochs=50)
    # loss = criterion(predict, target, ep)

    return loss.mean(), thresh_mask.float().mean()