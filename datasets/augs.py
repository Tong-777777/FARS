import numpy as np
import scipy.stats as stats
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import random
import collections
import cv2
import torch
from torchvision import transforms
import torch.nn.functional as F


# =========================
# 工具函数
# =========================
def _to_pil(img):
    """把 numpy 或 PIL 统一成 PIL.Image"""
    if isinstance(img, Image.Image):
        return img
    return Image.fromarray(np.uint8(img))


def _is_pair_call(image, label):
    """判断当前调用是否有标签"""
    return label is not None


# =========================
# 1. Augmentation for image and (optional) labels
# =========================
class Compose(object):
    """
    既支持 (image, label) 也支持 (image)。
    - 传入 (image, label) -> 返回 (image, label)
    - 传入 (image, None)  -> 返回 image
    - 传入 (image)        -> 返回 image
    """
    def __init__(self, segtransforms):
        self.segtransforms = segtransforms

    def __call__(self, image, label=None):
        has_label = _is_pair_call(image, label)
        if has_label:
            img, lab = image, label
            for t in self.segtransforms:
                if isinstance(t, strong_img_aug):
                    img = t(_to_pil(img))
                else:
                    img, lab = t(_to_pil(img), _to_pil(lab))
            return img, lab
        else:
            img = image
            for t in self.segtransforms:
                if isinstance(t, strong_img_aug):
                    img = t(_to_pil(img))
                else:
                    # 这些算子在 label=None 时只处理图像
                    img, _ = t(_to_pil(img), None)
            return img


class ToTensorAndNormalize(object):
    """
    - (image, label) -> (tensor_image, tensor_label)
    - (image, None)  -> tensor_image
    """
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        assert len(mean) == len(std) == 3
        self.normalize = transforms.Normalize(mean, std)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, in_image, in_label=None):
        in_image = _to_pil(in_image)
        image = self.normalize(self.to_tensor(in_image))

        if in_label is None:
            return image, None  # Compose 会自动只返回 image
        else:
            # 标签转 long（不做标准化）
            label_np = np.array(in_label, dtype=np.int32)
            label = torch.from_numpy(label_np).long()
            return image, label


class Resize(object):
    """
    支持：
    - base_size=int：按长边缩放（可随机比例）
    - base_size=(h, w)：整图同比例缩放（可随机比例）
    标签为 None 时，仅缩放图像。
    """
    def __init__(self, base_size, ratio_range, scale=True, bigger_side_to_base_size=True):
        assert isinstance(ratio_range, collections.abc.Iterable) and len(ratio_range) == 2
        self.base_size = base_size
        self.ratio_range = ratio_range
        self.scale = scale
        self.bigger_side_to_base_size = bigger_side_to_base_size

    def __call__(self, in_image, in_label=None):
        in_image = _to_pil(in_image)
        w, h = in_image.size

        if isinstance(self.base_size, int):
            # obtain long_side
            if self.scale:
                long_side = random.randint(int(self.base_size * self.ratio_range[0]),
                                           int(self.base_size * self.ratio_range[1]))
            else:
                long_side = self.base_size

            # obtain new oh, ow
            if self.bigger_side_to_base_size:
                if h > w:
                    oh = long_side
                    ow = int(1.0 * long_side * w / h + 0.5)
                else:
                    oh = int(1.0 * long_side * h / w + 0.5)
                    ow = long_side
            else:
                oh, ow = (long_side, int(1.0 * long_side * w / h + 0.5)) if h < w else (
                    int(1.0 * long_side * h / w + 0.5), long_side)

            image = in_image.resize((ow, oh), Image.BILINEAR)

            if in_label is None:
                return image, None
            else:
                label = _to_pil(in_label).resize((ow, oh), Image.NEAREST)
                return image, label

        elif (isinstance(self.base_size, (list, tuple)) and len(self.base_size) == 2):
            if self.scale:
                scale = self.ratio_range[0] + random.random() * (self.ratio_range[1] - self.ratio_range[0])
                oh, ow = int(self.base_size[0] * scale), int(self.base_size[1] * scale)
            else:
                oh, ow = self.base_size

            image = in_image.resize((ow, oh), Image.BILINEAR)
            if in_label is None:
                return image, None
            else:
                label = _to_pil(in_label).resize((ow, oh), Image.NEAREST)
                return image, label
        else:
            raise ValueError("base_size must be int or (h,w)")


class Crop(object):
    """
    随机/中心裁剪。若尺寸不足会用均值填充图像、ignore_value(默认255)填充标签。
    标签为 None 时，只裁剪图像。
    """
    def __init__(self, crop_size, crop_type="rand", mean=[0.485, 0.456, 0.406], ignore_value=255):
        if isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
            self.crop_h, self.crop_w = crop_size
        elif isinstance(crop_size, int):
            self.crop_h, self.crop_w = crop_size, crop_size
        else:
            raise ValueError

        self.crop_type = crop_type
        self.image_padding = (np.array(mean) * 255.).tolist()
        self.ignore_value = ignore_value

    def __call__(self, in_image, in_label=None):
        in_image = _to_pil(in_image)
        w, h = in_image.size

        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,
        }

        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(np.asarray(in_image, dtype=np.float32),
                                       value=self.image_padding, **pad_kwargs)
            image = Image.fromarray(np.uint8(image))
            if in_label is not None:
                label_np = np.asarray(_to_pil(in_label), dtype=np.int32)
                label_np = cv2.copyMakeBorder(label_np, value=self.ignore_value, **pad_kwargs)
                label = Image.fromarray(np.uint8(label_np))
            else:
                label = None
        else:
            image = in_image
            label = _to_pil(in_label) if in_label is not None else None

        # cropping
        w, h = image.size
        if self.crop_type == "rand":
            x = random.randint(0, w - self.crop_w)
            y = random.randint(0, h - self.crop_h)
        else:
            x = (w - self.crop_w) // 2
            y = (h - self.crop_h) // 2

        image = image.crop((x, y, x + self.crop_w, y + self.crop_h))
        if label is not None:
            label = label.crop((x, y, x + self.crop_w, y + self.crop_h))
            return image, label
        else:
            return image, None


class RandomFlip(object):
    """
    随机水平或垂直翻转。label 为 None 时，只翻图像。
    """
    def __init__(self, prob=0.5, flag_hflip=True):
        self.prob = prob
        self.type_flip = Image.FLIP_LEFT_RIGHT if flag_hflip else Image.FLIP_TOP_BOTTOM

    def __call__(self, in_image, in_label=None):
        in_image = _to_pil(in_image)
        if random.random() < self.prob:
            in_image = in_image.transpose(self.type_flip)
            if in_label is not None:
                in_label = _to_pil(in_label).transpose(self.type_flip)
        return in_image, (in_label if in_label is not None else None)


# =========================
# 2. Strong Augmentation for image only
# =========================
def img_aug_identity(img, scale=None):
    return img

def img_aug_autocontrast(img, scale=None):
    return ImageOps.autocontrast(img)

def img_aug_equalize(img, scale=None):
    return ImageOps.equalize(img)

def img_aug_invert(img, scale=None):
    return ImageOps.invert(img)

def img_aug_blur(img, scale=[0.1, 2.0]):
    assert scale[0] < scale[1]
    sigma = np.random.uniform(scale[0], scale[1])
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))

def img_aug_contrast(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = max_v - (max_v - min_v) * random.random()
    return ImageEnhance.Contrast(img).enhance(v)

def img_aug_brightness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = max_v - (max_v - min_v) * random.random()
    return ImageEnhance.Brightness(img).enhance(v)

def img_aug_color(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = max_v - (max_v - min_v) * random.random()
    return ImageEnhance.Color(img).enhance(v)

def img_aug_sharpness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = max_v - (max_v - min_v) * random.random()
    return ImageEnhance.Sharpness(img).enhance(v)

def img_aug_hue(img, scale=[0, 0.5]):
    min_v, max_v = min(scale), max(scale)
    v = min_v + (max_v - min_v) * random.random()
    hue_factor = -v if np.random.random() < 0.5 else v
    input_mode = img.mode
    if input_mode in {"L", "1", "I", "F"}:
        return img
    h, s, v_ = img.convert("HSV").split()
    np_h = np.array(h, dtype=np.uint8)
    with np.errstate(over="ignore"):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, "L")
    img = Image.merge("HSV", (h, s, v_)).convert(input_mode)
    return img

def img_aug_posterize(img, scale=[4, 8]):
    min_v, max_v = min(scale), max(scale)
    v = int(np.ceil((max_v - min_v) * random.random()))
    v = max(1, v)
    v = max_v - v
    return ImageOps.posterize(img, v)

def img_aug_solarize(img, scale=[1, 256]):
    min_v, max_v = min(scale), max(scale)
    v = int(np.ceil((max_v - min_v) * random.random()))
    v = max(1, v)
    v = max_v - v
    return ImageOps.solarize(img, v)

def get_augment_list(flag_using_wide=False):
    if flag_using_wide:
        l = [
            (img_aug_identity, None),
            (img_aug_autocontrast, None),
            (img_aug_equalize, None),
            (img_aug_blur, [0.1, 2.0]),
            (img_aug_contrast, [0.1, 1.8]),
            (img_aug_brightness, [0.1, 1.8]),
            (img_aug_color, [0.1, 1.8]),
            (img_aug_sharpness, [0.1, 1.8]),
            (img_aug_posterize, [2, 8]),
            (img_aug_solarize, [1, 256]),
            (img_aug_hue, [0, 0.5]),
        ]
    else:
        l = [
            (img_aug_identity, None),
            (img_aug_autocontrast, None),
            (img_aug_equalize, None),
            (img_aug_blur, [0.1, 2.0]),
            (img_aug_contrast, [0.05, 0.95]),
            (img_aug_brightness, [0.05, 0.95]),
            (img_aug_color, [0.05, 0.95]),
            (img_aug_sharpness, [0.05, 0.95]),
            (img_aug_posterize, [4, 8]),
            (img_aug_solarize, [1, 256]),
            (img_aug_hue, [0, 0.5]),
        ]
    return l


class strong_img_aug:
    """
    只增强图像，不动标签。可安全放在 Compose 里；
    当 Compose 接收 (image, label) 时，label 原样透传。
    """
    def __init__(self, num_augs, flag_using_random_num=False, flag_using_wide=False):
        assert 1 <= num_augs <= 11
        self.n = num_augs
        self.augment_list = get_augment_list(flag_using_wide=flag_using_wide)
        self.flag_using_random_num = flag_using_random_num

    def __call__(self, img):
        img = _to_pil(img)
        max_num = np.random.randint(1, high=self.n + 1) if self.flag_using_random_num else self.n
        ops = random.choices(self.augment_list, k=max_num)
        for op, scales in ops:
            img = op(img, scales)
        return img

def randow_window(image):
    height, wight = image.shape
    max_length = int(height * 0.25)
    # default min_length = 30
    min_length = 30
    window_height = np.random.randint(min_length, max_length + 1)
    window_width = np.random.randint(min_length, max_length + 1)
    return window_height, window_width

def find_lowest_avg_area_rect(confidence, window_height, window_width):

    # l_bbx1, l_bby1, l_bbx2, l_bby2 = rand_bbox(confidence.size(), lam=np.random.beta(8, 2))

    confidence = confidence.unsqueeze(0).unsqueeze(0)

    kernel = torch.ones(1, 1, window_height, window_width, device=confidence.device) / (window_height * window_width)

    result = F.conv2d(confidence, kernel, stride=min(window_height, window_width), padding=0)

    result_flat = result.view(result.size(0), result.size(1), -1)  # Flatten H and W into a single dimension
    min_value, min_flat_idx = torch.min(result_flat, dim=2)  # Find min value and index in the flattened view

    # Convert the flat index back to two dimensions
    min_idx_h = min_flat_idx // result.size(3)
    min_idx_w = min_flat_idx % result.size(3)

    l_bbx1, l_bby1 = min_idx_h.item(), min_idx_w.item()
    l_bbx2, l_bby2 = l_bbx1 + window_height - 1, l_bby1 + window_width - 1

    return l_bbx1, l_bby1, l_bbx2, l_bby2

def cut_mix_label_lowerest_area(unlabeled_image, unlabeled_mask, unlabeled_logits,
        labeled_image, labeled_mask, confidences, lst_confidences):
    assert len(lst_confidences) == len(unlabeled_image), "Ensure the confidence is properly obtained"
    assert labeled_image.shape == unlabeled_image.shape, "Ensure shape match between lb and unlb"
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    labeled_logits = torch.ones_like(labeled_mask)

    # 1) get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]

    # 2) get box
    # l_bbx1, l_bby1, l_bbx2, l_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(8, 2))
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))

    for i in range(0, mix_unlabeled_image.shape[0]):
        if np.random.random() > lst_confidences[i]:
        # if np.random.random() > np.random.random():
            confidence = confidences[i]
            window_height, window_width = randow_window(confidence)

            l_bbx1, l_bby1, l_bbx2, l_bby2 = find_lowest_avg_area_rect(confidence, window_height, window_width)

            mix_unlabeled_image[i, :, l_bbx1:l_bbx2, l_bby1:l_bby2] = \
                labeled_image[u_rand_index[i], :, l_bbx1:l_bbx2, l_bby1:l_bby2]

            mix_unlabeled_target[i, l_bbx1:l_bbx2, l_bby1:l_bby2] = \
                labeled_mask[u_rand_index[i], l_bbx1:l_bbx2, l_bby1:l_bby2]

            mix_unlabeled_logits[i, l_bbx1:l_bbx2, l_bby1:l_bby2] = \
                labeled_logits[u_rand_index[i], l_bbx1:l_bbx2, l_bby1:l_bby2]

    for i in range(0, unlabeled_image.shape[0]):
        unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            mix_unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        unlabeled_mask[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            mix_unlabeled_target[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            mix_unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    del mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, labeled_logits

    return unlabeled_image, unlabeled_mask, unlabeled_logits


def rand_bbox(size, lam=None):
    # past implementation
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception
    B = size[0]

    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W / 8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H / 8), high=H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
