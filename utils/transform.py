import numpy as np
import scipy.stats as stats
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import random
import collections
import cv2
import torch
from torchvision import transforms

class Compose(object):
    def __init__(self, segtransforms):
        self.segtransforms = segtransforms

    def __call__(self, image, sparse_label, pseudo_label):
        for idx, t in enumerate(self.segtransforms):
            image, sparse_label, pseudo_label = t(image, sparse_label, pseudo_label)
        return image, sparse_label, pseudo_label

def to_tensor(pil_img):
    """PIL -> torch.float32 tensor in [0,1], shape (C,H,W)."""
    arr = np.array(pil_img, copy=False)
    if arr.ndim == 2:  # grayscale
        arr = arr[:, :, None]
    arr = torch.from_numpy(arr).float() / 255.0  # HWC in [0,1]
    return arr.permute(2, 0, 1).contiguous()     # CHW

class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    def __call__(self, x):
        # x: (C,H,W)
        if x.shape[0] != self.mean.shape[0]:
            if x.shape[0] == 1 and self.mean.shape[0] == 3:
                mean = self.mean[:1]
                std = self.std[:1]
            elif x.shape[0] == 3 and self.mean.shape[0] == 1:
                mean = self.mean.repeat(3,1,1)
                std = self.std.repeat(3,1,1)
            else:
                raise ValueError("Channel mismatch for normalization.")
        else:
            mean, std = self.mean, self.std
        return (x - mean) / std

class ToTensorAndNormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        assert len(mean) == len(std), "mean/std length mismatch"
        assert len(mean) in (1, 3), "mean/std must have length 1 or 3 (grayscale or RGB)"
        self.base_mean = list(mean)
        self.base_std = list(std)

    @staticmethod
    def _to_pil(img):
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW -> HWC
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            return Image.fromarray(img)
        if torch.is_tensor(img):
            x = img.detach().cpu()
            if x.ndim == 3 and x.shape[0] in (1, 3):  # CHW -> HWC
                x = x.permute(1, 2, 0)
            x = x.numpy()
            if x.dtype != np.uint8:
                x_min, x_max = x.min(), x.max()
                if x_max <= 1.0 and x_min >= 0.0:
                    x = (x * 255.0).astype(np.uint8)
                else:
                    x = np.clip(x, 0, 255).astype(np.uint8)
            return Image.fromarray(x)
        raise TypeError(f"Unsupported image type: {type(img)}")

    @staticmethod
    def _to_label_tensor(label):
        if label is None:
            return None
        if isinstance(label, Image.Image):
            arr = np.array(label, dtype=np.int32)
        elif isinstance(label, np.ndarray):
            arr = label.astype(np.int32, copy=False)
        elif torch.is_tensor(label):
            return label.long()
        else:
            raise TypeError(f"Unsupported label type: {type(label)}")
        return torch.from_numpy(arr).long()

    def _build_normalizer(self, c):
        if c == len(self.base_mean):
            mean, std = self.base_mean, self.base_std
        elif c == 1 and len(self.base_mean) == 3:
            mean, std = [self.base_mean[0]], [self.base_std[0]]
        elif c == 3 and len(self.base_mean) == 1:
            mean, std = self.base_mean * 3, self.base_std * 3
        else:
            raise ValueError(
                f"Channel count {c} incompatible with mean/std length {len(self.base_mean)}"
            )
        return Normalize(mean, std)

    def __call__(self, *args):
        if len(args) == 1:
            in_image = args[0]
            in_sparse_label = None
            in_pseudo_label = None
        elif len(args) == 2:
            in_image, in_sparse_label = args
            in_pseudo_label = None
        elif len(args) == 3:
            in_image, in_sparse_label, in_pseudo_label = args
        else:
            raise ValueError(f"Expected 1, 2 or 3 arguments, but got {len(args)}.")

        pil_img = self._to_pil(in_image)
        img_tensor = to_tensor(pil_img)              # [C,H,W] in [0,1]
        normalizer = self._build_normalizer(img_tensor.shape[0])
        image = normalizer(img_tensor)

        if in_sparse_label is None and in_pseudo_label is None:
            return image
        sparse_label = self._to_label_tensor(in_sparse_label)
        if in_pseudo_label is not None:
            pseudo_label = self._to_label_tensor(in_pseudo_label)
            return image, sparse_label, pseudo_label
        else:
            return image, sparse_label

class Resize(object):
    def __init__(self, base_size, ratio_range, scale=True, bigger_side_to_base_size=True):
        assert isinstance(ratio_range, collections.Iterable) and len(ratio_range) == 2
        self.base_size = base_size
        self.ratio_range = ratio_range
        self.scale = scale
        self.bigger_side_to_base_size = bigger_side_to_base_size

    def __call__(self, in_image, in_sparse_label, in_pseudo_label):
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
            sparse_label = in_sparse_label.resize((ow, oh), Image.NEAREST)
            pseudo_label = in_pseudo_label.resize((ow, oh), Image.NEAREST)

            return image, sparse_label, pseudo_label

        elif (isinstance(self.base_size, list) or isinstance(self.base_size, tuple)) and len(self.base_size) == 2:
            if self.scale:
                # scale = random.random() * 1.5 + 0.5  # Scaling between [0.5, 2]
                scale = self.ratio_range[0] + random.random() * (self.ratio_range[1] - self.ratio_range[0])
                oh, ow = int(self.base_size[0] * scale), int(self.base_size[1] * scale)
            else:
                oh, ow = self.base_size

            image = in_image.resize((ow, oh), Image.BILINEAR)
            sparse_label = in_sparse_label.resize((ow, oh), Image.NEAREST)
            pseudo_label = in_pseudo_label.resize((ow, oh), Image.NEAREST)

            return image, sparse_label, pseudo_label

        else:
            raise ValueError

class Crop(object):
    def __init__(self, crop_size, crop_type="rand", mean=[0.485, 0.456, 0.406], ignore_value=255):
        if (isinstance(crop_size, list) or isinstance(crop_size, tuple)) and len(crop_size) == 2:
            self.crop_h, self.crop_w = crop_size
        elif isinstance(crop_size, int):
            self.crop_h, self.crop_w = crop_size, crop_size
        else:
            raise ValueError

        self.crop_type = crop_type
        self.image_padding = (np.array(mean) * 255.).tolist()
        self.ignore_value = ignore_value

    def __call__(self, in_image, in_sparse_label, in_pseudo_label):
        # Padding to return the correct crop size
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
            sparse_label = cv2.copyMakeBorder(np.asarray(in_sparse_label, dtype=np.int32),
                                       value=self.ignore_value, **pad_kwargs)
            pseudo_label = cv2.copyMakeBorder(np.asarray(in_pseudo_label, dtype=np.int32),
                                              value=self.ignore_value, **pad_kwargs)

            image = Image.fromarray(np.uint8(image))
            sparse_label = Image.fromarray(np.uint8(sparse_label))
            pseudo_label = Image.fromarray(np.uint8(pseudo_label))

        else:
            image = in_image
            sparse_label = in_sparse_label
            pseudo_label = in_pseudo_label

        # cropping
        w, h = image.size
        if self.crop_type == "rand":
            x = random.randint(0, w - self.crop_w)
            y = random.randint(0, h - self.crop_h)
        else:
            x = (w - self.crop_w) // 2
            y = (h - self.crop_h) // 2

        image = image.crop((x, y, x + self.crop_w, y + self.crop_h))
        sparse_label = sparse_label.crop((x, y, x + self.crop_w, y + self.crop_h))
        pseudo_label = pseudo_label.crop((x, y, x + self.crop_w, y + self.crop_h))

        return image, sparse_label, pseudo_label

class RandomFlip(object):
    def __init__(self, prob=0.5, flag_hflip=True, ):
        self.prob = prob
        if flag_hflip:
            self.type_flip = Image.FLIP_LEFT_RIGHT
        else:
            self.type_flip = Image.FLIP_TOP_BOTTOM

    def __call__(self, in_image, in_sparse_label, in_pseudo_label):
        if random.random() < self.prob:
            in_image = in_image.transpose(self.type_flip)
            in_sparse_label = in_sparse_label.transpose(self.type_flip)
            in_pseudo_label = in_pseudo_label.transpose(self.type_flip)
        return in_image, in_sparse_label, in_pseudo_label


if __name__ == '__main__':

    batch = torch.randint(0, 256, (4, 3, 256, 256), dtype=torch.uint8)

    # Instantiate the transform
    t = ToTensorAndNormalize()

    # Apply to each image in the batch (no labels)
    out_tensors = []
    for i in range(batch.shape[0]):
        img_chw = batch[i]  # (3,256,256)
        img_tensor = t(img_chw)  # normalized float tensor (3,256,256)
        out_tensors.append(img_tensor)

