import os
from utils import transform as trs
import PIL.Image
from PIL import ImageFile, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from .register import DATASETS
from typing import Dict, Tuple, Union
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

@DATASETS.register()
class MarsDataset(Dataset):
    def __init__(
        self,
        cfg: Dict,
        is_labeled: bool,
        trs_form = None,
        trs_form_strong = None
        ) -> None:

        self.cfg = cfg
        self.data_root = cfg["dataset"]["data_root"]
        self.transform_weak = trs_form
        self.transform_strong = trs_form_strong
        self.trf_normalize = self._get_to_tensor_and_normalize(self.cfg["dataset"]["mean"], self.cfg["dataset"]["std"])
        self.trs = self._transforms(self.cfg)
        self.islabeled = is_labeled

        self.img = []

        self.mask_root = os.path.join(self.data_root, "mask")

        self.train_root = os.path.join(self.data_root, "image")

        self.unlabeled_image_root = os.path.join(self.data_root, "unlabeled_image")

        train_imgs = os.listdir(self.train_root)
        for img in train_imgs:
            if img.endswith("png"):
                self.img.append(img)

    def _load_image(self, index: int) ->Tuple:

        img_name = self.img[index]
        img_path = os.path.join(self.train_root, img_name)

        return PIL.Image.open(img_path).convert("RGB"), img_name

    def _load_unlabeled_image(self, index: int) ->Tuple:

        img_name = self.img[index]
        img_path = os.path.join(self.unlabeled_image_root, img_name)

        return PIL.Image.open(img_path).convert("RGB"), img_name

    def _load_target(self, index: int) -> Union[Tuple[PIL.Image.Image, PIL.Image.Image], PIL.Image.Image]:

        target_name = self.img[index]
        mask_path = os.path.join(self.mask_root, target_name)

        return  PIL.Image.open(mask_path).convert("L").point(lambda p: 1 if p > 0 else 0)

    @staticmethod
    def _transforms(cfg:dict) -> trs.Compose:

        ignore_label = cfg["dataset"]["ignore_label"]
        trs_form = []

        if cfg.get("rand_resize", True):
            trs_form.append(trs.Resize(cfg.get("resize_base_size", 600), cfg["dataset"]["train"]["rand_resize"]))

        if cfg.get("flip", True):
            trs_form.append(trs.RandomFlip(prob=0.5, flag_hflip=True))

        # crop also sometime for validating
        if cfg.get("crop", True):
            crop_size = cfg["dataset"]["train"]["crop"]["size"]
            crop_type = cfg["dataset"]["train"]["crop"].get("type", "rand")
            trs_form.append(trs.Crop(crop_size, crop_type=crop_type, mean=cfg["dataset"]["mean"], ignore_value=ignore_label))

        return trs.Compose(trs_form)

    @staticmethod
    def _get_to_tensor_and_normalize(mean, std):
        return trs.ToTensorAndNormalize(mean, std)


    def __getitem__(self, index: int) -> Union[Tuple[PIL.Image.Image, PIL.Image.Image, PIL.Image.Image, str], Tuple[PIL.Image.Image, PIL.Image.Image, str]]:
        if self.islabeled:

            image, name = self._load_image(index)
            label = self._load_target(index)
            image, label = self.trf_normalize(image, label)

            return image, label, name

        else:
            image, name = self._load_unlabeled_image(index)
            image_weak = self.transform_weak(image)
            image_strong = self.transform_strong(image_weak)

            image_weak = self.trf_normalize(image_weak)
            image_strong = self.trf_normalize(image_strong)

            return image_weak, image_strong, name

    def __len__(self) -> int:
        return len(self.img)

if __name__ == '__main__':
    pass




