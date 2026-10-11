from datasets import *
import logging
from torch.utils.data import DataLoader
from datasets import augs as img_trsform

logger = logging.getLogger("global")

def get_loader(cfg, baseline = False):

    train_batch_size = cfg["dataset"]["train"]["batch_size"]

    workers = cfg.get("workers", 0)

    labeled_train_sample_loader = MarsDataset(cfg, is_labeled=True)

    labeled_train_loader = DataLoader(
        labeled_train_sample_loader,
        batch_size=train_batch_size,
        num_workers=workers,
        shuffle=True,
        pin_memory=True,
    )

    if baseline:

        return labeled_train_loader

    else:

        trs_form_weak = build_basic_transfrom(cfg,)

        trs_form_strong = build_additional_strong_transform(cfg)

        unlabeled_train_sample_loader = MarsDataset(cfg, is_labeled=False, trs_form=trs_form_weak, trs_form_strong=trs_form_strong)

        unlabeled_train_loader = DataLoader(
            unlabeled_train_sample_loader,
            batch_size=train_batch_size,
            num_workers=workers,
            shuffle=True,
            pin_memory=True,
        )

        return labeled_train_loader, unlabeled_train_loader

def build_basic_transfrom(cfg, mean=[0.485, 0.456, 0.406]):
    ignore_label = cfg["dataset"]["ignore_label"]

    trs_form = []

    if cfg.get("rand_resize", False):
        trs_form.append(img_trsform.Resize(cfg.get("resize_base_size", 600), cfg["rand_resize"]))
    if cfg.get("flip", False):
        trs_form.append(img_trsform.RandomFlip(prob=0.5, flag_hflip=True))

    # crop also sometime for validating
    if cfg.get("crop", False):
        crop_size = cfg["crop"]["size"]
        crop_type = cfg["crop"].get("type", "rand")
        trs_form.append(img_trsform.Crop(crop_size, crop_type=crop_type, mean=mean, ignore_value=ignore_label))

    return img_trsform.Compose(trs_form)

def build_additional_strong_transform(cfg):

    strong_aug_nums = cfg["dataset"]["strong_aug"].get("num_augs", 2)
    flag_use_rand_num = cfg["dataset"]["strong_aug"].get("flag_use_random_num_sampling", True)
    strong_img_aug = img_trsform.strong_img_aug(strong_aug_nums,
            flag_using_random_num=flag_use_rand_num)
    return strong_img_aug
