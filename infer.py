import argparse
from typing import Tuple

import yaml
import torch
import torch.nn as nn
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
from models import ModelBuilder
from utils.utils import AverageMeter, intersectionAndUnion
from utils.data_helper import get_loader

# INFER_PATH = r'F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\datasets\rough'
INFER_PATH = r'F:\mars_rock\Utopia Planitia\inference_2025_non_overlap_86_by_binning\pure_2x2_binning_8\ESP_086649_2155\img'
INFER_SAVE = r'F:\mars_rock\Utopia Planitia\inference_2025_non_overlap_86_by_binning\pure_2x2_binning_8\ESP_086649_2155\infer'
os.makedirs(INFER_SAVE, exist_ok=True)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def TIANWEN():
    class_names = ['background', 'rock']
    palette = [
        [0, 0, 0],  # background: Black,
        [255, 255, 255]  # rock: White,
    ]
    return class_names, palette

def preprocess_image(img_path, device="cuda"):
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device), img

def main(args):
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    ###########################
    # 1. output settings
    ###########################
    base_dir = os.path.dirname(args.config)

    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    weight_path = None
    for file_name in os.listdir(checkpoints_dir):
        if 'best' in file_name and file_name.endswith(".pth"):
            weight_path = os.path.join(checkpoints_dir, file_name)
            break
        weight_path = os.path.join(checkpoints_dir, 'ckpt.pth')
    if weight_path is None:
        raise FileNotFoundError("No best model checkpoint found.")

    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # prepare model
    model = ModelBuilder(cfg["net"], is_student=False)
    modules_back = [model.encoder]
    modules_head = [model.decoder]
    if cfg["net"].get("aux_loss", False):
        modules_head.append(model.auxor)
    if cfg["net"].get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model.load_state_dict(torch.load(weight_path)['model_state'], strict=False)

    # prepare dataset for inference
    img_files = [f for f in os.listdir(INFER_PATH) if f.endswith('png')]

    model.eval()

    classes, palette = TIANWEN()

    for i, f in enumerate(img_files):

        img_path = os.path.join(INFER_PATH, f)
        img_name = os.path.basename(img_path)

        img_tensor, raw_img = preprocess_image(img_path, 'cuda')

        output= model(img_tensor)
        pred = output.data.max(1)[1].cpu().numpy()[0]

        color_image = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

        for class_index, color in enumerate(palette):
            color_image[pred == class_index] = color
        image_path = os.path.join(INFER_SAVE, img_name)
        img = Image.fromarray(color_image, 'RGB')
        img.save(image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Martian Rock Segmentation Testing")
    parser.add_argument("--config", type=str, default=r"F:\mars_rock\code\rock_segmentation\exps\ssw_augmentation_5\rock.yaml")
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()
    main(args)