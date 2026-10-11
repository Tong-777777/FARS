import argparse
import logging
import os
import os.path as osp
import pprint
import time

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from models import ModelBuilder
from utils.data_helper import get_loader
from utils.loss_helper import get_criterion
from utils.lr_helper import get_optimizer, get_scheduler
from utils.para_helper import load_state
from utils.utils import *


def parse_model_outputs(outputs):
    """
    Parse outputs from different possible ModelBuilder implementations.

    The segmentation logits are always assumed to be the first output.
    """
    if isinstance(outputs, dict):
        for key in ["pred", "logits", "seg", "seg_logits", "out", "classification_head"]:
            if key in outputs:
                return outputs[key]
        raise RuntimeError("Could not parse segmentation logits from dict outputs.")

    if isinstance(outputs, (tuple, list)):
        if len(outputs) == 0:
            raise RuntimeError("Model returned an empty output list/tuple.")
        return outputs[0]

    return outputs


def parse_boundary_outputs(outputs):
    """
    Parse boundary logits from ModelBuilder outputs.

    For the current student ModelBuilder this is usually the second output:
        segmentation_logits, boundary_logits, domain_logits
    """
    if isinstance(outputs, dict):
        for key in ["boundary", "boundary_logits", "boundary_head"]:
            if key in outputs:
                return outputs[key]
        return None

    if isinstance(outputs, (tuple, list)):
        if len(outputs) > 1:
            return outputs[1]
        return None

    return None


def to_tensor_stats(mean, std, device, dtype):
    mean = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)
    return mean, std


def denormalize_image(x, mean, std):
    mean_t, std_t = to_tensor_stats(mean, std, x.device, x.dtype)
    return x * std_t + mean_t


def normalize_image(x, mean, std):
    mean_t, std_t = to_tensor_stats(mean, std, x.device, x.dtype)
    return (x - mean_t) / std_t


def normalize_map(x):
    return (x - x.mean(dim=[2, 3], keepdim=True)) / (
        x.std(dim=[2, 3], keepdim=True) + 1e-6
    )


def avg_blur(x, kernel_size):
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return F.avg_pool2d(
        x,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        count_include_pad=False,
    )


def apply_real_rough_reference(
    out,
    rough_ref,
    shading_strength_range=(0.00, 0.18),
    texture_strength_range=(0.00, 0.10),
    shadow_strength_range=(0.00, 0.25),
    color_match_weight=0.15,
):
    """
    Use real rough images to guide the existing rough-style perturbation.

    This avoids copying rough-image geometry into the label image. Only broad
    illumination/shadow fields and texture residuals are injected.
    """
    B, C, H, W = out.shape

    if rough_ref.shape[-2:] != (H, W):
        rough_ref = F.interpolate(
            rough_ref,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

    if rough_ref.size(0) != B:
        repeat = (B + rough_ref.size(0) - 1) // rough_ref.size(0)
        rough_ref = rough_ref.repeat(repeat, 1, 1, 1)[:B]

    rough_ref = rough_ref[torch.randperm(rough_ref.size(0), device=rough_ref.device)]
    rough_gray = rough_ref.mean(dim=1, keepdim=True)

    # 1. Real low-frequency terrain illumination/shading.
    low_kernel = max(9, min(H, W) // 4)
    rough_low = avg_blur(rough_gray, low_kernel)
    rough_low = normalize_map(rough_low)

    shading_strength = torch.empty(B, 1, 1, 1, device=out.device, dtype=out.dtype).uniform_(
        shading_strength_range[0], shading_strength_range[1]
    )
    out = out * (1.0 + rough_low * shading_strength)

    # 2. Real high-frequency rough terrain texture residual.
    texture_kernel = max(3, min(H, W) // 24)
    rough_base = avg_blur(rough_ref, texture_kernel)
    rough_residual = normalize_map(rough_ref - rough_base)

    texture_strength = torch.empty(B, 1, 1, 1, device=out.device, dtype=out.dtype).uniform_(
        texture_strength_range[0], texture_strength_range[1]
    )
    out = out + rough_residual * texture_strength

    # 3. Real dark-region mask from rough references, applied softly.
    shadow_q = torch.quantile(
        rough_gray.flatten(2),
        0.20,
        dim=2,
        keepdim=True,
    ).view(B, 1, 1, 1)
    shadow_mask = (shadow_q - rough_gray).clamp(min=0.0)
    shadow_mask = normalize_map(avg_blur(shadow_mask, max(7, min(H, W) // 18)))
    shadow_mask = torch.sigmoid(shadow_mask)

    shadow_strength = torch.empty(B, 1, 1, 1, device=out.device, dtype=out.dtype).uniform_(
        shadow_strength_range[0], shadow_strength_range[1]
    )
    out = out * (1.0 - shadow_mask * shadow_strength)

    # 4. Weak RGB mean/std matching to the rough reference.
    if color_match_weight > 0:
        out_mean = out.mean(dim=[2, 3], keepdim=True)
        out_std = out.std(dim=[2, 3], keepdim=True) + 1e-6
        ref_mean = rough_ref.mean(dim=[2, 3], keepdim=True)
        ref_std = rough_ref.std(dim=[2, 3], keepdim=True) + 1e-6
        matched = (out - out_mean) / out_std * ref_std + ref_mean
        out = (1.0 - color_match_weight) * out + color_match_weight * matched

    return out


def rough_style_augment(
    x,
    mean,
    std,
    rough_ref=None,
    p=0.7,
    contrast_range=(0.65, 1.45),
    brightness_range=(-0.18, 0.18),
    gamma_range=(0.70, 1.50),
    noise_std_range=(0.00, 0.08),
    texture_strength_range=(0.00, 0.12),
    shadow_prob=0.50,
    blur_prob=0.25,
    sharpen_prob=0.25,
    use_real_rough_reference=False,
    reference_shading_strength_range=(0.00, 0.18),
    reference_texture_strength_range=(0.00, 0.10),
    reference_shadow_strength_range=(0.00, 0.25),
    reference_color_match_weight=0.15,
):
    """
    Apply rough-terrain-like appearance perturbations to flat labeled images.

    Args:
        x:
            B x C x H x W normalized tensor.
        mean, std:
            Dataset normalization statistics.
        p:
            Probability of applying this augmentation to the current batch.

    Returns:
        Augmented normalized tensor.
    """
    if torch.rand(1, device=x.device).item() > p:
        return x

    # Work in [0, 1] image space for interpretable appearance augmentation.
    out = denormalize_image(x, mean, std).clamp(0.0, 1.0)

    B, C, H, W = out.shape

    # -------------------------------------------------
    # 1. Random contrast
    # -------------------------------------------------
    contrast = torch.empty(B, 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(
        contrast_range[0], contrast_range[1]
    )
    img_mean = out.mean(dim=[2, 3], keepdim=True)
    out = (out - img_mean) * contrast + img_mean

    # -------------------------------------------------
    # 2. Random brightness
    # -------------------------------------------------
    brightness = torch.empty(B, 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(
        brightness_range[0], brightness_range[1]
    )
    out = out + brightness

    # -------------------------------------------------
    # 3. Random gamma
    # -------------------------------------------------
    out = out.clamp(0.0, 1.0)
    gamma = torch.empty(B, 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(
        gamma_range[0], gamma_range[1]
    )
    out = out.clamp(1e-6, 1.0) ** gamma

    # -------------------------------------------------
    # 4. Rough texture perturbation
    # -------------------------------------------------
    if use_real_rough_reference and rough_ref is not None:
        rough_ref = denormalize_image(rough_ref, mean, std).clamp(0.0, 1.0)
        out = apply_real_rough_reference(
            out,
            rough_ref,
            shading_strength_range=reference_shading_strength_range,
            texture_strength_range=reference_texture_strength_range,
            shadow_strength_range=reference_shadow_strength_range,
            color_match_weight=reference_color_match_weight,
        )
    else:
        low_h = max(4, H // 16)
        low_w = max(4, W // 16)

        texture = torch.randn(B, 1, low_h, low_w, device=x.device, dtype=x.dtype)
        texture = F.interpolate(texture, size=(H, W), mode="bilinear", align_corners=False)

        texture = normalize_map(texture)

        texture_strength = torch.empty(B, 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(
            texture_strength_range[0], texture_strength_range[1]
        )
        out = out + texture * texture_strength

    # -------------------------------------------------
    # 5. Random noise
    # -------------------------------------------------
    noise_std = torch.empty(B, 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(
        noise_std_range[0], noise_std_range[1]
    )
    out = out + torch.randn_like(out) * noise_std

    # -------------------------------------------------
    # 6. Shadow-like dark elongated regions
    # -------------------------------------------------
    if torch.rand(1, device=x.device).item() < shadow_prob:
        yy = torch.arange(H, device=x.device, dtype=x.dtype).view(H, 1).expand(H, W)
        xx = torch.arange(W, device=x.device, dtype=x.dtype).view(1, W).expand(H, W)

        for b in range(B):
            cx = torch.empty(1, device=x.device, dtype=x.dtype).uniform_(0, W - 1)
            cy = torch.empty(1, device=x.device, dtype=x.dtype).uniform_(0, H - 1)

            rx = torch.empty(1, device=x.device, dtype=x.dtype).uniform_(W * 0.10, W * 0.45)
            ry = torch.empty(1, device=x.device, dtype=x.dtype).uniform_(H * 0.03, H * 0.18)

            angle = torch.empty(1, device=x.device, dtype=x.dtype).uniform_(-3.14159, 3.14159)
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)

            x_shift = xx - cx
            y_shift = yy - cy

            x_rot = x_shift * cos_a + y_shift * sin_a
            y_rot = -x_shift * sin_a + y_shift * cos_a

            mask = ((x_rot / rx) ** 2 + (y_rot / ry) ** 2) < 1.0

            dark_factor = torch.empty(1, device=x.device, dtype=x.dtype).uniform_(0.45, 0.85)
            out[b, :, mask] = out[b, :, mask] * dark_factor

    # -------------------------------------------------
    # 7. Blur or sharpen
    # -------------------------------------------------
    r = torch.rand(1, device=x.device).item()

    if r < blur_prob:
        out = F.avg_pool2d(out, kernel_size=3, stride=1, padding=1)
    elif r < blur_prob + sharpen_prob:
        blurred = F.avg_pool2d(out, kernel_size=3, stride=1, padding=1)
        out = out + 0.7 * (out - blurred)

    out = out.clamp(0.0, 1.0)

    # Return normalized image.
    out = normalize_image(out, mean, std)

    return out


def next_batch(loader_iter, loader):
    """
    Safely get the next batch. If the iterator is exhausted, restart it.
    """
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)

    return batch, loader_iter


def main(args):
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=True)

    ###########################
    # 1. output settings
    ###########################
    cfg["exp_path"] = osp.dirname(args.config)
    cfg["save_path"] = osp.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])
    cfg["log_path"] = osp.join(cfg["exp_path"], "log")
    flag_use_tb = cfg["saver"]["use_tb"]

    # Avoid FileExistsError in HPC/srun environments.
    os.makedirs(cfg["log_path"], exist_ok=True)
    os.makedirs(cfg["save_path"], exist_ok=True)

    logger, curr_timestr = setup_default_logging("global", cfg["log_path"])
    curr_timestr = curr_timestr.replace(":", "_")
    csv_path = os.path.join(cfg["log_path"], f"seg_stage3_rough_style_aug_{curr_timestr}_stat.csv")
    logger.info("{}".format(pprint.pformat(cfg)))

    if flag_use_tb:
        tb_logger = SummaryWriter(
            osp.join(cfg["log_path"], "events_stage3_rough_style_aug", curr_timestr)
        )
    else:
        tb_logger = None

    ###########################
    # 2. prepare model
    ###########################
    model = ModelBuilder(cfg["net"], is_student=True)

    modules_back = [model.encoder]
    modules_head = [model.decoder]

    if cfg["net"].get("aux_loss", False):
        modules_head.append(model.auxor)

    if hasattr(model, "boundary_head"):
        modules_head.append(model.boundary_head)

    if hasattr(model, "domain_head"):
        modules_head.append(model.domain_head)

    if cfg["net"].get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()

    ###########################
    # 3. loss & data
    ###########################
    loss_sup_hybrid = get_criterion(cfg, type="sup_hybrid_loss")
    loss_sup_boundary = get_criterion(cfg, type="boundary_loss")

    labeled_train_loader, rough_train_loader = get_loader(cfg)

    ##############################
    # 4. optimizer & scheduler
    ##############################
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]
    times = 10 if "pascal" in cfg["dataset"]["type"] else 1

    params_list = []

    for module in modules_back:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )

    for module in modules_head:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
        )

    optimizer = get_optimizer(params_list, cfg_optim)

    ######################################
    # 5. resume
    ######################################
    last_epoch = 0
    best_prec = None
    best_epoch = -1

    if cfg["saver"].get("auto_resume", False):
        latest_model = os.path.join(cfg["save_path"], "ckpt.pth")
        if not os.path.exists(latest_model):
            logger.info(f"No checkpoint found in '{latest_model}'")
        else:
            logger.info(f"Resume models from: '{latest_model}'")
            best_prec, last_epoch = load_state(
                latest_model, model, optimizer=optimizer, key="model_state"
            )

    lr_scheduler = get_scheduler(
        cfg_trainer,
        len(labeled_train_loader),
        optimizer,
        start_epoch=last_epoch
    )

    ######################################
    # 6. training loop
    ######################################
    logger.info("-------------------------- start stage-3 rough-style augmentation training --------------------------")

    for epoch in range(last_epoch, cfg_trainer["epochs"]):
        loss, hybrid_loss, boundary_loss = train(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_sup_hybrid=loss_sup_hybrid,
            loss_sup_boundary=loss_sup_boundary,
            labeled_train_loader=labeled_train_loader,
            rough_train_loader=rough_train_loader,
            epoch=epoch,
            tb_logger=tb_logger,
            logger=logger,
            cfg=cfg
        )

        if best_prec is None:
            best_prec = loss

        state = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_miou": best_prec,
        }

        if loss < best_prec:
            best_prec = loss
            best_epoch = epoch
            torch.save(state, osp.join(cfg["save_path"], "ckpt_best.pth"))

        torch.save(state, osp.join(cfg["save_path"], "ckpt.pth"))

        tmp_results = {
            "loss": loss,
            "hybrid_loss": hybrid_loss,
            "boundary_loss": boundary_loss,
            "best": best_prec,
            "best_epoch": best_epoch,
        }

        data_frame = pd.DataFrame(data=tmp_results, index=range(epoch, epoch + 1))

        if epoch > 0 and osp.exists(csv_path):
            data_frame.to_csv(csv_path, mode="a", header=None, index_label="epoch")
        else:
            data_frame.to_csv(csv_path, index_label="epoch")

    if tb_logger is not None:
        tb_logger.close()


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_sup_hybrid: callable,
    loss_sup_boundary: callable,
    labeled_train_loader: torch.utils.data.DataLoader,
    rough_train_loader: torch.utils.data.DataLoader,
    epoch: int,
    tb_logger: SummaryWriter,
    logger: logging.Logger,
    cfg: dict
    ) -> tuple:
    """
    Train one epoch with flat-to-rough style augmentation.

    Objective:
        L = L_sup(rough_style_augmented_flat_image, flat_label)

    No teacher, no pseudo label, no domain adversarial loss.
    """

    labeled_train_loader_iter = iter(labeled_train_loader)
    rough_train_loader_iter = iter(rough_train_loader)

    total_losses = AverageMeter(20)
    hybrid_losses = AverageMeter(20)
    boundary_losses = AverageMeter(20)
    batch_times = AverageMeter(20)
    learning_rates = AverageMeter(20)

    print_freq = max(1, len(labeled_train_loader) // 8)
    print_freq_lst = [i * print_freq for i in range(1, 8)]
    print_freq_lst.append(len(labeled_train_loader) - 1)

    stage3_cfg = cfg.get("stage3", cfg.get("stage2", {}))

    use_rough_style_aug = stage3_cfg.get("use_rough_style_aug", True)
    rough_style_prob = stage3_cfg.get("rough_style_prob", 0.7)

    contrast_range = tuple(stage3_cfg.get("contrast_range", [0.65, 1.45]))
    brightness_range = tuple(stage3_cfg.get("brightness_range", [-0.18, 0.18]))
    gamma_range = tuple(stage3_cfg.get("gamma_range", [0.70, 1.50]))
    noise_std_range = tuple(stage3_cfg.get("noise_std_range", [0.00, 0.08]))
    texture_strength_range = tuple(stage3_cfg.get("texture_strength_range", [0.00, 0.12]))

    shadow_prob = stage3_cfg.get("shadow_prob", 0.50)
    blur_prob = stage3_cfg.get("blur_prob", 0.25)
    sharpen_prob = stage3_cfg.get("sharpen_prob", 0.25)

    use_real_rough_reference = stage3_cfg.get("use_real_rough_reference", False)
    reference_shading_strength_range = tuple(
        stage3_cfg.get("reference_shading_strength_range", [0.00, 0.18])
    )
    reference_texture_strength_range = tuple(
        stage3_cfg.get("reference_texture_strength_range", [0.00, 0.10])
    )
    reference_shadow_strength_range = tuple(
        stage3_cfg.get("reference_shadow_strength_range", [0.00, 0.25])
    )
    reference_color_match_weight = stage3_cfg.get("reference_color_match_weight", 0.15)

    use_boundary_loss = stage3_cfg.get("use_boundary_loss", True)
    boundary_loss_weight = stage3_cfg.get("boundary_loss_weight", 0.1)

    mean = cfg["dataset"]["mean"]
    std = cfg["dataset"]["std"]

    model.train()

    for step in range(len(labeled_train_loader)):
        batch_start = time.time()

        i_iter = epoch * len(labeled_train_loader) + step

        if hasattr(lr_scheduler, "get_last_lr"):
            lr = lr_scheduler.get_last_lr()
        else:
            lr = lr_scheduler.get_lr()

        learning_rates.update(lr[0])
        lr_scheduler.step()

        labeled_batch, labeled_train_loader_iter = next_batch(
            labeled_train_loader_iter,
            labeled_train_loader
        )

        image, label, _ = labeled_batch
        image = image.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        rough_ref = None
        if use_real_rough_reference:
            rough_batch, rough_train_loader_iter = next_batch(
                rough_train_loader_iter,
                rough_train_loader
            )
            rough_ref = rough_batch[0].cuda(non_blocking=True)

        if use_rough_style_aug:
            image_in = rough_style_augment(
                image,
                mean=mean,
                std=std,
                rough_ref=rough_ref,
                p=rough_style_prob,
                contrast_range=contrast_range,
                brightness_range=brightness_range,
                gamma_range=gamma_range,
                noise_std_range=noise_std_range,
                texture_strength_range=texture_strength_range,
                shadow_prob=shadow_prob,
                blur_prob=blur_prob,
                sharpen_prob=sharpen_prob,
                use_real_rough_reference=use_real_rough_reference,
                reference_shading_strength_range=reference_shading_strength_range,
                reference_texture_strength_range=reference_texture_strength_range,
                reference_shadow_strength_range=reference_shadow_strength_range,
                reference_color_match_weight=reference_color_match_weight,
            )
            aug_flag = 1.0
        else:
            image_in = image
            aug_flag = 0.0

        outputs = model(image_in)
        pred = parse_model_outputs(outputs)
        boundary_logits = parse_boundary_outputs(outputs)

        hybrid_loss = loss_sup_hybrid(pred, label)

        if use_boundary_loss:
            if boundary_logits is None:
                raise RuntimeError(
                    "Boundary loss is enabled, but the model did not return boundary logits."
                )
            # Build targets from the original flat image, before rough-style
            # augmentation. Restore image intensities before computing gradients.
            boundary_image = denormalize_image(image, mean, std).clamp(0.0, 1.0)
            boundary_loss = loss_sup_boundary(boundary_logits, boundary_image, label)[0]
        else:
            boundary_loss = hybrid_loss.new_tensor(0.0)

        loss = hybrid_loss + boundary_loss_weight * boundary_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_losses.update(loss.item())
        hybrid_losses.update(hybrid_loss.item())
        boundary_losses.update(boundary_loss.item())

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        if step in print_freq_lst:
            logger.info(
                "Epoch/Iter [{:3}:{:4}/{:4}]. "
                "Loss:{total_loss.val:.3f}({total_loss.avg:.3f}) "
                "Hy:{hybrid_loss.val:.3f}({hybrid_loss.avg:.3f}) "
                "Bd:{boundary_loss.val:.3f}({boundary_loss.avg:.3f}) "
                "Aug:{aug_flag:.1f} "
                "Time:{batch_time.avg:.4f} "
                "LR:{lr.val:.5f}".format(
                    epoch,
                    step,
                    len(labeled_train_loader),
                    total_loss=total_losses,
                    hybrid_loss=hybrid_losses,
                    boundary_loss=boundary_losses,
                    aug_flag=aug_flag,
                    batch_time=batch_times,
                    lr=learning_rates,
                )
            )

            if tb_logger is not None:
                tb_logger.add_scalar("lr", learning_rates.avg, i_iter)
                tb_logger.add_scalar("loss", total_losses.avg, i_iter)
                tb_logger.add_scalar("hybrid_loss", hybrid_losses.avg, i_iter)
                tb_logger.add_scalar("boundary_loss", boundary_losses.avg, i_iter)
                tb_logger.add_scalar("stage3/rough_style_aug_flag", aug_flag, i_iter)

        del image, image_in, label, pred, boundary_logits
        if rough_ref is not None:
            del rough_ref

    return total_losses.avg, hybrid_losses.avg, boundary_losses.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flat-to-rough style augmentation for Mars rock segmentation")
    parser.add_argument(
        "--config",
        type=str,
        default="./exps/FARS/rock.yaml",
        help="Path to yaml config file."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=9527,
        help="Random seed."
    )

    args = parser.parse_args()
    main(args)
