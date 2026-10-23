# FARS

Flat-to-Rough Adaptive Rock Segmentation (FARS) segments Martian rocks in HiRISE imagery without manual training annotations. Rule-based pseudo-labels from flat terrain supervise a segmentation network, while appearance statistics from unlabeled rough terrain guide style augmentation. Manual reference masks are used only for test.

## Implementation

The current implementation uses a local ResNet-101 encoder and a DeepLabV3+ decoder with semantic and boundary prediction heads. The semantic objective combines Symmetric Cross-Entropy and Dice losses. Boundary targets combine strong Sobel edges from the original flat image with a tolerance band around the pseudo-label contour.

The segmentation network receives the rough-style augmented image. Boundary supervision uses the original image restored from normalized tensors to intensities in [0, 1], before style augmentation. Augmentation preserves image geometry and the pseudo-label remains unchanged. The boundary tolerance radius and top-gradient ratio are currently set to 3 pixels and 0.2 in `utils/loss_helper.py`; the default boundary-loss weight is 0.1.

Use `train_fars.py` as the training entry point and `exps/FARS/rock.yaml` as its configuration.

## Environment

The manuscript reports PyTorch 1.12.1. Training and inference currently require a CUDA-enabled PyTorch environment. Install a compatible Torchvision version and the following core Python dependencies in your environment:

```bash
pip install numpy scipy Pillow PyYAML pandas tensorboard opencv-python
```

This list is based on the core imports, not a fully pinned reproduction environment. Optional data-processing and traditional-method scripts additionally use packages such as rasterio, scikit-image, matplotlib, scikit-learn, requests, beautifulsoup4 and imageio. GIS-BDQ requires ArcGIS/ArcPy; the Golombek wrapper requires a separately built ROSCO executable. These dependencies are not necessary for every training operation.

## Data preparation

Set `dataset.data_root` in `exps/FARS/rock.yaml` to a local directory with this structure:

```text
data_root/
  image/             Flat-terrain RGB or grayscale PNG images
  mask/              Corresponding binary pseudo-label PNG masks
  unlabeled_image/   Unlabeled rough-terrain PNG images
```

Images are converted to RGB. Mask pixel values greater than zero become rock (1), and zero becomes background (0). Each pseudo-label must have the same filename and dimensions as its flat image.

The current unlabeled reader obtains its filename list from `image/` and looks for matching names in `unlabeled_image/`. Consequently, this directory must contain those filenames. This is a reader constraint, not a requirement that flat and rough scenes be spatially paired. Independent rough-image sampling requires adapting the reader. The rough loader is constructed even when direct rough-reference augmentation is disabled.

The labeled reader currently returns full images without applying its constructed resize/flip/crop transforms. Prepare patches of compatible dimensions for batching; the YAML crop settings do not currently crop labeled inputs.

## Statistics-guided calibration

Run calibration on flat and rough training imagery only, excluding evaluation images:

```bash
python script/calibrate_rough_aug_ranges.py --flat_dir /path/to/flat/images --rough_dir /path/to/rough/images --max_size 1024 --seed 42
```

The script prints image counts followed by a `stage3` parameter block. It does not edit the training script or configuration automatically.

The original workflow is to manually copy the calibrated values into the defaults read inside `train()` in `train_fars.py`. Alternatively, add the printed `stage3` block to `exps/FARS/rock.yaml`: configuration values override those defaults. Preserve the calibrated values used for each experiment alongside its configuration.

The currently configurable fields are `rough_style_prob`, `contrast_range`, `brightness_range`, `gamma_range`, `noise_std_range`, `texture_strength_range`, `shadow_prob`, `blur_prob` and `sharpen_prob`. The calibration script also prints `shadow_rx_ratio_range`, `shadow_ry_ratio_range` and `shadow_dark_factor_range`

## Training

Run commands from the project root so local imports and the relative pretrained-weight path resolve correctly:

```bash
python train_fars.py --config exps/FARS/rock.yaml --seed 9527
```

Before training, check the data path and encoder initialization weights. The current default experiment configuration has two classes, 50 epochs, batch size 2, SGD learning rate 0.01, momentum 0.9, weight decay 0.0005 and polynomial decay power 0.9. Set these to the settings of the experiment being reproduced.

Logs are written under the configuration directory's `log/` folder. Checkpoints are written under `checkpoints/`. `ckpt.pth` stores the latest epoch; `ckpt_best.pth` is updated on improvement of training loss, not validation IoU.

## Inference

Edit `INFER_PATH` and `INFER_SAVE` in `infer.py` to select the input PNG directory and output directory. Then run:

```bash
python infer.py --config exps/FARS/rock.yaml
```

The script loads a checkpoint containing `best` in its filename from the experiment's `checkpoints/` folder, falling back to `ckpt.pth`. Use a folder containing only the intended best checkpoint to avoid ambiguous selection. Predictions are saved as black-background, white-rock RGB PNG masks. Supply `--config` explicitly: the script's default points to a different local project directory.

## Third-party code and acknowledgments

- GIS-BDQ scripts are based on [Boulder Identification](https://github.com/TimNagle-McNaughton/Boulder_Identification).
- Golombek-method scripts invoke [NASA/JPL ROSCO](https://github.com/nasa-jpl/ROSCO), which must be obtained and built separately. The wrapper currently points to an external `ExportRocksPGM.exe`; document its source/build steps when distributing the workflow.
- SSW-ROCK is implemented in `traditional_methods/SSW_rock/mian.py`.
