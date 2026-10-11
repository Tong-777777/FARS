from datetime import datetime
import logging
import os,sys
import random
import numpy as np
import torch

def set_random_seed(seed, deterministic=False):
    """Set random seed."""
    # https://blog.csdn.net/qq_41118968/article/details/118937199
    # https://wenku.baidu.com/view/ac0a1720753231126edb6f1aff00bed5b9f373bd.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            # torch.use_deterministic_algorithms(True) # >=1.8
            # torch.set_deterministic(True) # before 1.8
            # RuntimeError: SpatialClassNLLCriterion_updateOutput does not have a deterministic implementation, but you set 'torch.set_deterministic(True)'
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'
    return datetime.today().strftime(fmt)

def setup_default_logging(name, save_path, level=logging.INFO,
                          format="[%(asctime)s][%(levelname)s] - %(message)s"):

    tmp_timestr = time_str()
    logger = logging.getLogger(name)
    logging.basicConfig(
        filename=os.path.join(save_path, r'seg_{tmp_timestr}.log'),
        format=format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format))
    logger.addHandler(console_handler)
    return logger, tmp_timestr

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target