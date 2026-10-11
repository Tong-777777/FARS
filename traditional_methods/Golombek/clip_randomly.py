import os
import random
import rasterio
from rasterio.windows import Window

in_path = r"F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Golombek_2008\ESP_016526_2415\ESP_016526_2415_RED.NOMAP.JP2"
out_path = r"F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Golombek_2008\ESP_016526_2415\test.jp2"
patch = 1024

with rasterio.open(in_path) as src:
    w, h = src.width, src.height
    if w < patch or h < patch:
        raise ValueError(f"Image too small: {w}x{h}, cannot crop {patch}x{patch}")

    col_off = random.randint(0, w - patch)
    row_off = random.randint(0, h - patch)
    win = Window(col_off, row_off, patch, patch)

    data = src.read(window=win)
    profile = src.profile.copy()

    profile.update(
        width=patch,
        height=patch,
        transform=src.window_transform(win),
        driver="JP2OpenJPEG",
        tiled=False,
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)

print("Saved:", out_path)
print("Random window (col_off, row_off):", col_off, row_off)
