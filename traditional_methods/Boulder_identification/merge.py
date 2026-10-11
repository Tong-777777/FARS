import os
import shutil

BASE_PATH = r'F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Arcpy\train'

OUT_IMG = os.path.join(BASE_PATH, 'image')
OUT_MASK = os.path.join(BASE_PATH, 'mask')

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_MASK, exist_ok=True)

idx = 1

for dir in os.listdir(BASE_PATH):

    dir_type = dir.split("_")[0]
    if dir_type == "PSP" or dir_type == "ESP" or dir_type == "TRA":

        png_dir = os.path.join(os.path.join(BASE_PATH, dir), 'png')

        mask_path =  os.path.join(os.path.join(BASE_PATH, dir), 'OUTPUT')

        png_files = sorted(
            f for f in os.listdir(png_dir)
            if f.lower().endswith(".png") and not f.lower().endswith("_mask.png")
        )

        for png_name in png_files:
            base_name = os.path.splitext(png_name)[0]  # crop_0_0
            mask_name = base_name + ".png"

            src_img = os.path.join(png_dir, png_name)
            src_mask = os.path.join(mask_path, mask_name)

            if not os.path.exists(src_mask):
                print(f"缺少 mask，跳过：{src_img}")
                continue

            dst_img = os.path.join(OUT_IMG, f"{idx}.png")
            dst_mask = os.path.join(OUT_MASK, f"{idx}.png")

            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_mask, dst_mask)

            print(f"复制完成：{idx}.png")
            idx += 1

    print(f"\n全部完成，共复制 {idx - 1} 对 image/mask")

