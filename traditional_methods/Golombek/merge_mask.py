import os
import shutil

PATH = r'F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Golombek_2008\train'

OUT_IMG = os.path.join(PATH, "image")
OUT_MASK = os.path.join(PATH, "mask")

os.makedirs(OUT_MASK, exist_ok=True)
os.makedirs(OUT_IMG, exist_ok=True)

folders = os.listdir(PATH)

idx = 1

for folder in folders:
    folder_extension = folder.split("_")[0]

    if folder_extension == "ESP" or folder_extension == "PSP" or folder_extension == "TRA":

        folder_path = os.path.join(PATH, folder)
        png_dir = os.path.join(folder_path, 'png')
        pgm_dir = os.path.join(os.path.join(folder_path, 'pgm'), 'output')

        png_files = sorted(
            f for f in os.listdir(png_dir)
            if f.lower().endswith(".png") and not f.lower().endswith("_mask.png")
        )

        for png_name in png_files:
            base_name = os.path.splitext(png_name)[0]  # crop_0_0
            mask_name = "Rocks-" + base_name + ".png"

            src_img = os.path.join(png_dir, png_name)
            src_mask = os.path.join(pgm_dir, mask_name)

            if not os.path.exists(src_mask):
                print(f"缺少 mask，跳过：{src_img}")
                continue

            dst_img = os.path.join(OUT_IMG , f"{idx}.png")
            dst_mask = os.path.join(OUT_MASK, f"{idx}.png")

            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_mask, dst_mask)

            print(f"复制完成：{idx}.png")
            idx += 1

    print(f"\n全部完成，共复制 {idx - 1} 对 image/mask")
