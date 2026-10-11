import os
import shutil

def merge_datasets(source_dirs, target_dir):

    image_dir = os.path.join(target_dir, "image")
    mask_dir = os.path.join(target_dir, "mask")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    counter = 1

    for src in source_dirs:
        img_folder = os.path.join(src, "img")
        mask_folder = os.path.join(src, "mask_output")

        img_files = sorted(os.listdir(img_folder))

        for img_file in img_files:
            if not img_file.lower().endswith(".png"):
                continue

            base_name = os.path.splitext(img_file)[0]
            mask_file = f"{base_name}_mask.png"

            img_path = os.path.join(img_folder, img_file)
            mask_path = os.path.join(mask_folder, mask_file)

            if not os.path.exists(mask_path):
                print(f"⚠️ can not find: {mask_path}")
                continue

            new_img_name = f"{counter}.png"
            new_mask_name = f"{counter}.png"

            shutil.copy(img_path, os.path.join(image_dir, new_img_name))
            shutil.copy(mask_path, os.path.join(mask_dir, new_mask_name))

            counter += 1


if __name__ == "__main__":
    source_dirs = [r"F:\mars_rock\HiR\dataset\ESP_016526_2415_RED",
                   r"F:\mars_rock\HiR\dataset\ESP_049182_1285_RED",
                   r"F:\mars_rock\HiR\dataset\PSP_001391_2465_RED",
                   r'F:\mars_rock\HiR\dataset\TRA_000828_2495_RED',
                   r'F:\mars_rock\HiR\dataset\TRA_000841_2460_RED',
                   r'F:\mars_rock\HiR\dataset\TRA_000846_2475_RED',
                   r'F:\mars_rock\HiR\dataset\TRA_000894_2475_RED']
    
    target_dir = r"F:\mars_rock\HiR\train"
    merge_datasets(source_dirs, target_dir)
