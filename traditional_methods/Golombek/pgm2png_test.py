from PIL import Image
import os
import numpy as np

def one_image(pgm_path,png_path):
    img = Image.open(pgm_path)
    img.save(png_path)

    print("saved:", png_path)

def multi_image(input_path,output_path):

    files = os.listdir(input_path)
    for file in files:
        if file.endswith(".pgm"):

            file_name = file.split(".")[0]
            file_path = os.path.join(input_path,file)
            img = Image.open(file_path)
            arr = np.array(img)

            arr[arr == 255] = 0
            arr[arr == 128] = 255

            out = os.path.join(output_path, file_name + ".png")
            Image.fromarray(arr).save(out)

            print("saved:", out )

if __name__ == '__main__':

    input_path = r'F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Golombek_2008\train\TRA_000894_2475\pgm\Rocks_output'
    output_path = r'F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Golombek_2008\train\TRA_000894_2475\pgm\output'
    os.makedirs(output_path, exist_ok=True)
    multi_image(input_path,output_path)

    # pgm_path = r"F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Golombek_2008\ESP_017543_2330\Rocks_output\Rocks-1.pgm"
    # png_path = r"F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Golombek_2008\ESP_017543_2330\Rocks_output\Rocks-1.png"
    #
    #
    # img = Image.open(pgm_path)
    # img.save(png_path)
    #
    # print("saved:", png_path)