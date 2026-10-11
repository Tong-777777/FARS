import cv2
import os

def fast_run(input_image, output_image):

    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(output_image, img)

def run(dir_path, out_path):

    files = os.listdir(dir_path)
    for file in files:
        if file.endswith(".png"):
            fast_run(os.path.join(dir_path, file), os.path.join(out_path, file).replace(".png", ".pgm"))

if __name__ == '__main__':

    # input_image = r'F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Golombek_2008\PSP_001391_2465\crop_0_0.png'
    # output_image = r'F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Golombek_2008\PSP_001391_2465\crop_0_0.pgm'
    # fast_run(input_image, output_image)

    dir_path = r'F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Golombek_2008\train\TRA_000841_2460\png'
    out_path = r'F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Golombek_2008\train\TRA_000841_2460\pgm'

    os.makedirs(out_path, exist_ok=True)
    run(dir_path, out_path)