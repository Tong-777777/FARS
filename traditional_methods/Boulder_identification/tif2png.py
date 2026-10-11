import os
import rasterio
import numpy as np
import cv2

def find_output(path):
    floders = os.listdir(path)
    for floder in floders:
        if '_output' in floder:

            # name = floder.split('_')[0]
            name = floder.replace('_output', '')
            tif_path = os.path.join(path, floder)
            floder_path = os.path.join(path, floder)
            threshold_path = os.path.join(floder_path, 'threshold')

            tif_path = os.path.join(threshold_path, 'dark_70.tif')

            png_path = os.path.join(path, '{}.png'.format(name))

            if not os.path.exists(tif_path):
                print(f'File not found: {tif_path}')
                continue

            with rasterio.open(tif_path) as src:
                img = src.read(1)

            png_img = (img > 0).astype(np.uint8) * 255

            cv2.imwrite(png_path, png_img)
            print(f'Saved: {png_path}')



if __name__ == '__main__':

    path =  r'F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Arcpy\test\flat'
    find_output(path)