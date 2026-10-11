from PIL import Image
import numpy as np
import cv2
import os

def pgm_to_png(pgm_path, png_path):
    img = Image.open(pgm_path)
    arr = np.array(img)

    if arr.dtype == np.uint16:
        mx = int(arr.max()) if arr.size else 0
        if mx > 0:
            arr = (arr.astype(np.float32) / mx * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    Image.fromarray(arr).save(png_path)

def overlay_red_on_image(
    img_path: str,
    mask_pgm_path: str,
    out_path: str,
    alpha: float = 0.45,
    treat_nonzero_as_mask: bool = True,
):

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    m = cv2.imread(mask_pgm_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_pgm_path}")

    if m.ndim == 3:
        m = m[:, :, 0]

    H, W = img.shape[:2]
    if m.shape[0] != H or m.shape[1] != W:
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

    if treat_nonzero_as_mask:
        mask = (m == 128)

    overlay = img.copy()
    overlay[mask] = (0, 0, 255)  # BGR 红色

    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(blended, contours, -1, (0, 0, 255), 1)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ok = cv2.imwrite(out_path, blended)
    if not ok:
        raise RuntimeError(f"Failed to write: {out_path}")

    print("Saved:", out_path)

if __name__ == '__main__':
    image_path = r'F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Golombek_2008\PSP_005775_1245\png\101.png'
    mask_pgm_path = r'F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Golombek_2008\PSP_005775_1245\Rocks_output\Rocks-101.pgm'
    out_path = r'F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Golombek_2008\PSP_005775_1245\Rocks_output\Rocks-101.png'
    overlay_red_on_image(image_path, mask_pgm_path, out_path, alpha=0.6)
    # pgm_to_png(r"F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Golombek_2008\PSP_001391_2465\Rocks-test.pgm", r"F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Golombek_2008\PSP_001391_2465\rock.png")